from __future__ import division
import sys
import logging
import time
import collections
import copy
import random
import cPickle
import pyprind
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import pearsonr, spearmanr

import lasagne
import theano
import theano.tensor as T

logger = logging.getLogger(__name__)


class Model(object):
    def __init__(self,
                 data, W,
                 word2idx, idx2word,
                 save_path='.',
                 save_prefix='twitter',
                 max_seqlen=160,
                 batch_size=50,
                 # Network architecture:
                 encoder='rnn',
                 hidden_size=100,
                 n_recurrent_layers=1,
                 is_bidirectional=False,
                 dropout_out=0.,
                 dropout_in=0.,
                 # Learning parameters:
                 patience=10,
                 optimizer='adam',
                 lr=0.001,
                 lr_decay=0.95,
                 fine_tune_W=False,
                 fine_tune_M=False,
                 use_ntn=False,
                 k=4,
                 penalize_emb_norm=False,
                 penalize_emb_drift=False,
                 penalize_activations=False,
                 emb_penalty=0.001,
                 act_penalty=500):

        # Data parameters:
        self.data = data
        vocab_size = W.shape[0]
        embedding_size = W.shape[1]
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.embeddings = theano.shared(W, name='embeddings', borrow=True)
        self.save_path = save_path
        self.save_prefix = save_prefix
        self.max_seqlen = max_seqlen
        self.batch_size = batch_size
        # Learning parameters:
        self.hidden_size = hidden_size
        self.patience = patience
        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay = lr_decay
        self.fine_tune_W = fine_tune_W
        self.fine_tune_M = fine_tune_M
        self.use_ntn = use_ntn

        if penalize_emb_drift:
            self.orig_embeddings = theano.shared(W.copy(), name='orig_embeddings', borrow=True)

        self.timings = {'train': {}, 'val': {}, 'test': {}}  # store performance at each time-step

        self.c = T.imatrix('c')  # context word indices, matrix of shape (batch_size, max_seqlen)
        self.r = T.imatrix('r')  # response word indices, matrix of shape (batch_size, max_seqlen)
        self.y = T.ivector('y')  # flag for each <context, response> pair within the batch, vector of size (batch_size)
        self.c_mask = T.fmatrix('c_mask')  # mask for contexts, same size as c
        self.r_mask = T.fmatrix('r_mask')  # mask for responses, same size as r
        self.c_seqlen = T.ivector('c_seqlen')  # length of each context within the batch, vector of size (batch_size)
        self.r_seqlen = T.ivector('r_seqlen')  # length of each response within the batch, vector of size (batch_size)

        zero_vec_tensor = T.fvector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        self.set_zero = theano.function(inputs=[zero_vec_tensor],
                                        updates=[(self.embeddings,
                                                  T.set_subtensor(self.embeddings[0,:], zero_vec_tensor))])

        if use_ntn:
            self.U = theano.shared(np.random.uniform(-0.01, 0.01, size=(k,)).astype(theano.config.floatX), borrow=True)
            self.V = theano.shared(np.random.uniform(-0.01, 0.01, size=(k, 2*hidden_size)).astype(theano.config.floatX), borrow=True)
            self.b = theano.shared(np.random.uniform(-0.01, 0.01, size=(k,)).astype(theano.config.floatX), borrow=True)
            self.M = theano.shared(np.random.uniform(-0.01, 0.01, size=(k, hidden_size, hidden_size)).astype(theano.config.floatX), borrow=True)
            self.f = lasagne.nonlinearities.tanh
        else:
            self.M = theano.shared(np.eye(hidden_size).astype(theano.config.floatX), borrow=True)
            self.b = theano.shared(0., borrow=True)

        # context word embeddings: Tensor of shape (batch_size, max_seqlen, embedding_size)
        c_input = self.embeddings[self.c.flatten()].reshape((self.c.shape[0], self.c.shape[1], self.embeddings.shape[1]))
        # response word embeddings: Tensor of shape (batch_size, max_seqlen, embedding_size)
        r_input = self.embeddings[self.r.flatten()].reshape((self.r.shape[0], self.r.shape[1], self.embeddings.shape[1]))

        # Variables to feed into the network:
        # input layer of the network (will be either `c_input` or `r_input`)
        l_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, embedding_size))
        # mask to apply on input layer (will be either `c_mask` or `r_mask`)
        l_mask = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen))

        if is_bidirectional:
            l_fwd = l_in
            l_bck = l_in
            if encoder == 'lstm':
                logger.info("Building a bidirectional LSTM model")
                for _ in xrange(n_recurrent_layers):
                    if dropout_in > 1e-9 and dropout_in < 1.-1e-9:
                        l_fwd = lasagne.layers.DropoutLayer(incoming=l_fwd, p=dropout_in)
                    l_fwd = lasagne.layers.LSTMLayer(incoming=l_fwd,
                                                     num_units=hidden_size,  # number of hidden units in the layer
                                                     mask_input=l_mask,
                                                     forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(2.0)),  # init forget gate bias to 2
                                                     backwards=False,   # forward pass
                                                     grad_clipping=10,  # avoid exploding gradients
                                                     learn_init=True)   # initial hidden values are learned
                    if dropout_out > 1e-9 and dropout_out < 1.-1e-9:
                        l_fwd = lasagne.layers.DropoutLayer(incoming=l_fwd, p=dropout_out)

                    if dropout_in > 1e-9 and dropout_in < 1.-1e-9:
                        l_bck = lasagne.layers.DropoutLayer(incoming=l_bck, p=dropout_in)
                    l_bck = lasagne.layers.LSTMLayer(incoming=l_bck,
                                                     num_units=hidden_size,  # number of hidden units in the layer
                                                     mask_input=l_mask,
                                                     forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(2.0)),  # init forget gate bias to 2
                                                     backwards=True,    # backward pass
                                                     grad_clipping=10,  # avoid exploding gradients
                                                     learn_init=True)   # initial hidden values are learned
                    if dropout_out > 1e-9 and dropout_out < 1.-1e-9:
                        l_bck = lasagne.layers.DropoutLayer(incoming=l_bck, p=dropout_out)
            elif encoder == 'gru':
                logger.info("Building a bidirectional GRU model")
                for _ in xrange(n_recurrent_layers):
                    if dropout_in > 1e-9 and dropout_in < 1.-1e-9:
                        l_fwd = lasagne.layers.DropoutLayer(incoming=l_fwd, p=dropout_in)
                    l_fwd = lasagne.layers.GRULayer(incoming=l_fwd,
                                                    num_units=hidden_size,  # number of hidden units in the layer
                                                    mask_input=l_mask,
                                                    backwards=False,   # forward pass
                                                    grad_clipping=10,  # avoid exploding gradients
                                                    learn_init=True)   # initial hidden values are learned
                    if dropout_out > 1e-9 and dropout_out < 1.-1e-9:
                        l_fwd = lasagne.layers.DropoutLayer(incoming=l_fwd, p=dropout_out)

                    if dropout_in > 1e-9 and dropout_in < 1.-1e-9:
                        l_bck = lasagne.layers.DropoutLayer(incoming=l_bck, p=dropout_in)
                    l_bck = lasagne.layers.GRULayer(incoming=l_bck,
                                                    num_units=hidden_size,  # Number of hidden units in the layer
                                                    mask_input=l_mask,
                                                    backwards=True,    # backward pass
                                                    grad_clipping=10,  # avoid exploding gradients
                                                    learn_init=True)   # initial hidden values are learned
                    if dropout_out > 1e-9 and dropout_out < 1.-1e-9:
                        l_bck = lasagne.layers.DropoutLayer(incoming=l_bck, p=dropout_out)
            elif encoder == 'rnn':
                logger.info("Building a bidirectional RNN model")
                for _ in xrange(n_recurrent_layers):
                    if dropout_in > 1e-9 and dropout_in < 1.-1e-9:
                        l_fwd = lasagne.layers.DropoutLayer(incoming=l_fwd, p=dropout_in)
                    l_fwd = lasagne.layers.RecurrentLayer(incoming=l_fwd,
                                                          num_units=hidden_size,  # number of hidden units in the layer
                                                          mask_input=l_mask,
                                                          nonlinearity=lasagne.nonlinearities.tanh,  # Nonlinearity to apply when computing new state
                                                          W_in_to_hid=lasagne.init.Orthogonal(),     # Initializer for input-to-hidden weight matrix
                                                          W_hid_to_hid=lasagne.init.Orthogonal(),    # Initializer for hidden-to-hidden weight matrix
                                                          backwards=False,   # forward pass
                                                          grad_clipping=10,  # avoid exploding gradients
                                                          learn_init=True)   # initial hidden values are learned
                    if dropout_out > 1e-9 and dropout_out < 1.-1e-9:
                        l_fwd = lasagne.layers.DropoutLayer(incoming=l_fwd, p=dropout_out)

                    if dropout_in > 1e-9 and dropout_in < 1.-1e-9:
                        l_bck = lasagne.layers.DropoutLayer(incoming=l_bck, p=dropout_in)
                    l_bck = lasagne.layers.RecurrentLayer(incoming=l_bck,
                                                          num_units=hidden_size,  # number of hidden units in the layer
                                                          mask_input=l_mask,
                                                          nonlinearity=lasagne.nonlinearities.tanh,  # Nonlinearity to apply when computing new state
                                                          W_in_to_hid=lasagne.init.Orthogonal(),     # Initializer for input-to-hidden weight matrix
                                                          W_hid_to_hid=lasagne.init.Orthogonal(),    # Initializer for hidden-to-hidden weight matrix
                                                          backwards=True,    # backward pass
                                                          grad_clipping=10,  # avoid exploding gradients
                                                          learn_init=True)   # initial hidden values are learned
                    if dropout_out > 1e-9 and dropout_out < 1.-1e-9:
                        l_bck = lasagne.layers.DropoutLayer(incoming=l_bck, p=dropout_out)
            else:
                raise ValueError("Unknown encoder %s", encoder)
            # concatenate forward and backward layers
            self.l_out = lasagne.layers.ConcatLayer([l_fwd, l_bck])

        else:
            l_recurrent = l_in
            if encoder == 'lstm':
                logger.info("Building an LSTM model")
                for _ in xrange(n_recurrent_layers):
                    if dropout_in > 1e-9 and dropout_in < 1.-1e-9:
                        l_recurrent = lasagne.layers.DropoutLayer(incoming=l_recurrent, p=dropout_in)
                    l_recurrent = lasagne.layers.LSTMLayer(incoming=l_recurrent,
                                                           num_units=hidden_size,  # number of hidden units in the layer
                                                           mask_input=l_mask,
                                                           forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(2.0)),  # init forget gate bias to 2
                                                           grad_clipping=10,       # avoid exploding gradients
                                                           learn_init=True)        # initial hidden values are learned
                    if dropout_out > 1e-9 and dropout_out < 1.-1e-9:
                        l_recurrent = lasagne.layers.DropoutLayer(incoming=l_recurrent, p=dropout_out)
            elif encoder == 'gru':
                logger.info("Building a GRU model")
                for _ in xrange(n_recurrent_layers):
                    if dropout_in > 1e-9 and dropout_in < 1.-1e-9:
                        l_recurrent = lasagne.layers.DropoutLayer(incoming=l_recurrent, p=dropout_in)
                    l_recurrent = lasagne.layers.GRULayer(incoming=l_recurrent,
                                                          num_units=hidden_size,  # number of hidden units in the layer
                                                          mask_input=l_mask,
                                                          grad_clipping=10,       # avoid exploding gradients
                                                          learn_init=True)        # initial hidden values are learned
                    if dropout_out > 1e-9 and dropout_out < 1.-1e-9:
                        l_recurrent = lasagne.layers.DropoutLayer(incoming=l_recurrent, p=dropout_out)
            elif encoder == 'rnn':
                logger.info("Building an RNN model")
                for _ in xrange(n_recurrent_layers):
                    if dropout_in > 1e-9 and dropout_in < 1.-1e-9:
                        l_recurrent = lasagne.layers.DropoutLayer(incoming=l_recurrent, p=dropout_in)
                    l_recurrent = lasagne.layers.RecurrentLayer(incoming = l_recurrent,
                                                                num_units = hidden_size,  # number of hidden units in the layer
                                                                mask_input = l_mask,
                                                                nonlinearity = lasagne.nonlinearities.tanh,  # Nonlinearity to apply when computing new state
                                                                W_in_to_hid = lasagne.init.Orthogonal(),     # Initializer for input-to-hidden weight matrix
                                                                W_hid_to_hid = lasagne.init.Orthogonal(),    # Initializer for hidden-to-hidden weight matrix
                                                                grad_clipping = 10,  # avoid exploding gradients
                                                                learn_init = True)   # initial hidden values are learned
                    if dropout_out > 1e-9 and dropout_out < 1.-1e-9:
                        l_recurrent = lasagne.layers.DropoutLayer(incoming=l_recurrent, p=dropout_out)
            else:
                raise ValueError("Unknown encoder %s", encoder)

            self.l_out = l_recurrent

        # Hidden states of network at each time step after feeding it the context:
        h_context = lasagne.layers.helper.get_output(
            layer_or_layers=self.l_out,
            inputs={l_in: c_input, l_mask: self.c_mask},  # set parameters of the network units: l_in and l_mask
            deterministic=False  # if True no dropout, default is False, better to be True at test time
        )
        # Hidden states of network at each time step after feeding it the response:
        h_response = lasagne.layers.helper.get_output(
            layer_or_layers=self.l_out,
            inputs={l_in: r_input, l_mask: self.r_mask},  # set parameters of the network units: l_in and l_mask
            deterministic=False  # if True no dropout, default is False, better to be True at test time
        )
        # Encoding of the context: take the encoding at the end of the context (self.c_seqlen)
        self.e_context = h_context[T.arange(batch_size), self.c_seqlen].reshape((batch_size, hidden_size))
        # Encoding of the response: take the encoding at the end of the response (self.r_seqlen)
        self.e_response = h_response[T.arange(batch_size), self.r_seqlen].reshape((batch_size, hidden_size))

        if use_ntn:
            dp = T.concatenate([T.batched_dot(self.e_context, T.dot(self.e_response, self.M[i])) for i in xrange(k)], axis=1)
            dp += T.concatenate([self.e_context, self.e_response], axis=1).dot(self.V.T) + self.b
            dp = self.f(dp).dot(self.U)
        else:
            dp = T.batched_dot(self.e_context, T.dot(self.e_response, self.M.T))  # (batch_size,)
            dp += self.b  # bias term

        o = T.nnet.sigmoid(dp)
        o = T.clip(o, 1e-7, 1.0-1e-7)  # clip output probabilities

        self.cost = T.nnet.binary_crossentropy(o, self.y).mean()  # used in `self.train_model()`
        self.probas = T.concatenate([(1-o).reshape((-1,1)), o.reshape((-1,1))], axis=1)  # used in `self.get_probas()`
        self.pred = T.argmax(self.probas, axis=1)  # used in `self.get_pred()`
        self.errors = T.sum(T.neq(self.pred, self.y))  # used in `self.get_loss()`

        if penalize_emb_norm:
            self.cost += emb_penalty * (self.embeddings ** 2).sum()
        if penalize_emb_drift:
            self.cost += emb_penalty * ((self.embeddings - self.orig_embeddings) ** 2).sum()
        if penalize_activations:
            self.cost += act_penalty * T.stack([((h_context[:,i] - h_context[:,i+1]) ** 2).sum(axis=1).mean() for i in xrange(max_seqlen-1)]).mean()
            self.cost += act_penalty * T.stack([((h_response[:,i] - h_response[:,i+1]) ** 2).sum(axis=1).mean() for i in xrange(max_seqlen-1)]).mean()

        self.update_params()

    def update_params(self):
        ###
        # Get all parameters of the network
        ###
        params = lasagne.layers.get_all_params(self.l_out)
        if self.use_ntn:
            params += [self.U, self.V, self.M, self.b]
        if self.fine_tune_W:
            params += [self.embeddings]
        if self.fine_tune_M and not self.use_ntn:
            params += [self.M, self.b]

        total_params = sum([p.get_value().size for p in params])
        logger.info("total_params: %d" % total_params)

        ###
        # Get parameter updates according to the optimizer
        ###
        updates = None
        if self.optimizer == 'adam':
            updates = lasagne.updates.adam(loss_or_grads=self.cost, params=params, learning_rate=self.lr)
        elif self.optimizer == 'adadelta':
            updates = lasagne.updates.adadelta(loss_or_grads=self.cost, params=params, rho=self.lr_decay)
        elif self.optimizer == 'adadegrad':
            updates = lasagne.updates.adagrad(loss_or_grads=self.cost, params=params)
        elif self.optimizer == 'sgd':
            updates = lasagne.updates.sgd(loss_or_grads=self.cost, params=params, learning_rate=self.lr)
        elif self.optimizer == 'rmsprop':
            lasagne.updates.rmsprop(loss_or_grads=self.cost, params=params, rho=self.lr_decay)
        else:
            raise 'Unsupported optimizer: %s' % self.optimizer

        ###
        # Initialize shared variables
        ###
        self.shared_data = {}
        for key in ['c', 'r']:
            self.shared_data[key] = theano.shared(np.zeros((self.batch_size, self.max_seqlen), dtype=np.int32), borrow=True)
        for key in ['c_mask', 'r_mask']:
            self.shared_data[key] = theano.shared(np.zeros((self.batch_size, self.max_seqlen), dtype=theano.config.floatX), borrow=True)
        for key in ['y', 'c_seqlen', 'r_seqlen']:
            self.shared_data[key] = theano.shared(np.zeros((self.batch_size,), dtype=np.int32), borrow=True)

        givens = {
            self.c: self.shared_data['c'],
            self.r: self.shared_data['r'],
            self.y: self.shared_data['y'],
            self.c_seqlen: self.shared_data['c_seqlen'],
            self.r_seqlen: self.shared_data['r_seqlen'],
            self.c_mask: self.shared_data['c_mask'],
            self.r_mask: self.shared_data['r_mask']
        }

        logger.info("compiling theano functions...")
        self.get_response_emb = theano.function(
            inputs=[],
            outputs=self.e_response,  # (batch_size, hidden_size)
            givens=givens,
            on_unused_input='ignore',
            name='get_response_emb'
        )
        self.get_context_emb = theano.function(
            inputs=[],
            outputs=self.e_context,  # (batch_size, hidden_size)
            givens=givens,
            on_unused_input='ignore',
            name='get_context_emb'
        )
        self.train_model = theano.function(
            inputs=[],
            outputs=self.cost,
            updates=updates,
            givens=givens,
            on_unused_input='ignore',
            name='train_model'
        )
        self.get_probas = theano.function(
            inputs=[],
            outputs=self.probas,
            givens=givens,
            on_unused_input='ignore',
            name='get_probas'
        )
        self.get_pred = theano.function(
            inputs=[],
            outputs=self.pred,
            givens=givens,
            on_unused_input='ignore',
            name='get_pred'
        )
        self.get_loss = theano.function(
            inputs=[],
            outputs=self.errors,
            givens=givens,
            on_unused_input='ignore',
            name='get_loss'
        )

    def string2indices(self, p_str, bpe=None):
        """
        Lookup dictionary from word to index to retrieve the list of indices from a string of words.
        If bpe is present, will automatically convert regular string p_str to bpe formatted string.
        :param p_str: string of words corresponding to indices.
        :param bpe: byte pair encoding object from apply_bpe.py
        :return: a new list of indices corresponding to the given string of words.
        """
        idx = []
        if bpe:
            bpe_string = bpe.segment(p_str.strip())  # convert from regular to bpe format
            for w in bpe_string.split():
                idx.append(self.word2idx.get(w, self.word2idx['**unknown**']))
        else:
            for w in p_str.strip().split():
                idx.append(self.word2idx.get(w, self.word2idx['**unknown**']))
        return idx

    def indices2string(self, p_indices, bpe_separator=None):
        """
        Lookup dictionary from word to index to retrieve the list of words from a list of indices.
        :param p_indices: list of indices corresponding to words.
        :param bpe_separator: token separator from BPE pre-processing
        :return: a new string corresponding to the given list of indices.
        """
        string = []
        for idx in p_indices:
            string.append(self.idx2word.get(idx, '**unknown**'))
        
        if bpe_separator:
            return ' '.join(string).replace(bpe_separator+' ', '')
        else:
            return ' '.join(string)
        

    def get_batch(self, dataset, index, truncate_type):
        """
        Get description of the data at a given batch index.
        :param dataset: array of tokens (can represent either contexts or responses)
        :param index: current index of the batch
        :param truncate_type: 'c' will truncate backward & 'r' will truncate forward
        :return: batch data, sequence length for each element in batch, mask for each element in batch
        """
        seqlen = np.zeros((self.batch_size,), dtype=np.int32)
        mask = np.zeros((self.batch_size, self.max_seqlen), dtype=theano.config.floatX)
        batch = np.zeros((self.batch_size, self.max_seqlen), dtype=np.int32)
        data = dataset[index*self.batch_size: (index+1) * self.batch_size]
        for i, row in enumerate(data):
            if truncate_type == 'c':
                row = row[-self.max_seqlen:]  # cut start of context if longer than max_seqlen
            elif truncate_type == 'r':
                row = row[:self.max_seqlen]  # cut end of response if longer than max_seqlen
            else:
                logger.error("truncate_type %s unknown" % truncate_type)

            batch[i, 0:len(row)] = row  # put the data into our batch
            seqlen[i] = len(row) - 1  # max index for that batch element
            mask[i, 0:len(row)] = 1  # put a '1' on the sequence, 0 else where
        return batch, seqlen, mask

    def set_shared_variables(self, dataset, index, training):
        """
        Set shared variables for that batch index.
        Set context and response: value, mask and length
        :param dataset: dictionary of contexts, responses, and flags
        :param index: batch index to work on
        :param training: if true, dataset['y'] is required, else not used
        :return: None
        """
        c, c_seqlen, c_mask = self.get_batch(dataset['c'], index, 'c')
        r, r_seqlen, r_mask = self.get_batch(dataset['r'], index, 'r')
        if training:
            y = np.array(dataset['y'][index*self.batch_size:(index+1)*self.batch_size], dtype=np.int32)
            self.shared_data['y'].set_value(y)
        self.shared_data['c'].set_value(c)
        self.shared_data['r'].set_value(r)
        self.shared_data['c_seqlen'].set_value(c_seqlen)
        self.shared_data['r_seqlen'].set_value(r_seqlen)
        self.shared_data['c_mask'].set_value(c_mask)
        self.shared_data['r_mask'].set_value(r_mask)

    def compute_response_embeddings(self, dataset, index):
        """
        :param dataset: list of responses
        :param index: current batch index
        :return: list of embeddings for that batch
        """
        r, r_seqlen, r_mask = self.get_batch(dataset, index, 'r')
        self.shared_data['r'].set_value(r)
        self.shared_data['r_seqlen'].set_value(r_seqlen)
        self.shared_data['r_mask'].set_value(r_mask)
        return self.get_response_emb()  # (batch_size, hidden_size)

    def compute_context_embeddings(self, dataset, index, training=False):
        """
        :param dataset: list of contexts
        :param index: current batch index
        :return: list of embeddings for that batch
        """
        c, c_seqlen, c_mask = self.get_batch(dataset, index, 'c')
        self.shared_data['c'].set_value(c)
        self.shared_data['c_seqlen'].set_value(c_seqlen)
        self.shared_data['c_mask'].set_value(c_mask)
        return self.get_context_emb()  # (batch_size, hidden_size)

    def compute_probas(self, dataset, index, training=False):
        """
        :param dataset: dictionary of contexts, responses, flags
        :param index: current batch index
        :param training: if true, dataset['y'] is required, else assumed not required
        :return: array of probability of being a good response for each context-response pair in the batch
        """
        self.set_shared_variables(dataset, index, training)
        return self.get_probas()[:,1]  # [:, 1] <=> Pr(y=1)

    def compute_pred(self, dataset, index, training=False):
        """
        :param dataset: dictionary of contexts, responses, flags
        :param index: current batch index
        :param training: if true, dataset['y'] is required, else assumed not required
        :return: array of predictions for each context-response pair in that batch
        """
        self.set_shared_variables(dataset, index, training)
        return self.get_pred()

    def compute_loss(self, dataset, index, training=True):
        """
        :param dataset: dictionary of contexts, responses, flags
        :param index: current batch index
        :param training: if true, dataset['y'] is required, else assumed not required
        :return: number of prediction not equal to y in the batch
        """
        self.set_shared_variables(dataset, index, training)
        return self.get_loss()

    def save_performance(self, scope, model_name, perf):
        """
        Save the performance into self.timings
        :param scope: :param scope: either "train", "val" or "test" sets.
        :param model_name: current model being tested against
        :param perf: the discriminator performance to save
        :return: None
        """
        assert scope in self.timings.keys()
        if model_name not in self.timings[scope]:
            self.timings[scope][model_name] = []
        self.timings[scope][model_name].append(perf)

    def compute_and_save_performance_models(self, scope, compute_recalls=False):
        """
        Measure the accuracy of the current Discriminator on each dialogue model.
        :param scope: either "train", "val" or "test" sets.
        :param compute_recalls: decide if compute recall@(10,5,1)
        :return: array of discriminator accuracies for each model.
        """
        assert scope in ["train", "val", "test"]

        # Reformat data to get: scope --> <dialogue_model_name> : {'c':[], 'r':[], 'y':[]}
        if not hasattr(self, 'data_by_models'):
            self.data_by_models = {}
        if scope not in self.data_by_models:
            self.data_by_models[scope] = {}
        # If we are missing any model_name for this scope, check which one is it and add it to the data
        missing_models = [name for name in self.data[scope]['id'] if name not in self.data_by_models[scope]]
        if len(missing_models) > 0:
            # make sure we have all model_name in the data
            for idx, model_name in enumerate(self.data[scope]['id']):
                # add data to the missing model_name only
                if model_name in missing_models:
                    if model_name not in self.data_by_models[scope]:
                        self.data_by_models[scope][model_name] = {'c': [], 'r': [], 'y': []}
                    self.data_by_models[scope][model_name]['c'].append(self.data[scope]['c'][idx])
                    self.data_by_models[scope][model_name]['r'].append(self.data[scope]['r'][idx])
                    self.data_by_models[scope][model_name]['y'].append(self.data[scope]['y'][idx])

        performances = []
        for model_name, data in self.data_by_models[scope].iteritems():
            logger.info("evaluating %s" % model_name)
            # Pad data set to be divisible by batch_size
            length = len(data['y'])
            to_pad = self.batch_size - (length % self.batch_size)
            for pad_idx in range(to_pad):
                data['c'].append(data['c'][pad_idx])
                data['r'].append(data['r'][pad_idx])
                data['y'].append(data['y'][pad_idx])
            # Compute performance:
            n_batches = len(data['y']) // self.batch_size
            losses = [self.compute_loss(data, i) for i in xrange(n_batches)]  # number of wrong predictions for each batch
            # ignore extra losses from padding
            losses = losses[:length]
            data['c'] = data['c'][:length]
            data['r'] = data['r'][:length]
            data['y'] = data['y'][:length]

            perf = 1 - np.sum(losses) / len(data['y'])  # 1 - total number of errors / total number of examples
            performances.append(perf)
            logger.info('%s_perf: %.9f%%' % (scope, perf*100))
            self.save_performance(scope, model_name, perf)  # save performance of the discriminator for that model under this scope
        
        if compute_recalls:
            if scope == 'train':
                logger.warning("train set has been shuffled, unable to compute recalls!")
            else:
                logger.info("")
                logger.debug("verifying data format for recalls...")
                # WARNING: assumes data[scope] has been created like so: [1 true response, 9 false, 1 true, 9 false, ...]
                # -> no model responses, only 1 true vs 9 random, data not shuffled!
                test_size = 10
                try:
                    for idx in xrange(0, len(self.data[scope]['c']), test_size):
                        # If size of data not divisible by `test_size`, ignore the last few examples
                        if idx+test_size-1 >= len(self.data[scope]['c']):
                            break
                        # Make sure that true response is always the first
                        assert self.data[scope]['y'][idx] == 1, "data[%s][y][%d]=%s != 1" % (scope, idx, self.data[scope]['y'][idx])
                        assert self.data[scope]['id'][idx] == 'true', "data[%s][id][%d]=%s != true" % (scope, idx, self.data[scope]['id'][idx])
                        for t in range(test_size)[1:]:
                            # Make sure that contexts are the same `test_size` times in a row
                            assert self.data[scope]['c'][idx] == self.data[scope]['c'][idx+t], "contexts don't match! has data been shuffled?"
                            # Make sure that remaining 9 responses are false random
                            assert self.data[scope]['y'][idx+t] == 0, "data[%s][y][%d]=%s != 0" % (scope, idx+t, self.data[scope]['y'][idx+t])
                            assert self.data[scope]['id'][idx+t] == 'rand', "data[%s][id][%d]=%s != rand" % (scope, idx+t, self.data[scope]['id'][idx+t])
                except AssertionError as e:
                    logger.error("AssertionError: %s" % e)
                    logger.error("Cannot compute recalls")
                    return performances, None
                
                # Pad data set to be divisible by batch_size
                length = len(self.data[scope]['y'])
                to_pad = self.batch_size - (length % self.batch_size)
                if to_pad >= test_size:  # padding less than test_size won't change recalls since truncated in compute_recall_ks
                    for pad_idx in range(to_pad):
                        self.data[scope]['c'].append(self.data[scope]['c'][pad_idx])
                        self.data[scope]['r'].append(self.data[scope]['r'][pad_idx])
                        self.data[scope]['y'].append(self.data[scope]['y'][pad_idx])
                        self.data[scope]['id'].append(self.data[scope]['id'][pad_idx])
                # compute probabilities
                n_batches = len(self.data[scope]['y']) // self.batch_size
                logger.debug("ok. computing %d probabilities now..." % (n_batches*self.batch_size))
                probas = np.concatenate([self.compute_probas(self.data[scope], i) for i in xrange(n_batches)])
                # if padded, ignore extra probabilities
                if len(probas) > length:
                    probas = probas[:length]
                    self.data[scope]['c'] = self.data[scope]['c'][:length]
                    self.data[scope]['r'] = self.data[scope]['r'][:length]
                    self.data[scope]['y'] = self.data[scope]['y'][:length]
                    self.data[scope]['id'] = self.data[scope]['id'][:length]
                # compute recalls
                recall_k = self.compute_recall_ks(probas)
                return performances, recall_k

        return performances, None

    def test(self):
        """
        Compute performances on test set
        :return: None
        """
        # Compute TEST performance:
        # evaluation for each model id in data['test']['id']
        test_perfs, test_recalls = self.compute_and_save_performance_models("test", compute_recalls=True)
        test_perf = np.average(test_perfs)
        logger.info("")
        logger.info('Average test_perf=%.9f%%' % (test_perf*100))

    def plot_score_per_length(self, scope='train'):
        """
        Compute the probability of each response being true and plot according to length of the response
        :param scope: scope of the data to look at: 'train' or 'val' or 'test'
        :return: None, plot instead.
        """
        logger.info("")
        logger.info("Get probabilities of each response in data[%s]..." % scope)
        n_batches = len(self.data[scope]['r']) // self.batch_size
        length_to_scores = {}
        all_lengths = [len(resp) for resp in self.data[scope]['r']]
        all_probas = []
        for i in xrange(n_batches):  # i = batch index
            probas = self.compute_probas(self.data[scope], i)  # probabilities of each response within that batch of being true
            lengths = [len(resp) for resp in self.data[scope]['r'][i*len(probas):(i+1)*len(probas)]]
            for j, l in enumerate(lengths):  # j = response index
                if l not in length_to_scores:
                    length_to_scores[l] = [probas[j]]
                else:
                    length_to_scores[l].append(probas[j])
            all_probas.extend(probas)

        # Plot for all points: score by length
        logger.info("[%s]lengths: %d" % (scope, len(all_lengths)))
        logger.info("[%s]probas: %d" % (scope, len(all_probas)))
        n = min(len(all_lengths), len(all_probas))
        fig = plt.figure()
        plt.plot(all_lengths[:n], all_probas[:n], 'r.')
        plt.title('Length - Score correlation')
        plt.xlabel('Response Length')
        plt.ylabel('Discriminator Score')
        plt.savefig('./plots/plot_%s_length-score_dots.png' % scope)
        plt.close(fig)

        logger.info("[%s]score--length pearson: %s" % (scope, pearsonr(all_lengths[:n], all_probas[:n])))
        logger.info("[%s]score--length spearman: %s" % (scope, spearmanr(all_lengths[:n], all_probas[:n])))

        # Plot average score by length
        logger.info("[%s]number of different lengths: %d" % (scope, len(length_to_scores)))
        # Order dictionary by keys (by response length)
        length_to_scores = collections.OrderedDict(sorted(length_to_scores.items()))
        fig = plt.figure()
        plt.plot(length_to_scores.keys(), [np.average(p) for p in length_to_scores.values()], 'r-')
        plt.title('Length - Score correlation')
        plt.xlabel('Response Length')
        plt.ylabel('Avg. Score')
        plt.savefig('./plots/plot_%s_length-score.png' % scope)
        plt.close(fig)
        logger.info("[%s]saved plots." % scope)

    def plot_learning_curves(self, scope):
        """
        Plot accuracy curves of each model within a specified scope
        :param scope: scope of the data to look at: 'train' or 'val' or 'test'
        :return: None, plot instead
        """
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        fig = plt.figure()
        average_acc = []
        for i, (model_name, accuracies) in enumerate(self.timings[scope].iteritems()):
            # store accuracies into average
            if len(average_acc) == 0:
                average_acc = accuracies
            else:
                average_acc = [average_acc[j]+accuracies[j] for j in range(len(accuracies))]
            # set model name to show on plot
            if '/VHRED/' in model_name:
                if 'Stochastic' in model_name: model_name = 'VHRED-rnd'
                elif 'BeamSearch_5' in model_name: model_name = 'VHRED-beam5'
            elif '/HRED/' in model_name:
                if 'Stochastic' in model_name: model_name = 'HRED-rnd'
                elif 'BeamSearch_5' in model_name: model_name = 'HRED-beam5'
            elif 'c_tfidf' in model_name: model_name = 'TF-IDF'
            # plot individiual lines
            plt.plot(range(len(accuracies)), accuracies, colors[i]+'--', label=model_name)
        # get average line
        average_acc = [average_acc[j]/float(len(self.timings[scope])) for j in range(len(average_acc))]
        # plot it
        plt.plot(range(len(average_acc)), average_acc, colors[len(self.timings[scope])]+'-', label='AVERAGE')

        plt.legend(loc='lower right', fontsize='small')
        plt.grid(True, axis='y')
        plt.xlabel('epoch')
        plt.ylabel('Discriminator Accuracy')
        plt.savefig('./plots/plot_%s_accuracies.png' % scope)
        plt.close(fig)
        logger.info("saved plot.")

    def plot_human_correlation(self, data):
        logger.info("")
        logger.info("Get probabilities of each response...")
        n_batches = len(data['r']) // self.batch_size
        all_human_scores = [s for s in data['score']]
        all_discriminator_scores = []
        human_to_disc = {}
        for i in xrange(n_batches):  # i = batch index
            probas = self.compute_probas(data, i)  # probabilities of each response within that batch of being true
            scores = [s for s in data['score'][i*len(probas): (i+1)*len(probas)]]
            for j, s in enumerate(scores):  # j = context-response-score triple index
                if s not in human_to_disc:
                    human_to_disc[s] = [probas[j]]
                else:
                    human_to_disc[s].append(probas[j])
            all_discriminator_scores.extend(probas)

        n = min(len(all_human_scores), len(all_discriminator_scores))
        logger.info("discriminator--human pearson: %s" % (pearsonr(all_human_scores[:n], all_discriminator_scores[:n]),))
        logger.info("discriminator--human spearman: %s" % (spearmanr(all_human_scores[:n], all_discriminator_scores[:n]),))

        logger.info("number of different scores: %d" % len(human_to_disc))
        # Order dictionary by keys (by human scores)
        human_to_disc = collections.OrderedDict(sorted(human_to_disc.items()))
        fig = plt.figure()
        plt.boxplot([human_to_disc[s] for s in human_to_disc.keys()], labels=human_to_disc.keys())
        plt.title('Human - Discriminator score correlation')
        plt.xlabel('human score')
        plt.ylabel('discriminator score')
        plt.savefig('./plots/plot_human-disc_scores.png')
        plt.close(fig)
        logger.info("saved plot.")

    def train(self, n_epochs=100, patience=10, verbose=True):
        """
        Train the model
        :param n_epochs: max number of training epochs to perform
        :param patience: epochs to continue training if validation score not improved
        :param verbose: show training cost after each batch
        :return: test performance and test probabilities
        """
        epoch = 0           # keep track of number of epochs we ran
        best_val_perf = 0   # keep track of best validation score
        best_val_recall = 0    # keep track of best validation recall@1 score
        test_perf = 0       # keep track of current test score
        test_probas = None  # keep track of current best probabilities

        ###
        # RESUMED TRAINING - RESET VARIABLES:
        ###
        if 'train' in self.timings and 'val' in self.timings and 'test' in self.timings\
                and len(self.timings['train']) > 0 and len(self.timings['val']) > 0:
            assert 'true' in self.timings['train']
            # Reset epoch:
            epoch = len(self.timings['train']['true'])
            logger.info("reset epoch: %d" % epoch)

            # Reset best_val_perf:
            average_val_perf = []  # average validation performance of all models over each epochs
            for model, val_perfs in self.timings['val'].iteritems():
                if len(average_val_perf) == 0:
                    average_val_perf = val_perfs
                else:
                    assert len(average_val_perf) == len(val_perfs)
                    # add the performance of that model over all time steps i
                    average_val_perf = [average_val_perf[i]+val_perfs[i] for i in range(len(val_perfs))]
            # make it an average of all models over all time steps i:
            average_val_perf = [average_val_perf[i]/len(self.timings['val']) for i in range(len(average_val_perf))]
            best_val_perf = np.max(average_val_perf)
            logger.info("reset best_val_perf: %.9f" % best_val_perf)

            if len(self.timings['test']) > 0:
                # Reset test_perf:
                average_test_perf = []  # average test performance of all models over each epochs
                for model, test_perfs in self.timings['test'].iteritems():
                    if len(average_test_perf) == 0:
                        average_test_perf = test_perfs
                    else:
                        assert len(average_test_perf) == len(test_perfs)
                        # add the performance of that model over all time steps i
                        average_test_perf = [average_test_perf[i]+test_perfs[i] for i in range(len(test_perfs))]
                # make it an average of all models over all time steps i:
                average_test_perf = [average_test_perf[i]/len(self.timings['test']) for i in range(len(average_test_perf))]
                test_perf = average_test_perf[-1]
                logger.info("reset test_perf: %.9f" % test_perf)

        n_train_batches = len(self.data['train']['y']) // self.batch_size
        n_val_batches = len(self.data['val']['y']) // self.batch_size
        n_test_batches = len(self.data['test']['y']) // self.batch_size

        ######################
        # MAIN TRAINING LOOP #
        ######################
        while epoch < n_epochs and patience > 0:
            epoch += 1
            epoch_cost = 0  # keep track of training cost for each epoch
            start_time = time.time()

            bar = pyprind.ProgBar(n_train_batches, monitor=True, stream=sys.stdout)  # show a progression bar on the screen
            print ""
            ############################
            # Loop through all batches #
            ############################
            for minibatch_index in range(n_train_batches):
                # Set context, response, flag, mask, and other variables for that batch index
                self.set_shared_variables(self.data['train'], minibatch_index, training=True)
                # Train model on this current batch
                batch_cost = self.train_model()
                if verbose: logger.info("epoch %i: batch %i/%i cost: %.9f" % (epoch, minibatch_index+1, n_train_batches, batch_cost))
                epoch_cost += batch_cost
                # self.set_zero(self.zero_vec)  # Set embeddings[0] to zero vector?
                bar.update()
            ### we trained the model on all the data once! ###

            end_time = time.time()
            logger.info("epoch %i: training cost %.9f, took %d(s)" % (epoch, epoch_cost/n_train_batches, end_time-start_time))

            ###
            # Compute TRAIN performance:
            ###
            logger.info("")
            logger.info("Evaluating Training set:")
            # evaluation for each model id in data['train']['id']
            train_perfs, _ = self.compute_and_save_performance_models("train")
            train_perf = np.average(train_perfs)
            logger.info("epoch %i: train perf=%.9f%%" % (epoch, train_perf*100))

            ###
            # Compute VALIDATION performance:
            ###
            logger.info("")
            logger.info("Evaluating Validation set:")
            # evaluation for each model id in data['val']['id']
            val_perfs, val_recalls = self.compute_and_save_performance_models("val", compute_recalls=True)
            val_perf = np.average(val_perfs)
            logger.info('epoch %i: val_perf=%.9f%%' % (epoch, val_perf*100))

            ###
            # If doing better on validation set, measure each model test performance and same model parameters!
            ###
            if val_perf > best_val_perf or val_recalls[10][1] > best_val_recall:
                logger.info("")
                logger.info("Improved average validation score!")
                best_val_perf = val_perf
                best_val_recall = val_recalls[10][1]  # score of 1 in 10: R@1
                patience = self.patience  # reset patience to initial value

                # Save current best model parameters.
                logger.info("")
                logger.info("Saving current model parameters...")
                with open('%s/%s_best_weights.pkl' % (self.save_path, self.save_prefix), 'wb') as handle:
                    params = [np.asarray(p.eval()) for p in lasagne.layers.get_all_params(self.l_out)]
                    cPickle.dump(params, handle, protocol=cPickle.HIGHEST_PROTOCOL)
                with open('%s/%s_best_embed.pkl' % (self.save_path, self.save_prefix), 'wb') as handle:
                    cPickle.dump(self.embeddings.eval(), handle, protocol=cPickle.HIGHEST_PROTOCOL)
                with open('%s/%s_best_M.pkl' % (self.save_path, self.save_prefix), 'wb') as handle:
                    cPickle.dump(self.M.eval(), handle, protocol=cPickle.HIGHEST_PROTOCOL)
                # Save model.
                logger.info("Saving model...")
                with open("%s/%s_model.pkl" % (self.save_path, self.save_prefix), 'wb') as handle:
                    cPickle.dump(self, handle, protocol=cPickle.HIGHEST_PROTOCOL)
                logger.info("Saved.")
            else:
                patience -= 1  # decrease patience
                logger.info("")
                logger.info("No improvement! patience: %d" % patience)

            # In any case, save performances.
            logger.info("")
            logger.info("Saving performances...")
            with open('%s/%s_timings.pkl' % (self.save_path, self.save_prefix), 'wb') as handle:
                cPickle.dump(self.timings, handle, protocol=cPickle.HIGHEST_PROTOCOL)
            logger.info("Saved.")

        return test_perf  # , test_probas

    def compute_recall_ks(self, probas):
        """
        Return accuracy to get the true response in the top k from a group of responses according to current probabilities
        :param probas: current learned probabilities
        """
        """
        def recall(k, group_size):
            '''
            :param k: the margin in which the true response must be
            :param group_size:  the number of responses to collect
            :return: accuracy
            '''
            n_correct = 0  # keep track of the number of times we got the true response in the top k
            for i in xrange(num_false_responses):
                batch = [probas[i], probas[i+split_index]]  # probas of the true response, and a random one for the same context
                indices = np.argpartition(batch, -k)[-k:]  # get top k highest probabilities
                if 0 in indices:  # assumes the true context-response pair is the first one
                    n_correct += 1
            return n_correct / num_false_responses
        """
        logger.info("Computing recalls...")
        def recall(k, group_size):
            test_size = 10  # 10 responses for each context: 1 true, 9 false
            n_batches = len(probas) // test_size
            n_correct = 0
            for i in xrange(n_batches):
                batch = np.array(probas[i*test_size: (i+1)*test_size])  # get the 10 probas for that context
                batch = [batch[0]] + random.sample(batch[1:], group_size-1)  # [0]=true answer & sample of group_size-1 false responses
                indices = np.argpartition(batch, -k)[-k:]  # get the top k probas
                if 0 in indices:
                    n_correct += 1
            return n_correct / (len(probas)/test_size)

        recall_k = {}
        for group_size in [2, 5, 10]:
            recall_k[group_size] = {}
            logger.info('group_size: %d' % group_size)
            for k in [1, 2, 5]:
                if k < group_size:
                    recall_k[group_size][k] = recall(k, group_size)
                    logger.info('recall@%d = %s' % (k, recall_k[group_size][k]))
        return recall_k

    def retrieve(self, context_set=None, context_embs=None, response_set=None, response_embs=None, k=10, batch_size=100, verbose=True):
        """
        For each context in the context_set, return the k most probable responses from the response_set
        :param context_set: list of contexts strings to query. default=data[train][c]
        :param context_embs: list of context embeddings to combine with responses (used for speedup when doing multiple calls)
        :param response_set: list of responses strings to try for each context. default=data[train][r]
        :param response_embs: list of response embeddings to combine with contexts (used for speedup when doing multiple calls)
        :param k: number of top responses to retrieve
        :param batch_size: number of contexts to consider at a time
        :param verbose: be berbose
        """
        # response_set not provided, use the training ones, and create a str version
        if response_set is None or len(response_set) == 0:
            response_set = copy.deepcopy(self.data['train']['r'])
            # convert response_set from idx to string!!
            response_set_str = []
            for r in response_set:
                response_set_str.append(self.indices2string(r))
        # response_set provided, use those strings and create an idx version
        else:
            response_set_str = copy.deepcopy(response_set)
            if response_embs is None or len(response_embs) == 0:
                # convert response_set from string to idx!!
                response_set = []
                for r in response_set_str:
                    response_set.append(self.string2indices(r))

        # context_set not provided, use the training ones, and create a str version
        if context_set is None or len(context_set) == 0:
            context_set = copy.deepcopy(self.data['train']['c'])
            # convert context_set from idx to string!!
            context_set_str = []
            for c in context_set:
                context_set_str.append(self.indices2string(c))
        # context_set provided, use those strings and create an idx version
        else:
            context_set_str = copy.deepcopy(context_set)
            if context_embs is None or len(context_embs) == 0:
                # convert context_set from string to idx!!
                context_set = []
                for c in context_set_str:
                    context_set.append(self.string2indices(c))

        if response_embs is None or len(response_embs) == 0:
            ###
            # Compute embedding of each response
            ###
            if verbose: logger.info("Computing response embeddings...")
            # Pad response set to be divisible by batch_size
            length = len(response_set)
            while len(response_set) % self.batch_size != 0:
                response_set.append(response_set[0])
            # Compute embeddings
            n_batches = len(response_set) // self.batch_size
            response_embs = []  # list of response embeddings
            if n_batches > 100: bar = pyprind.ProgBar(n_batches, monitor=True, stream=sys.stdout)  # show a progression bar on the screen
            for i in xrange(n_batches):
                response_embs.extend(self.compute_response_embeddings(response_set, i))
                if n_batches > 100: bar.update()
            if verbose: print ""
            # Ignore padded embeddings
            response_embs = response_embs[:length]
            response_set = response_set[:length]

        if context_embs is None or len(context_embs) == 0:
            ###
            # Compute embedding of each context
            ###
            if verbose: logger.info("Computing context embeddings...")
            # Pad context set to be divisible by batch_size
            length = len(context_set)
            while len(context_set) % self.batch_size != 0:
                context_set.append(context_set[0])
            # Compute embeddings
            n_batches = len(context_set) // self.batch_size
            context_embs = []  # list of context embeddings
            if n_batches > 100: bar = pyprind.ProgBar(n_batches, monitor=True, stream=sys.stdout)  # show a progression bar on the screen
            for i in xrange(n_batches):
                context_embs.extend(self.compute_context_embeddings(context_set, i))
                if n_batches > 100: bar.update()
            if verbose: print ""
            # Ignore padded embeddings
            context_embs = context_embs[:length]
            context_set = context_set[:length]

        retrieved_data = {
            'c': context_set_str,  # each context
            'c_embs': context_embs,
            'r': response_set_str,  # responses to consider
            'r_embs': response_embs,
            'r_retrieved': [],  # list of k responses for each context
            'r_retrieved_embs': [],
            'proba_retrieved': []  # list of k probabilities for each context
        }
        assert len(retrieved_data['c']) == len(retrieved_data['c_embs'])
        assert len(retrieved_data['r']) == len(retrieved_data['r_embs'])

        e_context = T.fmatrix('e_context')  # (batch_size, hidden_size)
        e_responses = T.fmatrix('e_responses')  # (#_responses, hidden_size)

        dp = T.dot(e_context, T.dot(e_responses, self.M.T).T)  # (batch_size, #of_responses)
        dp += self.b
        o = T.nnet.sigmoid(dp)
        o = T.clip(o, 1e-7, 1.0-1e-7)
        # proba = T.concatenate([(1-o).reshape((-1,1)), o.reshape((-1,1))], axis=1)  # (batch_size*#_responses , 2)

        get_response_probas = theano.function(
            inputs=[e_context, e_responses],
            outputs=o,
            name='get_response_probas'
        )

        for c_idx in range(0, len(context_embs), batch_size):
            end = min(c_idx+batch_size, len(context_embs))
            if verbose: logger.info("Retrieving top %d responses for context %d-->%d/%d..." % (k, c_idx+1, end, len(context_embs)))
            probas = get_response_probas(context_embs[c_idx: end], response_embs)  # (batch_size, #of_responses)
            
            # NOTE: the next line is only true when response_set & context_set are well alligned!
            # if verbose: logger.info("average proba of true responses: %.9f" % np.mean([probas[i, c_idx+i] for i in range(probas.shape[0])]))

            # get top k responses: from less probable to most probable
            top_k_indices = np.argpartition(probas, -k)[:, -k:]  # (batch_size, k)
            for b in range(top_k_indices.shape[0]):  # for each batch of contexts
                # NOTE: the next 2 lines are only true when response_set & context_set are well alligned!
                # if c_idx+b in top_k_indices[b]:
                #     if verbose: logger.info("  Got true response!")
            
                ret_responses = [response_set_str[i] for i in top_k_indices[b]]
                ret_response_embs = [response_embs[i] for i in top_k_indices[b]]
                ret_probas = [probas[b][i] for i in top_k_indices[b]]
                retrieved_data['r_retrieved'].append(ret_responses)
                retrieved_data['r_retrieved_embs'].append(ret_response_embs)
                retrieved_data['proba_retrieved'].append(ret_probas)

        assert len(retrieved_data['c']) == len(retrieved_data['r_retrieved']) == len(retrieved_data['r_retrieved_embs']) == len(retrieved_data['proba_retrieved'])
        return retrieved_data
