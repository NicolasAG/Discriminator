from __future__ import division
import argparse
import cPickle
import lasagne
import lasagne as nn
import numpy as np
import pyprind
import re
import sys
import theano
import theano.tensor as T
import time
from collections import defaultdict, OrderedDict
from theano.ifelse import ifelse
from theano.printing import Print as pp
from lasagne import nonlinearities, init, utils
from lasagne.layers import Layer, InputLayer, DenseLayer, helper
sys.setrecursionlimit(10000)

class GradClip(theano.compile.ViewOp):
    def __init__(self, clip_lower_bound, clip_upper_bound):
        self.clip_lower_bound = clip_lower_bound
        self.clip_upper_bound = clip_upper_bound
        assert(self.clip_upper_bound >= self.clip_lower_bound)

    def grad(self, args, g_outs):
        def pgrad(g_out):
            g_out = T.clip(g_out, self.clip_lower_bound, self.clip_upper_bound)
            g_out = ifelse(T.any(T.isnan(g_out)), T.ones_like(g_out)*0.00001, g_out)
            return g_out
        return [pgrad(g_out) for g_out in g_outs]

gradient_clipper = GradClip(-10.0, 10.0)
#T.opt.register_canonicalize(theano.gof.OpRemove(gradient_clipper), name='gradient_clipper')

def adam(loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8,
         gamma=1-1e-8):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]

    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf

    """
    updates = []
    all_grads = theano.grad(gradient_clipper(loss), all_params)
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))

        m = b1_t*m_previous + (1 - b1_t)*g                             # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2                              # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)

        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
    updates.append((t, t + 1.))
    return updates

class Model:
    def __init__(self,
                 data,
                 W,
                 max_seqlen=160,
                 hidden_size=100,
                 batch_size=50,
                 lr=0.001,
                 lr_decay=0.95,
                 sqr_norm_lim=9,
                 fine_tune_W=False,
                 fine_tune_M=False,
                 use_ntn=False,
                 optimizer='adam',
                 forget_gate_bias=2,
                 filter_sizes=[3,4,5],
                 num_filters=100,
                 conv_attn=False,
                 encoder='rnn',
                 elemwise_sum=True,
                 corr_penalty=0.0,
                 xcov_penalty=0.0,
                 penalize_emb_norm=False,
                 penalize_emb_drift=False,
                 penalize_activations=False,
                 emb_penalty=0.001,
                 act_penalty=500,
                 k=4,
                 n_recurrent_layers=1,
                 is_bidirectional=False,
                 **kwargs):
        embedding_size = W.shape[1]
        self.data = data
        self.max_seqlen = max_seqlen
        self.batch_size = batch_size
        self.fine_tune_W = fine_tune_W
        self.fine_tune_M = fine_tune_M
        self.use_ntn = use_ntn
        self.lr = lr
        self.lr_decay = lr_decay
        self.optimizer = optimizer
        self.sqr_norm_lim = sqr_norm_lim
        self.conv_attn = conv_attn
        self.emb_penalty = emb_penalty
        self.penalize_emb_norm = penalize_emb_norm
        self.penalize_emb_drift = penalize_emb_drift
        if penalize_emb_drift:
            self.orig_embeddings = theano.shared(W.copy(), name='orig_embeddings', borrow=True)
        self.encoder = encoder

        c = T.imatrix('c')
        r = T.imatrix('r')
        y = T.ivector('y')
        c_mask = T.fmatrix('c_mask')
        r_mask = T.fmatrix('r_mask')
        c_seqlen = T.ivector('c_seqlen')
        r_seqlen = T.ivector('r_seqlen')
        embeddings = theano.shared(W, name='embeddings', borrow=True)
        zero_vec_tensor = T.fvector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], updates=[(embeddings, T.set_subtensor(embeddings[0,:], zero_vec_tensor))])
        if encoder.find('cnn') > -1 and (encoder.find('rnn') > -1 or encoder.find('lstm') > -1) and not elemwise_sum:
            self.M = theano.shared(np.eye(2*hidden_size).astype(theano.config.floatX), borrow=True)
        elif use_ntn:
            self.U = theano.shared(np.random.uniform(-0.01, 0.01, size=(k,)).astype(theano.config.floatX), borrow=True)
            self.V = theano.shared(np.random.uniform(-0.01, 0.01, size=(k, 2*hidden_size)).astype(theano.config.floatX), borrow=True)
            self.b = theano.shared(np.random.uniform(-0.01, 0.01, size=(k,)).astype(theano.config.floatX), borrow=True)
            self.M = theano.shared(np.random.uniform(-0.01, 0.01, size=(k, hidden_size, hidden_size)).astype(theano.config.floatX), borrow=True)
            self.f = lasagne.nonlinearities.tanh
        else:
            self.M = theano.shared(np.eye(hidden_size).astype(theano.config.floatX), borrow=True)

        c_input = embeddings[c.flatten()].reshape((c.shape[0], c.shape[1], embeddings.shape[1]))
        r_input = embeddings[r.flatten()].reshape((r.shape[0], r.shape[1], embeddings.shape[1]))

        l_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, embedding_size))
        l_mask = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen))

        if encoder.find('cnn') > -1:
            # Building a CNN model
            l_conv_in = lasagne.layers.ReshapeLayer(l_in, shape=(batch_size, 1, max_seqlen, embedding_size))
            conv_layers = []
            for filter_size in filter_sizes:
                conv_layer = lasagne.layers.Conv2DLayer(
                        l_conv_in,
                        num_filters=num_filters,
                        filter_size=(filter_size, embedding_size),
                        stride=(1,1),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        pad='valid'
                        )
                pool_layer = lasagne.layers.MaxPool2DLayer(
                        conv_layer,
                        pool_size=(max_seqlen-filter_size+1, 1)
                        )
                conv_layers.append(pool_layer)

            l_conv = lasagne.layers.ConcatLayer(conv_layers)
            l_conv = lasagne.layers.DenseLayer(l_conv, num_units=hidden_size, nonlinearity=lasagne.nonlinearities.tanh)

        if is_bidirectional:
            if encoder.find('lstm') > -1:
                # Building a bidirectional LSTM model:
                prev_fwd, prev_bck = l_in, l_in
                for _ in xrange(n_recurrent_layers):
                    l_fwd = lasagne.layers.LSTMLayer(prev_fwd,
                                                     hidden_size,
                                                     grad_clipping=10,
                                                     forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(forget_gate_bias)),
                                                     backwards=False,
                                                     learn_init=True,
                                                     peepholes=True,
                                                     mask_input=l_mask)

                    l_bck = lasagne.layers.LSTMLayer(prev_bck,
                                                     hidden_size,
                                                     grad_clipping=10,
                                                     forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(forget_gate_bias)),
                                                     backwards=True,
                                                     learn_init=True,
                                                     peepholes=True,
                                                     mask_input=l_mask)
                    prev_fwd, prev_bck = l_fwd, l_bck
            else:
                # Building a bidirectional RNN model:
                prev_fwd, prev_bck = l_in, l_in
                for _ in xrange(n_recurrent_layers):
                    l_fwd = lasagne.layers.RecurrentLayer(prev_fwd,
                                                          hidden_size,
                                                          nonlinearity=lasagne.nonlinearities.tanh,
                                                          W_hid_to_hid=lasagne.init.Orthogonal(),
                                                          W_in_to_hid=lasagne.init.Orthogonal(),
                                                          backwards=False,
                                                          learn_init=True,
                                                          mask_input=l_mask
                                                          )

                    l_bck = lasagne.layers.RecurrentLayer(prev_bck,
                                                          hidden_size,
                                                          nonlinearity=lasagne.nonlinearities.tanh,
                                                          W_hid_to_hid=lasagne.init.Orthogonal(),
                                                          W_in_to_hid=lasagne.init.Orthogonal(),
                                                          backwards=True,
                                                          learn_init=True,
                                                          mask_input=l_mask
                                                          )
                    prev_fwd, prev_bck = l_fwd, l_bck

            l_recurrent = lasagne.layers.ConcatLayer([l_fwd, l_bck])
        else:
            prev_fwd = l_in
            if encoder.find('lstm') > -1:
                # Building a LSTM model:
                for _ in xrange(n_recurrent_layers):
                    l_recurrent = lasagne.layers.LSTMLayer(incoming=prev_fwd,
                                                           num_units=hidden_size,
                                                           grad_clipping=10,
                                                           forgetgate=lasagne.layers.Gate(b=lasagne.init.Constant(forget_gate_bias)),
                                                           backwards=False,
                                                           learn_init=True,
                                                           peepholes=True,
                                                           mask_input=l_mask)
                    prev_fwd = l_recurrent
            elif encoder.find('gru') > -1:
                for _ in xrange(n_recurrent_layers):
                    l_recurrent = lasagne.layers.GRULayer(prev_fwd,
                                                          hidden_size,
                                                          grad_clipping=10,
                                                          resetgate=lasagne.layers.Gate(b=lasagne.init.Constant(forget_gate_bias)),
                                                          backwards=False,
                                                          learn_init=True,
                                                          mask_input=l_mask)
                    prev_fwd = l_recurrent
            else:
                # Building a RNN model:
                for _ in xrange(n_recurrent_layers):
                    l_recurrent = lasagne.layers.RecurrentLayer(incoming=prev_fwd,
                                                                num_units=hidden_size,
                                                                nonlinearity=lasagne.nonlinearities.tanh,
                                                                W_hid_to_hid=lasagne.init.Orthogonal(),
                                                                W_in_to_hid=lasagne.init.Orthogonal(),
                                                                backwards=False,
                                                                learn_init=True,
                                                                mask_input=l_mask
                                                                )
                    prev_fwd = l_recurrent

        recurrent_size = hidden_size * 2 if is_bidirectional else hidden_size

        if conv_attn:
            l_rconv_in = lasagne.layers.InputLayer(shape=(batch_size, max_seqlen, recurrent_size))
            l_rconv_in = lasagne.layers.ReshapeLayer(l_rconv_in, shape=(batch_size, 1, max_seqlen, recurrent_size))
            conv_layers = []
            for filter_size in filter_sizes:
                conv_layer = lasagne.layers.Conv2DLayer(
                        l_rconv_in,
                        num_filters=num_filters,
                        filter_size=(filter_size, recurrent_size),
                        stride=(1,1),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        pad='valid'
                        )
                pool_layer = lasagne.layers.MaxPool2DLayer(
                        conv_layer,
                        pool_size=(max_seqlen-filter_size+1, 1)
                        )
                conv_layers.append(pool_layer)

            l_hidden1 = lasagne.layers.ConcatLayer(conv_layers)
            l_hidden2 = lasagne.layers.DenseLayer(l_hidden1, num_units=hidden_size, nonlinearity=lasagne.nonlinearities.tanh)
            l_out = l_hidden2
        else:
            l_out = l_recurrent

        if conv_attn:
            e_context = lasagne.layers.helper.get_output(
                layer_or_layers=l_recurrent,
                inputs={l_in: c_input, l_mask: c_mask},
                deterministic=False
            )
            e_response = lasagne.layers.helper.get_output(
                layer_or_layers=l_recurrent,
                inputs={l_in: r_input, l_mask: r_mask},
                deterministic=False
            )
            def step_fn(row_t, mask_t):
                return row_t * mask_t.reshape((-1, 1))
            if is_bidirectional:
                e_context, _ = theano.scan(step_fn, outputs_info=None, sequences=[e_context, T.concatenate([c_mask, c_mask], axis=1)])
                e_response, _ = theano.scan(step_fn, outputs_info=None, sequences=[e_response, T.concatenate([r_mask, r_mask], axis=1)])
            else:
                e_context, _ = theano.scan(step_fn, outputs_info=None, sequences=[e_context, c_mask])
                e_response, _ = theano.scan(step_fn, outputs_info=None, sequences=[e_response, r_mask])

            e_context = lasagne.layers.helper.get_output(
                layer_or_layers=l_out,
                inputs={l_in: e_context, l_mask: c_mask},
                deterministic=False
            )
            e_response = lasagne.layers.helper.get_output(
                layer_or_layers=l_out,
                inputs={l_in: e_response, l_mask: r_mask},
                deterministic=False
            )
        else:
            # h_context = lasagne.layers.helper.get_output(l_out, c_input, mask=c_mask, deterministic=False)
            # h_response = lasagne.layers.helper.get_output(l_out, r_input, mask=r_mask, deterministic=False)
            h_context = lasagne.layers.helper.get_output(
                layer_or_layers=l_out,
                inputs={l_in: c_input, l_mask: c_mask},
                deterministic=False
            )
            h_response = lasagne.layers.helper.get_output(
                layer_or_layers=l_out,
                inputs={l_in: r_input, l_mask: r_mask},
                deterministic=False
            )
            e_context = h_context[T.arange(batch_size), c_seqlen].reshape((c.shape[0], hidden_size))
            e_response = h_response[T.arange(batch_size), r_seqlen].reshape((r.shape[0], hidden_size))

        if encoder.find('cnn') > -1:
            e_conv_context = lasagne.layers.helper.get_output(l_conv, c_input, deterministic=False)
            e_conv_response = lasagne.layers.helper.get_output(l_conv, r_input, deterministic=False)
            if encoder.find('rnn') > -1 or encoder.find('lstm') > -1:
                if elemwise_sum:
                    e_context = e_context + e_conv_context
                    e_response = e_response + e_conv_response
                else:
                    e_context = T.concatenate([e_context, e_conv_context], axis=1)
                    e_response = T.concatenate([e_response, e_conv_response], axis=1)

                # penalize correlation
                if abs(corr_penalty) > 0:
                    cor = []
                    for i in range(hidden_size if elemwise_sum else 2*hidden_size):
                        y1, y2 = e_context, e_response
                        x1 = y1[:,i] - (np.ones(batch_size)*(T.sum(y1[:,i])/batch_size))
                        x2 = y2[:,i] - (np.ones(batch_size)*(T.sum(y2[:,i])/batch_size))
                        nr = T.sum(x1 * x2) / (T.sqrt(T.sum(x1 * x1))*T.sqrt(T.sum(x2 * x2)))
                        cor.append(-nr)
                if abs(xcov_penalty) > 0:
                    e_context_mean = T.mean(e_context, axis=0, keepdims=True)
                    e_response_mean = T.mean(e_response, axis=0, keepdims=True)
                    e_context_centered = e_context - e_context_mean # (n, i)
                    e_response_centered = e_response - e_response_mean # (n, j)

                    outer_prod = (e_context_centered.dimshuffle(0, 1, 'x') *
                                  e_response_centered.dimshuffle(0, 'x', 1)) # (n, i, j)
                    xcov = T.sum(T.sqr(T.mean(outer_prod, axis=0)))
            else:
                e_context = e_conv_context
                e_response = e_conv_response

        if use_ntn:
            dp = T.concatenate([T.batched_dot(e_context, T.dot(e_response, self.M[i])) for i in xrange(k)], axis=1)
            dp += T.concatenate([e_context, e_response], axis=1).dot(self.V.T) + self.b
            dp = self.f(dp).dot(self.U)
        else:
            dp = T.batched_dot(e_context, T.dot(e_response, self.M.T))

        o = T.nnet.sigmoid(dp)
        o = T.clip(o, 1e-7, 1.0-1e-7)

        self.shared_data = {}
        for key in ['c', 'r']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size, max_seqlen), dtype=np.int32), borrow=True)
        for key in ['c_mask', 'r_mask']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size, max_seqlen), dtype=theano.config.floatX), borrow=True)
        for key in ['y', 'c_seqlen', 'r_seqlen']:
            self.shared_data[key] = theano.shared(np.zeros((batch_size,), dtype=np.int32), borrow=True)

        self.probas = T.concatenate([(1-o).reshape((-1,1)), o.reshape((-1,1))], axis=1)
        self.pred = T.argmax(self.probas, axis=1)
        self.errors = T.sum(T.neq(self.pred, y))
        self.cost = T.nnet.binary_crossentropy(o, y).mean()

        if self.penalize_emb_norm:
            self.cost += self.emb_penalty * (embeddings ** 2).sum()

        if self.penalize_emb_drift:
            self.cost += self.emb_penalty * ((embeddings - self.orig_embeddings) ** 2).sum()

        if penalize_activations and not conv_attn:
            self.cost += act_penalty * T.stack([((h_context[:,i] - h_context[:,i+1]) ** 2).sum(axis=1).mean() for i in xrange(max_seqlen-1)]).mean()
            self.cost += act_penalty * T.stack([((h_response[:,i] - h_response[:,i+1]) ** 2).sum(axis=1).mean() for i in xrange(max_seqlen-1)]).mean()

        if encoder.find('cnn') > -1 and (encoder.find('rnn') > -1 or encoder.find('lstm') > -1):
            if abs(corr_penalty) > 0:
                self.cost += corr_penalty * T.sum(cor)
            if abs(xcov_penalty) > 0:
                self.cost += xcov_penalty * xcov
        self.l_out = l_out
        self.l_recurrent = l_recurrent
        self.embeddings = embeddings
        self.c = c
        self.r = r
        self.y = y
        self.c_seqlen = c_seqlen
        self.r_seqlen = r_seqlen
        self.c_mask = c_mask
        self.r_mask = r_mask

        self.update_params()

    def update_params(self):
        params = lasagne.layers.get_all_params(self.l_out)
        if self.use_ntn:
            params += [self.U, self.V, self.M, self.b]
        if self.conv_attn:
            params += lasagne.layers.get_all_params(self.l_recurrent)
        if self.fine_tune_W:
            params += [self.embeddings]
        if self.fine_tune_M and not self.use_ntn:
            params += [self.M]

        total_params = sum([p.get_value().size for p in params])
        print "total_params: ", total_params

        if 'adam' == self.optimizer:
            updates = lasagne.updates.adam(loss_or_grads=self.cost, params=params, learning_rate=self.lr)
        elif 'adadelta' == self.optimizer:
            updates = sgd_updates_adadelta(self.cost, params, self.lr_decay, 1e-6, self.sqr_norm_lim)
            # updates = lasagne.updates.adadelta(self.cost, params, learning_rate=1.0, rho=self.lr_decay)
        else:
            raise 'Unsupported optimizer: %s' % self.optimizer

        givens = {
            self.c: self.shared_data['c'],
            self.r: self.shared_data['r'],
            self.y: self.shared_data['y'],
            self.c_seqlen: self.shared_data['c_seqlen'],
            self.r_seqlen: self.shared_data['r_seqlen'],
            self.c_mask: self.shared_data['c_mask'],
            self.r_mask: self.shared_data['r_mask']
        }

        # self.get_pred = theano.function(
        #     inputs=[],
        #     outputs=self.pred,
        #     givens=givens,
        #     on_unused_input='ignore',
        #     allow_input_downcast=True
        # )
        self.train_model = theano.function(
            inputs=[],
            outputs=self.cost,
            updates=updates,
            givens=givens,
            on_unused_input='ignore',
            allow_input_downcast=True
        )
        self.get_loss = theano.function(
            inputs=[],
            outputs=self.errors,
            givens=givens,
            on_unused_input='ignore',
            allow_input_downcast=True
        )
        self.get_probas = theano.function(
            inputs=[],
            outputs=self.probas,
            givens=givens,
            on_unused_input='ignore',
            allow_input_downcast=True
        )

    def set_shared_variables(self, dataset, index):
        """
        Set shared variables for that batch index.
        Set context and response: value, mask and length
        :param dataset: dictionary of contexts, responses, and flags
        :param index: batch index to work on
        :return: None
        """
        def get_batch(dataset, index):
            """
            Get description of the data at a given batch index.
            :param dataset: array of tokens (can represent either contexts or responses)
            :param index: current index of the batch
            :return: batch data, sequence length for each element in batch, mask for each element in batch
            """
            seqlen = np.zeros((self.batch_size,), dtype=np.int32)
            mask = np.zeros((self.batch_size, self.max_seqlen), dtype=theano.config.floatX)
            batch = np.zeros((self.batch_size, self.max_seqlen), dtype=np.int32)
            data = dataset[index * self.batch_size:(index + 1) * self.batch_size]
            for i, row in enumerate(data):
                row = row[:self.max_seqlen]  # cut the sequence if longer than max_seqlen
                batch[i, 0:len(row)] = row  # put the data into our batch
                seqlen[i] = len(row) - 1  # max index for that batch element
                mask[i, 0:len(row)] = 1  # put a '1' on the sequence, 0 else where
            return batch, seqlen, mask

        c, c_seqlen, c_mask = get_batch(dataset['c'], index)
        r, r_seqlen, r_mask = get_batch(dataset['r'], index)
        y = np.array(dataset['y'][index*self.batch_size:(index+1)*self.batch_size], dtype=np.int32)
        self.shared_data['c'].set_value(c)
        self.shared_data['r'].set_value(r)
        self.shared_data['y'].set_value(y)
        self.shared_data['c_seqlen'].set_value(c_seqlen)
        self.shared_data['r_seqlen'].set_value(r_seqlen)
        self.shared_data['c_mask'].set_value(c_mask)
        self.shared_data['r_mask'].set_value(r_mask)

    def compute_loss(self, dataset, index):
        self.set_shared_variables(dataset, index)
        return self.get_loss()

    def compute_probas(self, dataset, index):
        self.set_shared_variables(dataset, index)
        return self.get_probas()[:,1]

    def compute_performance_models(self, scope):
        """
        Measure the accuracy of the current Discriminator on each dialogue model.
        :param scope: either "train", "val" or "test" sets.
        :return: output the accuracy of the discriminator for each model.
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

        for model_name, data in self.data_by_models[scope].iteritems():
            print "evaluating", model_name
            n_batches = len(data['y']) // self.batch_size
            # Compute performance:
            losses = [self.compute_loss(data, i) for i in xrange(n_batches)]
            perf = 1 - np.sum(losses) / len(data['y'])
            print '%s_perf: %f' % (scope, perf * 100)

    def test(self):
        """
        Compute performances on test set
        :return: None
        """
        n_test_batches = len(self.data['test']['y']) // self.batch_size
        # Compute TEST performance:
        test_losses = [self.compute_loss(self.data['test'], i) for i in xrange(n_test_batches)]
        test_perf = 1 - np.sum(test_losses) / len(self.data['test']['y'])
        print 'test_perf: %f' % (test_perf * 100)
        # evaluation for each model id in data['test']['id']
        self.compute_performance_models("test")

    def train(self, n_epochs=100, verbose=False):
        """
        Train the model
        :param n_epochs: number of training epochs to perform
        :return: test performance and test probabilities
        """
        epoch = 0           # keep track of number of epochs we ran
        best_val_perf = 0   # keep track of best validation score
        test_perf = 0       # keep track of current test score
        test_probas = None  # keep track of current best probabilities

        n_train_batches = len(self.data['train']['y']) // self.batch_size
        n_val_batches = len(self.data['val']['y']) // self.batch_size
        n_test_batches = len(self.data['test']['y']) // self.batch_size

        ######################
        # MAIN TRAINING LOOP #
        ######################
        while (epoch < n_epochs):
            epoch += 1
            indices = range(n_train_batches)  # list of batch indices to train on (1 batch at a time)

            bar = pyprind.ProgBar(len(indices), monitor=True)  # show a progression bar on the screen

            total_cost = 0  # keep track of training cost
            start_time = time.time()

            ############################
            # Loop through all batches #
            ############################
            for minibatch_index in indices:
                # Set context, response, flag, mask, and other variables for that batch index
                self.set_shared_variables(self.data['train'], minibatch_index)
                # Train model on this current batch
                cost_epoch = self.train_model()
                if verbose: print "cost epoch:", cost_epoch
                total_cost += cost_epoch
                self.set_zero(self.zero_vec)  # TODO: check what this does?
                bar.update()
            ### we trained the model on all the data once! ###

            end_time = time.time()
            print "average training batch cost: ", (total_cost / len(indices)), " took: %d(s)" % (end_time - start_time)

            ###
            # Compute TRAIN performance:
            ###
            train_losses = [self.compute_loss(self.data['train'], i) for i in xrange(n_train_batches)]
            train_perf = 1 - np.sum(train_losses) / len(self.data['train']['y'])
            print "epoch %i, train perf %f" % (epoch, train_perf*100)
            # evaluation for each model id in data['train']['id']
            self.compute_performance_models("train")

            ###
            # Compute VALIDATION performance:
            ###
            val_losses = [self.compute_loss(self.data['val'], i) for i in xrange(n_val_batches)]
            val_perf = 1 - np.sum(val_losses) / len(self.data['val']['y'])
            print 'epoch %i, val_perf %f' % (epoch, val_perf*100)
            # evaluation for each model id in data['val']['id']
            self.compute_performance_models("val")

            ###
            # If doing better on validation set, measure test performance and same model parameters!
            ###
            if val_perf > best_val_perf:
                print "\nImproved validation score!"
                best_val_perf = val_perf

                # Compute TEST performance:
                test_losses = [self.compute_loss(self.data['test'], i) for i in xrange(n_test_batches)]
                test_probas = [self.compute_probas(self.data['test'], i) for i in xrange(n_test_batches)]
                test_perf = 1 - np.sum(test_losses) / len(self.data['test']['y'])
                print 'epoch %i, test_perf %f' % (epoch, test_perf*100)
                # evaluation for each model id in data['test']['id']
                self.compute_performance_models("test")

                # Save current best model parameters.
                print "\nSaving current model parameters..."
                with open('weights_%s_best.pkl' % self.encoder, 'wb') as handle:
                    params = [np.asarray(p.eval()) for p in lasagne.layers.get_all_params(self.l_out)]
                    cPickle.dump(params, handle)
                with open('embed_%s_best.pkl' % self.encoder, 'wb') as handle:
                    cPickle.dump(self.embeddings.eval(), handle)
                with open('M_%s_best.pkl' % self.encoder, 'wb') as handle:
                    cPickle.dump(self.M.eval(), handle)
                print "Saved.\n"

        return test_perf, test_probas

    # TODO: never used!
    def compute_recall_ks(self, probas):
        def recall(probas, k, group_size):
            """
            Return accuracy to get the true response in the top k from a group of responses according to current probabilities
            :param probas: current learned probabilities
            :param k: the margin in which the true response must be
            :param group_size:  the number of responses to collect
            :return: accuracy
            """
            test_size = 10
            n_batches = len(probas) // test_size
            n_correct = 0  # keep track of the number of times we got the true response in the top k
            for i in xrange(n_batches):
                batch = np.array(probas[i*test_size: (i+1)*test_size])[:group_size]
                indices = np.argpartition(batch, -k)[-k:]
                if 0 in indices:
                    n_correct += 1
            return n_correct / (len(probas) / test_size)

        recall_k = {}
        for group_size in [2, 5, 10]:
            recall_k[group_size] = {}
            print 'group_size: %d' % group_size
            for k in [1, 2, 5]:
                if k < group_size:
                    recall_k[group_size][k] = recall(probas, k, group_size)
                    print 'recall@%d' % k, recall_k[group_size][k]
        return recall_k

# TODO: probably don't need this if we can use lasagna implementation
def sgd_updates_adadelta(cost, params, rho=0.95, epsilon=1e-6, norm_lim=9, word_vec_name='embeddings'):
    def as_floatX(variable):
        if isinstance(variable, float):
            return np.cast[theano.config.floatX](variable)
        elif isinstance(variable, np.ndarray):
            return np.cast[theano.config.floatX](variable)
        return theano.tensor.cast(variable, theano.config.floatX)

    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name != word_vec_name):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates

def sort_by_len(dataset):
    """
    Sort a given data set by its context length
    :param data set: dictionary of contexts, responses, flags
    :return: ordered dictionary
    """
    c, r, y = dataset['c'], dataset['r'], dataset['y']
    indices = range(len(y))
    indices.sort(key=lambda i: len(c[i]))
    for k in ['c', 'r', 'y']:
        dataset[k] = np.array(dataset[k])[indices]

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def main():
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)

    # TODO: check if I can remove those:
    parser.add_argument('--conv_attn', type='bool', default=False, help='Use convolutional attention')
    parser.add_argument('--use_ntn', type='bool', default=False, help='Whether to use NTN')
    parser.add_argument('--k', type=int, default=4, help='Size of k in NTN')
    parser.add_argument('--corr_penalty', type=float, default=0.0, help='Correlation penalty')
    parser.add_argument('--xcov_penalty', type=float, default=0.0, help='XCov penalty')
    parser.add_argument('--penalize_emb_norm', type='bool', default=False, help='Whether to penalize norm of embeddings')
    parser.add_argument('--penalize_emb_drift', type='bool', default=False, help='Whether to use re-embedding words penalty')
    parser.add_argument('--penalize_activations', type='bool', default=False, help='Whether to penalize activations')
    parser.add_argument('--emb_penalty', type=float, default=0.001, help='Embedding penalty')
    parser.add_argument('--act_penalty', type=float, default=500, help='Activation penalty')

    # Structure of Network:
    parser.add_argument('--encoder', type=str, default='rnn', help='Type of encoding RNN units: rnn, gru, lst')
    parser.add_argument('--hidden_size', type=int, default=200, help='Hidden size')
    parser.add_argument('--is_bidirectional', type='bool', default=False, help='Bidirectional RNN')
    parser.add_argument('--n_recurrent_layers', type=int, default=1, help='Num recurrent layers')

    # What to optimize in the network:
    parser.add_argument('--fine_tune_W', type='bool', default=False, help='Whether to fine-tune word embeddings W')
    parser.add_argument('--fine_tune_M', type='bool', default=False, help='Whether to fine-tune M')

    # Size of the batches:
    parser.add_argument('--max_seqlen', type=int, default=160, help='Max seqlen')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')

    # Learning parameters:
    parser.add_argument('--n_epochs', type=int, default=100, help='Num epochs')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay')
    # TODO: can these be fixed?
    parser.add_argument('--sqr_norm_lim', type=float, default=1, help='Squared norm limit')
    parser.add_argument('--forget_gate_bias', type=float, default=2.0, help='Forget gate bias')

    # Saving parameters:
    parser.add_argument('--save_model', type='bool', default=False, help='Whether to save the model')
    parser.add_argument('--model_fname', type=str, default='model.pkl', help='Model filename')

    # Loading parameters:
    parser.add_argument('--input_dir', type=str, default='.', help='Input dir')
    parser.add_argument('--dataset_fname', type=str, default='dataset.pkl', help='Dataset filename')
    parser.add_argument('--W_fname', type=str, default='W.pkl', help='W filename')
    parser.add_argument('--train_examples', type=int, required=False, help='Number of training examples to keep')
    parser.add_argument('--sort_by_len', type='bool', default=False, help='Whether to sort examples by context length')

    # Script parameters:
    parser.add_argument('--seed', type=int, default=4213, help='Random seed')
    parser.add_argument('--test', type='bool', default=False, help='Use the presaved model.')

    args = parser.parse_args()
    print 'args:', args
    np.random.seed(args.seed)

    print "\nLoading data..."
    # data sets are dictionaries containing contexts, responses, flag
    train_data, val_data, test_data = cPickle.load(open('%s/%s' % (args.input_dir, args.dataset_fname), 'rb'))
    # W is the word embedding matrix and word2idx is a dictionary.
    W, word2idx = cPickle.load(open('%s/%s' % (args.input_dir, args.W_fname), 'rb'))
    print "W.shape:", W.shape  # (5092,300) = word embedding for each vocab word

    print('Number of training examples: %d' % (len(train_data['c'])))
    # Cap the number of training examples
    if args.train_examples:
        num_train_examples = args.train_examples + args.train_examples % args.batch_size
        train_data['c'] = train_data['c'][:num_train_examples]
        train_data['r'] = train_data['r'][:num_train_examples]
        train_data['y'] = train_data['y'][:num_train_examples]
        print('New number of training examples: %d' % (len(train_data['c'])))

    print("data loaded!")

    data = { 'train' : train_data, 'val': val_data, 'test': test_data }

    # sort the training data by context length
    if args.sort_by_len:
        sort_by_len(data['train'])

    print "\nCreating model..."
    model = Model(
        data=data,
        W=W.astype(theano.config.floatX),
        max_seqlen=args.max_seqlen,                         # default 160
        hidden_size=args.hidden_size,                       # default 200
        batch_size=args.batch_size,                         # default 256
        lr=args.lr,                                         # default 0.001
        lr_decay=args.lr_decay,                             # default 0.95
        sqr_norm_lim=args.sqr_norm_lim,                     # default 1 TODO: Can this be fixed?
        fine_tune_W=args.fine_tune_W,                       # default False
        fine_tune_M=args.fine_tune_M,                       # default False TODO: check what is M?
        use_ntn=args.use_ntn,                               # default False TODO: check if I can remove this
        optimizer=args.optimizer,                           # default ADAM
        forget_gate_bias=args.forget_gate_bias,             # default 2.0 TODO: Can this be fixed?
        conv_attn=args.conv_attn,                           # default False TODO: check if I can remove this
        encoder=args.encoder,                               # default RNN
        corr_penalty=args.corr_penalty,                     # default 0.0 TODO: check if I can remove this
        xcov_penalty=args.xcov_penalty,                     # default 0.0 TODO: check if I can remove this
        penalize_emb_norm=args.penalize_emb_norm,           # default False TODO: check if I can remove this
        penalize_emb_drift=args.penalize_emb_drift,         # default False TODO: check if I can remove this
        penalize_activations=args.penalize_activations,     # default False TODO: check if I can remove this
        emb_penalty=args.emb_penalty,                       # default 0.001 TODO: If I can remove the above, then I should remove this
        act_penalty=args.act_penalty,                       # default 500 TODO: If I can remove the above, then I should remove this
        k=args.k,                                           # default 4 TODO: If I can remove use_ntn, then I should remove this
        n_recurrent_layers=args.n_recurrent_layers,         # default 1
        is_bidirectional=args.is_bidirectional              # default False
    )
    print "Model created."

    # If testing a pre-saved model
    if args.test:
        "\nWill test the model:"
        # load parameter values into the network
        print "loading trained weights..."
        with open('weights_%s_best.pkl' % args.encoder, 'rb') as handle:
            params = cPickle.load(handle)
            lasagne.layers.set_all_param_values(model.l_out, params)
        # load the M matrix
        print "loading M matrix..."
        with open('M_%s_best.pkl' % args.encoder, 'rb') as handle:
            M = cPickle.load(handle)
            model.M.set_value(M)
        # load the W matrix
        print "loading W matrix..."
        with open('embed_%s_best.pkl' % args.encoder, 'rb') as handle:
            em = cPickle.load(handle)
            model.embeddings.set_value(em)
        # run the test function
        print "Testing model..."
        model.test()

    # If training the modeL
    else:
        "\nTraining model..."
        test_perf, test_probas = model.train(n_epochs=args.n_epochs)  # default 100
        "Model trained."
        print "test_perfs =", test_perf
        print "test_probas =", test_probas

        if args.save_model:
            print "\nSaving model..."
            cPickle.dump(model, open(args.model_fname, 'wb'))
            cPickle.dump(test_probas, open('probas_%s' % args.model_fname, 'wb'))
            print "Model saved."

if __name__ == '__main__':
  main()
