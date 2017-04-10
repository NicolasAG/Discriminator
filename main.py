from __future__ import division
import argparse
import cPickle
import lasagne
import numpy as np
import pyprind
import sys
import theano
import theano.tensor as T
import time

sys.setrecursionlimit(10000)


class Model:
    def __init__(self,
                 data,
                 W,
                 save_prefix,
                 max_seqlen=160,
                 batch_size=50,
                 # Network architecture:
                 encoder='rnn',
                 hidden_size=100,
                 n_recurrent_layers=1,
                 is_bidirectional=False,
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
        self.embeddings = theano.shared(W, name='embeddings', borrow=True)
        self.save_prefix = save_prefix
        self.max_seqlen = max_seqlen
        self.batch_size = batch_size
        # Learning parameters:
        self.patience = patience
        self.optimizer = optimizer
        self.lr = lr
        self.lr_decay = lr_decay
        self.fine_tune_W = fine_tune_W
        self.fine_tune_M = fine_tune_M
        self.use_ntn = use_ntn

        if penalize_emb_drift:
            self.orig_embeddings = theano.shared(W.copy(), name='orig_embeddings', borrow=True)

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
                print "Building a bidirectional LSTM model"
                for _ in xrange(n_recurrent_layers):
                    l_fwd = lasagne.layers.LSTMLayer(incoming=l_fwd,
                                                     num_units=hidden_size,  # number of hidden units in the layer
                                                     mask_input=l_mask,
                                                     backwards=False,   # forward pass
                                                     grad_clipping=10,  # avoid exploding gradients
                                                     learn_init=True)   # initial hidden values are learned

                    l_bck = lasagne.layers.LSTMLayer(incoming=l_bck,
                                                     num_units=hidden_size,  # number of hidden units in the layer
                                                     mask_input=l_mask,
                                                     backwards=True,    # backward pass
                                                     grad_clipping=10,  # avoid exploding gradients
                                                     learn_init=True)   # initial hidden values are learned
            elif encoder == 'gru':
                print "Building a bidirectional GRU model"
                for _ in xrange(n_recurrent_layers):
                    l_fwd = lasagne.layers.GRULayer(incoming=l_fwd,
                                                    num_units=hidden_size,  # number of hidden units in the layer
                                                    mask_input=l_mask,
                                                    backwards=False,   # forward pass
                                                    grad_clipping=10,  # avoid exploding gradients
                                                    learn_init=True)   # initial hidden values are learned
                    l_bck = lasagne.layers.GRULayer(incoming=l_bck,
                                                    num_units=hidden_size,  # Number of hidden units in the layer
                                                    mask_input=l_mask,
                                                    backwards=True,    # backward pass
                                                    grad_clipping=10,  # avoid exploding gradients
                                                    learn_init=True)   # initial hidden values are learned
            elif encoder == 'rnn':
                print "Building a bidirectional RNN model"
                for _ in xrange(n_recurrent_layers):
                    l_fwd = lasagne.layers.RecurrentLayer(incoming=l_fwd,
                                                          num_units=hidden_size,  # number of hidden units in the layer
                                                          mask_input=l_mask,
                                                          nonlinearity=lasagne.nonlinearities.tanh,  # Nonlinearity to apply when computing new state
                                                          W_in_to_hid=lasagne.init.Orthogonal(),     # Initializer for input-to-hidden weight matrix
                                                          W_hid_to_hid=lasagne.init.Orthogonal(),    # Initializer for hidden-to-hidden weight matrix
                                                          backwards=False,   # forward pass
                                                          grad_clipping=10,  # avoid exploding gradients
                                                          learn_init=True)   # initial hidden values are learned

                    l_bck = lasagne.layers.RecurrentLayer(incoming=l_bck,
                                                          num_units=hidden_size,  # number of hidden units in the layer
                                                          mask_input=l_mask,
                                                          nonlinearity=lasagne.nonlinearities.tanh,  # Nonlinearity to apply when computing new state
                                                          W_in_to_hid=lasagne.init.Orthogonal(),     # Initializer for input-to-hidden weight matrix
                                                          W_hid_to_hid=lasagne.init.Orthogonal(),    # Initializer for hidden-to-hidden weight matrix
                                                          backwards=True,    # backward pass
                                                          grad_clipping=10,  # avoid exploding gradients
                                                          learn_init=True)   # initial hidden values are learned
            else:
                raise ValueError("Unknown encoder %s", encoder)
            # concatenate forward and backward layers
            self.l_out = lasagne.layers.ConcatLayer([l_fwd, l_bck])

        else:
            l_recurrent = l_in
            if encoder == 'lstm':
                print "Building an LSTM model"
                for _ in xrange(n_recurrent_layers):
                    l_recurrent = lasagne.layers.LSTMLayer(incoming=l_recurrent,
                                                           num_units=hidden_size,  # number of hidden units in the layer
                                                           mask_input=l_mask,
                                                           grad_clipping=10,       # avoid exploding gradients
                                                           learn_init=True)        # initial hidden values are learned
            elif encoder == 'gru':
                print "Building a GRU model"
                for _ in xrange(n_recurrent_layers):
                    l_recurrent = lasagne.layers.GRULayer(incoming=l_recurrent,
                                                          num_units=hidden_size,  # number of hidden units in the layer
                                                          mask_input=l_mask,
                                                          grad_clipping=10,       # avoid exploding gradients
                                                          learn_init=True)        # initial hidden values are learned
            elif encoder == 'rnn':
                print "Building an RNN model"
                for _ in xrange(n_recurrent_layers):
                    l_recurrent = lasagne.layers.RecurrentLayer(incoming = l_recurrent,
                                                                num_units = hidden_size,  # number of hidden units in the layer
                                                                mask_input = l_mask,
                                                                nonlinearity = lasagne.nonlinearities.tanh,  # Nonlinearity to apply when computing new state
                                                                W_in_to_hid = lasagne.init.Orthogonal(),     # Initializer for input-to-hidden weight matrix
                                                                W_hid_to_hid = lasagne.init.Orthogonal(),    # Initializer for hidden-to-hidden weight matrix
                                                                grad_clipping = 10,  # avoid exploding gradients
                                                                learn_init = True)   # initial hidden values are learned
            else:
                raise ValueError("Unknown encoder %s", encoder)

            self.l_out = l_recurrent

        # Last hidden state of network after feeding it the context:
        h_context = lasagne.layers.helper.get_output(
            layer_or_layers=self.l_out,
            inputs={l_in: c_input, l_mask: self.c_mask},  # set parameters of the network units: l_in and l_mask
            deterministic=False
        )
        # Last hidden state of network after feeding it the response:
        h_response = lasagne.layers.helper.get_output(
            layer_or_layers=self.l_out,
            inputs={l_in: r_input, l_mask: self.r_mask},  # set parameters of the network units: l_in and l_mask
            deterministic=False
        )
        # Encoding of the context: take the encoding at the end of the context (self.c_seqlen)
        e_context = h_context[T.arange(batch_size), self.c_seqlen].reshape((batch_size, hidden_size))
        # Encoding of the response: take the encoding at the end of the response (self.r_seqlen)
        e_response = h_response[T.arange(batch_size), self.r_seqlen].reshape((batch_size, hidden_size))

        if use_ntn:
            dp = T.concatenate([T.batched_dot(e_context, T.dot(e_response, self.M[i])) for i in xrange(k)], axis=1)
            dp += T.concatenate([e_context, e_response], axis=1).dot(self.V.T) + self.b
            dp = self.f(dp).dot(self.U)
        else:
            dp = T.batched_dot(e_context, T.dot(e_response, self.M.T))

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
            params += [self.M]

        total_params = sum([p.get_value().size for p in params])
        print "total_params: ", total_params

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

        print "compiling theano functions..."
        self.train_model = theano.function(
            inputs=[],
            outputs=self.cost,
            updates=updates,
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
        self.get_pred = theano.function(
            inputs=[],
            outputs=self.pred,
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

    def compute_probas(self, dataset, index):
        self.set_shared_variables(dataset, index)
        return self.get_probas()[:,1]

    def compute_pred(self, dataset, index):
        self.set_shared_variables(dataset, index)
        return self.get_pred()

    def compute_loss(self, dataset, index):
        self.set_shared_variables(dataset, index)
        return self.get_loss()

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

    def train(self, n_epochs=100, patience=10, verbose=True):
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
        while epoch < n_epochs and patience > 0:
            epoch += 1
            epoch_cost = 0  # keep track of training cost for each epoch
            start_time = time.time()

            bar = pyprind.ProgBar(n_train_batches, monitor=True)  # show a progression bar on the screen
            print ""
            ############################
            # Loop through all batches #
            ############################
            for minibatch_index in range(n_train_batches):
                # Set context, response, flag, mask, and other variables for that batch index
                self.set_shared_variables(self.data['train'], minibatch_index)
                # Train model on this current batch
                batch_cost = self.train_model()
                if verbose: print "epoch %i: batch %i/%i cost: %f" % (epoch, minibatch_index+1, n_train_batches, batch_cost)
                epoch_cost += batch_cost
                self.set_zero(self.zero_vec)  # TODO: check what this does?
                bar.update()
            ### we trained the model on all the data once! ###

            end_time = time.time()
            print "epoch %i: training cost %f, took %d(s)" % (epoch, epoch_cost/n_train_batches, end_time-start_time)

            ###
            # Compute TRAIN performance:
            ###
            train_losses = [self.compute_loss(self.data['train'], i) for i in xrange(n_train_batches)]
            train_perf = 1 - np.sum(train_losses) / len(self.data['train']['y'])
            print "epoch %i: train perf %f" % (epoch, train_perf*100)
            # evaluation for each model id in data['train']['id']
            self.compute_performance_models("train")

            ###
            # Compute VALIDATION performance:
            ###
            val_losses = [self.compute_loss(self.data['val'], i) for i in xrange(n_val_batches)]
            val_perf = 1 - np.sum(val_losses) / len(self.data['val']['y'])
            print 'epoch %i: val_perf %f' % (epoch, val_perf*100)
            # evaluation for each model id in data['val']['id']
            self.compute_performance_models("val")

            ###
            # If doing better on validation set, measure test performance and same model parameters!
            ###
            if val_perf > best_val_perf:
                print "\nImproved validation score!"
                best_val_perf = val_perf
                patience = self.patience  # reset patience to initial value

                # Compute TEST performance:
                test_losses = [self.compute_loss(self.data['test'], i) for i in xrange(n_test_batches)]
                test_probas = [self.compute_probas(self.data['test'], i) for i in xrange(n_test_batches)]
                test_perf = 1 - np.sum(test_losses) / len(self.data['test']['y'])
                print 'epoch %i, test_perf %f' % (epoch, test_perf*100)
                # evaluation for each model id in data['test']['id']
                self.compute_performance_models("test")

                # Save current best model parameters.
                print "\nSaving current model parameters..."
                with open('%s_best_weights.pkl' % self.save_prefix, 'wb') as handle:
                    params = [np.asarray(p.eval()) for p in lasagne.layers.get_all_params(self.l_out)]
                    cPickle.dump(params, handle)
                with open('%s_best_embed.pkl' % self.save_prefix, 'wb') as handle:
                    cPickle.dump(self.embeddings.eval(), handle)
                with open('%s_best_M.pkl' % self.save_prefix, 'wb') as handle:
                    cPickle.dump(self.M.eval(), handle)
                print "Saved.\n"
            else:
                patience -= 1  # decrease patience

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
    parser.add_argument('--use_ntn', type='bool', default=False, help='Whether to use NTN')
    parser.add_argument('--k', type=int, default=4, help='Size of k in NTN')
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
    parser.add_argument('--fine_tune_M', type='bool', default=False, help='Whether to fine-tune context-response mapping M')

    # Size of the batches:
    parser.add_argument('--max_seqlen', type=int, default=160, help='Max seqlen')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')

    # Learning parameters:
    parser.add_argument('--n_epochs', type=int, default=100, help='Num epochs')
    parser.add_argument('--patience', type=int, default=10, help='Number of epochs to wait before exiting if no validation set improvement')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay')

    # Saving parameters:
    parser.add_argument('--save_model', type='bool', default=False, help='Whether to save the model')
    parser.add_argument('--save_prefix', type=str, default='twitter', help='prefix for all saved file names')

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
    W, word2idx, idx2word = cPickle.load(open('%s/%s' % (args.input_dir, args.W_fname), 'rb'))
    print "W.shape:", W.shape  # (5092,300) = word embedding for each vocab word

    print "Number of training examples: %d" % (len(train_data['c']))
    print "Number of validation examples: %d" % (len(val_data['c']))
    print "Number of test examples: %d" % (len(test_data['c']))

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
        save_prefix=args.save_prefix,
        max_seqlen=args.max_seqlen,                         # default 160
        batch_size=args.batch_size,                         # default 256
        # Network architecture:
        encoder=args.encoder,                               # default RNN
        hidden_size=args.hidden_size,                       # default 200
        n_recurrent_layers=args.n_recurrent_layers,         # default 1
        is_bidirectional=args.is_bidirectional,             # default False
        # Learning parameters:
        patience=args.patience,                             # default 10
        optimizer=args.optimizer,                           # default ADAM
        lr=args.lr,                                         # default 0.001
        lr_decay=args.lr_decay,                             # default 0.95
        fine_tune_W=args.fine_tune_W,                       # default False
        fine_tune_M=args.fine_tune_M,                       # default False
        use_ntn=args.use_ntn,                               # default False TODO: check if I can remove this
        k=args.k,                                           # default 4 TODO: If I can remove use_ntn, then I should remove this
        penalize_emb_norm=args.penalize_emb_norm,           # default False TODO: check if I can remove this
        penalize_emb_drift=args.penalize_emb_drift,         # default False TODO: check if I can remove this
        emb_penalty=args.emb_penalty,                       # default 0.001 TODO: If I can remove the above, then I should remove this
        penalize_activations=args.penalize_activations,     # default False TODO: check if I can remove this
        act_penalty=args.act_penalty                        # default 500 TODO: If I can remove the above, then I should remove this
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
        # load the M matrix: (from c.M.r)
        print "loading M matrix..."
        with open('M_%s_best.pkl' % args.encoder, 'rb') as handle:
            M = cPickle.load(handle)
            model.M.set_value(M)
        # load the word embeddings: W matrix
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
        test_perf, test_probas = model.train(n_epochs=args.n_epochs, patience=args.patience, verbose=False)  # default 100
        "Model trained."
        print "test_perfs =", test_perf
        print "test_probas =", test_probas

        if args.save_model:
            print "\nSaving model..."
            cPickle.dump(model, open("%s_model.pkl" % args.save_prefix, 'wb'))
            cPickle.dump(test_probas, open('%s_probas.pkl' % args.save_prefix, 'wb'))
            print "Model saved."

if __name__ == '__main__':
  main()
