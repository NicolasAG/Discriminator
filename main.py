from __future__ import division
import argparse
import cPickle
import numpy as np
import theano
import lasagne

import sys
sys.setrecursionlimit(10000)

from model import Model as DE_Model

def string2indices(p_str, str_to_idx):
    """
    Lookup dictionary from word to index to retrieve the list of indices from a string of words.
    If bpe is present, will automatically convert regular string p_str to bpe formatted string.
    :param p_str: string of words corresponding to indices.
    :param str_to_idx: contains the mapping from words to indices.
    :return: a new list of indices corresponding to the given string of words.
    """
    return [str_to_idx[w] for w in p_str.strip().split() if w in str_to_idx]

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
    parser.register('type', 'bool', str2bool)

    # NTN parameters:
    parser.add_argument('--use_ntn', type='bool', default=False, help='Whether to use NTN')
    parser.add_argument('--k', type=int, default=4, help='Size of k in NTN')

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

    # Regularization parameters:
    parser.add_argument('--penalize_emb_norm', type='bool', default=False, help='Whether to penalize norm of embeddings')
    parser.add_argument('--penalize_emb_drift', type='bool', default=False, help='Whether to use re-embedding words penalty')
    parser.add_argument('--penalize_activations', type='bool', default=False, help='Whether to penalize activations')
    parser.add_argument('--emb_penalty', type=float, default=0.001, help='Embedding penalty')
    parser.add_argument('--act_penalty', type=float, default=500, help='Activation penalty')

    # Loading data parameters:
    parser.add_argument('--data_path', type=str, default='.', help='Path of the data to load')
    parser.add_argument('--dataset_fname', type=str, default='dataset.pkl', help='Dataset filename')
    parser.add_argument('--W_fname', type=str, default='W.pkl', help='Word embeddings filename')
    parser.add_argument('--train_examples', type=int, required=False, help='Number of training examples to keep')
    parser.add_argument('--sort_by_len', type='bool', default=False, help='Whether to sort examples by context length')

    # Saving model parameters:
    parser.add_argument('--save_path', type=str, default='.', help='Path where to save model')
    parser.add_argument('--save_prefix', type=str, default='twitter', help='prefix for all saved file names')

    # Loading model parameters:
    parser.add_argument('--load_path', type=str, default='.', help='Path to load the model from')
    parser.add_argument('--load_prefix', type=str, default='twitter', help='Prefix for all the model files to load')

    # Script parameters:
    parser.add_argument('--resume', type='bool', default=False, help='Resume training from a pre-trained model or not')
    parser.add_argument('--retrieve', type='bool', default=False, help='Return responses for each context')
    parser.add_argument('--seed', type=int, default=4213, help='Random seed')
    parser.add_argument('--test', type='bool', default=False, help='Get test accuracies with pre-saved model')
    # Plot parameters:
    parser.add_argument('--plot_human_scores', type='bool', default=False, help='Plot model score correlation with human scores')
    parser.add_argument('--plot_response_length', type='bool', default=False, help='Plot model score correlation with response length')
    parser.add_argument('--plot_learning_curves', type='bool', default=False, help='Plot train & val learning curves')

    args = parser.parse_args()
    print 'args:', args
    np.random.seed(args.seed)

    print "\nLoading data..."
    # data sets are dictionaries containing contexts, responses, flag
    train_data, val_data, test_data = cPickle.load(open('%s/%s' % (args.data_path, args.dataset_fname), 'rb'))
    # W is the word embedding matrix and word2idx is a dictionary.
    W, word2idx, idx2word = cPickle.load(open('%s/%s' % (args.data_path, args.W_fname), 'rb'))
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

    data = {'train': train_data, 'val': val_data, 'test': test_data}
    print("data loaded!")

    # sort the training data by context length
    if args.sort_by_len:
        sort_by_len(data['train'])

    if args.resume or args.test or args.retrieve:
        print "\nLoading model..."
        with open('%s/%s_model.pkl' % (args.load_path, args.load_prefix), 'rb') as handle:
            model = cPickle.load(handle)
            model.data = data  # in cases when we want to test or resume with new data
        with open('%s/%s_timings.pkl' % (args.load_path, args.load_prefix), 'rb') as handle:
            timings = cPickle.load(handle)
            model.timings = timings  # load last timings (when no improvement was done)
        print "Model loaded."
    else:
        print "\nCreating model..."
        model = DE_Model(
            data=data,
            W=W.astype(theano.config.floatX),
            save_path=args.save_path,
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
            # NTN parameters:
            use_ntn=args.use_ntn,                               # default False
            k=args.k,                                           # default 4
            # Regularization parameters:
            penalize_emb_norm=args.penalize_emb_norm,           # default False
            penalize_emb_drift=args.penalize_emb_drift,         # default False
            emb_penalty=args.emb_penalty,                       # default 0.001
            penalize_activations=args.penalize_activations,     # default False
            act_penalty=args.act_penalty                        # default 500
        )
        print "Model created."

    # Get correlation with Human scores:
    if args.plot_human_scores:
        # Load human scores
        print "\nLoading human score data..."
        with open('../data/twitter/human_scores-vhred_embeddings.pkl', 'rb') as handle:
            vhred_embeddings = cPickle.load(handle)

        human_scores = {'c': [], 'r': [], 'score': []}
        for data in vhred_embeddings:
            # data of the form {'c':<str>, 'r_gt':<str>, 'r_models':{
            #     'hred':<str>, 'tfidf':<str>, 'de':<str>, 'human':<str>
            # }, ... }
            for model_responses in data['r_models'].values():
                # model_responses of the form [<str>, <score>, <??>]
                resp = model_responses[0]
                score = model_responses[1]
                human_scores['c'].append(string2indices(data['c'], word2idx))
                human_scores['r'].append(string2indices(resp, word2idx))
                human_scores['score'].append(score)

        # normalize scores to 0-1 probability
        scores = human_scores['score']
        human_scores['score'] = (scores-np.min(scores)) / (np.max(scores)-np.min(scores))

        model.plot_human_correlation(human_scores)

    if args.plot_response_length:
        model.plot_score_per_length('train')
        model.plot_score_per_length('val')
        model.plot_score_per_length('test')

    if args.plot_learning_curves:
        model.plot_learning_curves('train')
        model.plot_learning_curves('val')

    # If testing a pre-saved model
    if args.test:
        print "\nTesting the model..."
        model.test()

    # If retrieving responses from training set
    elif args.retrieve:
        print "\nGenerating responses..."
        # TODO: run tfidf retriever on model.data['train'] to get a smaller list of responses
        retrieved_responses = model.retrieve('test', k=10, response_set=None)

    # If training the model
    else:
        "\nTraining model..."
        test_perf = model.train(n_epochs=args.n_epochs, patience=args.patience, verbose=False)
        "Model trained."
        print "test_perfs =", test_perf
        # print "test_probas =", test_probas

if __name__ == '__main__':
    main()

    ###
    # DEBUG compute_embedding
    ###
    """
    print "\nLoading original twitter data..."
    dialogues = cPickle.load(open('./twitter_dataset/bpe/Train.dialogues.pkl', 'rb'))

    # get the list of contexts, and the list of TRUE responses.
    def process_dialogues(dialogues):
        '''Removes </d> </s> at end, splits into contexts/ responses '''
        contexts = []
        responses = []
        for d in dialogues:
            d_proc = d[:-3]
            index_list = [i for i, j in enumerate(d_proc) if j == 1]
            split = index_list[-1] + 1
            context = filter(lambda idx: idx != 1, d_proc[:split])  # remove </s> from context
            contexts.append(context)
            response = filter(lambda idx: idx != 1, d_proc[split:])  # remove </s> from response
            responses.append(response)
        return contexts, responses

    contexts, true_responses = process_dialogues(dialogues)
    print "number of contexts:", len(contexts)
    print "number of responses:", len(true_responses)

    n_batches = len(contexts) // model.batch_size
    c_embs = []
    for i in xrange(n_batches):
        c_embs.extend(model.compute_context_embeddings(contexts, i))
    print "number of c_embs:", len(c_embs)
    print c_embs[0]

    n_batches = len(true_responses) // model.batch_size
    r_embs = []
    for i in xrange(n_batches):
        r_embs.extend(model.compute_response_embeddings(true_responses, i))
    print "number of r_embs:", len(r_embs)
    print r_embs[0]
    """

