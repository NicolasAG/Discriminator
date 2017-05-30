from __future__ import division
import logging
import argparse
import cPickle
import numpy as np
import theano
import lasagne

import sys
sys.setrecursionlimit(10000)

from model import Model as DE_Model

logger = logging.getLogger(__name__)


def string2indices(p_str, str_to_idx):
    """
    Lookup dictionary from word to index to retrieve the list of indices from a string of words.
    If bpe is present, will automatically convert regular string p_str to bpe formatted string.
    :param p_str: string of words corresponding to indices.
    :param str_to_idx: contains the mapping from words to indices.
    :return: a new list of indices corresponding to the given string of words.
    """
    return [str_to_idx[w] for w in p_str.strip().split() if w in str_to_idx]

def process_dialogues(dialogues):
    """
    get the list of contexts, and the list of TRUE responses.
    Removes </d> </s> at end, splits into contexts/ responses
    """
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

def create_model(data, w, args):
    return DE_Model(
        data=data,
        W=w.astype(theano.config.floatX),
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

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def main():
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s", stream=sys.stdout)
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
    logger.info('args: %s' % args)
    # Saving args to file
    with open("%s/%s_args.pkl" % (args.save_path, args.save_prefix), 'wb') as handle:
        cPickle.dump(args, handle, protocol=cPickle.HIGHEST_PROTOCOL)
    logger.info('saved.')

    np.random.seed(args.seed)

    logger.info("")
    logger.info("Loading data...")
    # data sets are dictionaries containing contexts, responses, flag
    with open('%s/%s' % (args.data_path, args.dataset_fname), 'rb') as handle:
        train_data, val_data, test_data = cPickle.load(handle)
    # W is the word embedding matrix and word2idx is a dictionary.
    with open('%s/%s' % (args.data_path, args.W_fname), 'rb') as handle:
        W, word2idx, idx2word = cPickle.load(handle)
    logger.info("W.shape: %s" % (W.shape,))  # (5092,300) = word embedding for each vocab word

    logger.info("Number of training examples: %d" % len(train_data['c']))
    logger.info("Number of validation examples: %d" % len(val_data['c']))
    logger.info("Number of test examples: %d" % len(test_data['c']))

    # Cap the number of training examples
    if args.train_examples:
        num_train_examples = args.train_examples + args.train_examples % args.batch_size
        train_data['c'] = train_data['c'][:num_train_examples]
        train_data['r'] = train_data['r'][:num_train_examples]
        train_data['y'] = train_data['y'][:num_train_examples]
        logger.info('New number of training examples: %d' % len(train_data['c']))

    data = {'train': train_data, 'val': val_data, 'test': test_data}
    logger.info("data loaded!")

    # sort the training data by context length
    if args.sort_by_len:
        sort_by_len(data['train'])

    if args.resume or args.test or args.retrieve:
        logger.info("")
        logger.info("Loading model...")
        try:
            with open('%s/%s_model.pkl' % (args.load_path, args.load_prefix), 'rb') as handle:
                model = cPickle.load(handle)
                model.data = data  # in cases when we want to test or resume with new data
        except cPickle.UnpicklingError:
            logger.info("cPickle.UnpicklingError: couldn't load the model")
            # Loading old arguments
            with open('%s/%s_args.pkl' % (args.load_path, args.load_prefix), 'rb') as handle:
                old_args = cPickle.load(handle)
            
            logger.info("Creating a new one...")
            model = create_model(data, W, old_args)

            logger.info("Set the learned weights...")
            with open('%s/%s_best_weights.pkl' % (args.load_path, args.load_prefix), 'rb') as handle:
                params = cPickle.load(handle)
                lasagne.layers.set_all_param_values(model.l_out, params)
            with open('%s/%s_best_M.pkl' % (args.load_path, args.load_prefix), 'rb') as handle:
                M = cPickle.load(handle)
                model.M.set_value(M)
            with open('%s/%s_best_embed' % (args.load_path, args.load_prefix), 'rb') as handle:
                em = cPickle.load(handle)
                model.embeddings.set_value(em)

            logger.info("Testing the model...")
            model.test()
            ok = raw_input("\nDoes the performance look good? If so, will save the model. (y/n): ")
            if ok == 'y':
                logger.info("Saving new model in %s/%s_model.pkl" % (args.save_path, args.save_prefix))
                with open("%s/%s_model.pkl" % (args.save_path, args.save_prefix), 'wb') as handle:
                    cPickle.dump(model, handle, protocol=cPickle.HIGHEST_PROTOCOL)
                logger.info("Saved.")
            else:
                logger.info("Not saved.")

        with open('%s/%s_timings.pkl' % (args.load_path, args.load_prefix), 'rb') as handle:
            timings = cPickle.load(handle)
            model.timings = timings  # load last timings (when no improvement was done)
        logger.info("Model loaded.")
    else:
        logger.info("")
        logger.info("Creating model...")
        model = create_model(data, W, args)
        logger.info("Model created.")

    # Get correlation with Human scores:
    if args.plot_human_scores:
        # Load human scores
        logger.info("")
        logger.info("Loading human score data...")
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
        logger.info("")
        logger.info("Testing the model...")
        model.test()

    # If retrieving responses from training set
    elif args.retrieve:
        logger.info("")
        logger.info("Loading original twitter data...")
        with open('%s/bpe/Train.dialogues.pkl' % args.data_path, 'rb') as handle:
            train_conversations = cPickle.load(handle)
        train_contexts, train_responses = process_dialogues(train_conversations)
        logger.info("number of contexts: %d" % len(train_contexts))
        logger.info("number of responses: %d" % len(train_responses))

        logger.info("")
        logger.info("Retrieving responses...")
        retrieved_responses = model.retrieve(context_set=train_contexts, response_set=train_responses, k=10, batch_size=100)
        with open("%s/%s_Train_retrieved_responses.pkl" % (args.save_path, args.save_prefix), 'wb') as handle:
            cPickle.dump(retrieved_responses, handle, protocol=cPickle.HIGHEST_PROTOCOL)
        logger.info('saved.')

    # If training the model
    else:
        logger.info("")
        logger.info("Training model...")
        test_perf = model.train(n_epochs=args.n_epochs, patience=args.patience, verbose=False)
        logger.info("Model trained.")
        logger.info("test_perfs = %.9f" % test_perf)
        # logger.info("test_probas = %.9f" % test_probas)

if __name__ == '__main__':
    main()

