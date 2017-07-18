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


def string2indices(p_str, str_to_idx, bpe=None):
    """
    Lookup dictionary from word to index to retrieve the list of indices from a string of words.
    If bpe is present, will automatically convert regular string p_str to bpe formatted string.
    :param p_str: string of words corresponding to indices.
    :param str_to_idx: contains the mapping from words to indices.
    :param bpe: byte pair encoding object from apply_bpe.py
    :return: a new list of indices corresponding to the given string of words.
    """
    if bpe:
        bpe_string = bpe.segment(p_str.strip())  # convert from regular to bpe format
        return [str_to_idx[w] for w in bpe_string.split() if w in str_to_idx]
    else:
        return [str_to_idx[w] for w in p_str.strip().split() if w in str_to_idx]

def indices2string(p_indices, idx_to_str, bpe=None):
    """
    Lookup dictionary from word to index to retrieve the list of words from a list of indices.
    :param p_indices: list of indices corresponding to words.
    :param idx_to_str: contains the mapping from indices to words.
    :param bpe: byte pair encoding object from apply_bpe.py
    :return: a new string corresponding to the given list of indices.
    """
    if bpe:
        return ' '.join([idx_to_str[idx] for idx in p_indices if idx in idx_to_str]).replace(bpe.separator+' ', '')
    else:
        return ' '.join([idx_to_str[idx] for idx in p_indices if idx in idx_to_str])

def process_twitter_dialogues(dialogues):
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

def process_ubuntu_dialogues(dialogues, str_to_idx):
    """
    splits list of dialogues into lists of contexts & responses like so:
    context = first 2,3,4,... utterances & response the next utterance
    :param dialogues: list of idx formated dialogues
    :param str_to_idx: dictionary from string to idx
    :return: list of contexts, list of responses
    """
    contexts = []
    responses = []
    for d in dialogues:
        # d_proc = d[:-1]  # remove the last '__eou__' token
        index_list = [i for i, j in enumerate(d) if j == str_to_idx['__eot__']]  # split at every __eot__
        if len(index_list) < 2:
            print "ignoring short dialogues: # of __eot__ = %d < 2" % len(index_list)
        for i, start_idx in enumerate(index_list):
            if i == 0:
                continue  # take at least 2 turns for the context
            elif i == len(index_list)-1:  # if last __eot__ token:
                end_idx = len(d)  # response goes until the end of the dialogue
            else:
                end_idx = index_list[i+1]
            context = filter(lambda idx: idx!=str_to_idx['__eou__'], d[:start_idx])  # remove all __eou__ and last __eot__ from context
            contexts.append(context)
            response = filter(lambda idx: idx!=str_to_idx['__eou__'], d[start_idx+1:end_idx])  # remove all __eou__ and first __eot__ from response
            responses.append(response)
    return contexts, responses

def sort_by_len(dataset):
    """
    Sort a given data set by its context length
    :param data set: dictionary of contexts, responses, flags
    :return: ordered dictionary
    """
    c, r, y, model_name = dataset['c'], dataset['r'], dataset['y'], dataset['id']
    indices = range(len(y))
    indices.sort(key=lambda i: len(c[i]))
    for k in ['c', 'r', 'y', 'id']:
        dataset[k] = np.array(dataset[k])[indices]

def create_model(data, w, word2idx, idx2word, args):
    return DE_Model(
        data=data,
        W=w.astype(theano.config.floatX),
        word2idx=word2idx,
        idx2word=idx2word,
        save_path=args.save_path,
        save_prefix=args.save_prefix,
        max_seqlen=args.max_seqlen,                         # default 160
        batch_size=args.batch_size,                         # default 256
        # Network architecture:
        encoder=args.encoder,                               # default RNN
        hidden_size=args.hidden_size,                       # default 200
        n_recurrent_layers=args.n_recurrent_layers,         # default 1
        is_bidirectional=args.is_bidirectional,             # default False
        dropout_out=args.dropout_out,                       # default 0.
        dropout_in=args.dropout_in,                         # default 0.
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

def pad_to_batch_size(x, batch_size):
    to_pad = len(x) % batch_size
    if to_pad > 0:
        x += x[:batch_size-to_pad]
    return x

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
    parser.add_argument('--emb_size', type=int, default=300, help="bpe token embedding size")
    parser.add_argument('--encoder', type=str, default='rnn', help='Type of encoding RNN units: rnn, gru, lstm')
    parser.add_argument('--hidden_size', type=int, default=200, help='Hidden size')
    parser.add_argument('--is_bidirectional', type='bool', default=False, help='Bidirectional RNN')
    parser.add_argument('--n_recurrent_layers', type=int, default=1, help='Num recurrent layers')
    parser.add_argument('--dropout_out', type=float, default=0., help='Dropout probability of input layer')
    parser.add_argument('--dropout_in', type=float, default=0., help='Dropout probability of output layer')

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
    parser.add_argument('--dict_fname', type=str, default='W.pkl', help='Word dictionary filename')
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
    # W is the word embedding matrix and word2idx, idx2word are dictionaries
    with open('%s/%s' % (args.data_path, args.dict_fname), 'rb') as handle:
        word2idx, idx2word = cPickle.load(handle)
    W = np.zeros(shape=(len(word2idx), args.emb_size))
    for idx in idx2word:
        W[idx] = np.random.uniform(-0.25, 0.25, args.emb_size)
    logger.info("W.shape: %s" % (W.shape,))

    logger.info("Number of training examples: %d" % len(train_data['c']))
    if len(train_data['c']) % args.batch_size > 0:
        for key in ['c', 'r', 'y', 'id']:
            train_data[key] = pad_to_batch_size(train_data[key], args.batch_size)
        logger.info("(padded to batch size) New number of training examples: %d" % len(train_data['c']))
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
            logger.error("cPickle.UnpicklingError: couldn't load the model")
            # Loading old arguments
            with open('%s/%s_args.pkl' % (args.load_path, args.load_prefix), 'rb') as handle:
                old_args = cPickle.load(handle)
            
            logger.info("Creating a new one...")
            model = create_model(data, W, word2idx, idx2word, old_args)

            logger.info("Set the learned weights...")
            with open('%s/%s_best_weights.pkl' % (args.load_path, args.load_prefix), 'rb') as handle:
                params = cPickle.load(handle)
                lasagne.layers.set_all_param_values(model.l_out, params)
            with open('%s/%s_best_M.pkl' % (args.load_path, args.load_prefix), 'rb') as handle:
                M = cPickle.load(handle)
                model.M.set_value(M)
            with open('%s/%s_best_embed.pkl' % (args.load_path, args.load_prefix), 'rb') as handle:
                em = cPickle.load(handle)
                model.embeddings.set_value(em)

            logger.info("Testing the model...")
            model.test()
            logger.info("")
            logger.info("Does the performance look good? If so, will save the model. (y/n): ")
            ok = raw_input("")
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
        model = create_model(data, W, word2idx, idx2word, args)
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
        logger.info("")
        logger.info("plot score by length...")
        model.plot_score_per_length('train')
        model.plot_score_per_length('val')
        model.plot_score_per_length('test')

    if args.plot_learning_curves:
        logger.info("")
        logger.info("plot learning curves...")
        model.plot_learning_curves('train')
        model.plot_learning_curves('val')

    # If testing a pre-saved model
    if args.test:
        logger.info("")
        logger.info("Testing the model...")
        model.test()

    # If retrieving responses from training set
    elif args.retrieve:
        logger.info("Need to think about that...")
        # logger.info("Loading original data from %s/Training.dialogues.pkl..." % args.data_path)
        # with open('%s/Training.dialogues.pkl' % args.data_path, 'rb') as handle:
        #     train_conversations = cPickle.load(handle)
        # train_contexts, train_responses = process_ubuntu_dialogues(train_conversations, word2idx)
        # logger.info("number of contexts: %d" % len(train_contexts))
        # logger.info("number of responses: %d" % len(train_responses))

        # # Convert idx to string
        # train_contexts_str = []
        # for c in train_contexts:
        #     train_contexts_str.append(indices2string(c, idx2word))
        # train_responses_str = []
        # for r in train_responses:
        #     train_responses_str.append(indices2string(r, idx2word))

        # logger.info("")
        # logger.info("Retrieving responses...")
        # retrieved_responses = model.retrieve(context_set=train_contexts_str, response_set=train_responses_str, k=10, batch_size=1000)
        # with open("%s/%s_Train_retrieved_responses.pkl" % (args.save_path, args.save_prefix), 'wb') as handle:
        #     cPickle.dump(retrieved_responses, handle, protocol=cPickle.HIGHEST_PROTOCOL)
        # logger.info('saved.')

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

