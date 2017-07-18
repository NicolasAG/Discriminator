import argparse
import cPickle as pkl
import numpy as np
import logging
import copy
import sys
import pyprind
from model import Model as DE_Model

logger = logging.getLogger(__name__)

EOT_TOKEN = '</s>'


def create_model(de_model, test=False):
    # Loading old arguments
    with open('%s_args.pkl' % de_model, 'rb') as handle:
        old_args = pkl.load(handle)

    logger.info("Loading retriever data...")
    # data sets are dictionaries containing contexts, responses, flag
    with open('./%s/%s' % (old_args.data_path, old_args.dataset_fname), 'rb') as handle:
        train_data, val_data, test_data = pkl.load(handle)
    # word2idx, idx2word are dictionaries
    with open('./%s/%s' % (old_args.data_path, old_args.dict_fname), 'rb') as handle:
        word2idx, idx2word = pkl.load(handle)
    # start with random word embeddings for now, will set to the best ones after creating the model
    w_emb = np.zeros(shape=(len(word2idx), old_args.emb_size))
    for idx in idx2word:
        w_emb[idx] = np.random.uniform(-0.25, 0.25, args.emb_size)
    logger.info("w_emb.shape: %s" % (w_emb.shape,))  # (5092,300) = word embedding for each vocab word

    logger.info("Number of training examples: %d" % len(train_data['c']))
    logger.info("Number of validation examples: %d" % len(val_data['c']))
    logger.info("Number of test examples: %d" % len(test_data['c']))

    # Cap the number of training examples
    if old_args.train_examples:
        num_train_examples = old_args.train_examples + old_args.train_examples % old_args.batch_size
        train_data['c'] = train_data['c'][:num_train_examples]
        train_data['r'] = train_data['r'][:num_train_examples]
        train_data['y'] = train_data['y'][:num_train_examples]
        train_data['id'] = train_data['id'][:num_train_examples]
        logger.info('New number of training examples: %d' % len(train_data['c']))

    # sort the training data by context length
    if old_args.sort_by_len:
        c, r, y, model_name = train_data['c'], train_data['r'], train_data['y'], train_data['id']
        indices = range(len(y))
        indices.sort(key=lambda i: len(c[i]))
        for k in ['c', 'r', 'y', 'id']:
            train_data[k] = np.array(train_data[k])[indices]

    model = DE_Model(
        data={'train': train_data, 'val': val_data, 'test': test_data},
        W=w_emb.astype(theano.config.floatX),
        word2idx=word2idx,
        idx2word=idx2word,
        save_path=old_args.save_path,
        save_prefix=old_args.save_prefix,
        max_seqlen=old_args.max_seqlen,                         # default 160
        batch_size=old_args.batch_size,                         # default 256
        # Network architecture:
        encoder=old_args.encoder,                               # default RNN
        hidden_size=old_args.hidden_size,                       # default 200
        n_recurrent_layers=old_args.n_recurrent_layers,         # default 1
        is_bidirectional=old_args.is_bidirectional,             # default False
        dropout_out=old_args.dropout_out,                       # default 0.
        dropout_in=old_args.dropout_in,                         # default 0.
        # Learning parameters:
        patience=old_args.patience,                             # default 10
        optimizer=old_args.optimizer,                           # default ADAM
        lr=old_args.lr,                                         # default 0.001
        lr_decay=old_args.lr_decay,                             # default 0.95
        fine_tune_W=old_args.fine_tune_W,                       # default False
        fine_tune_M=old_args.fine_tune_M,                       # default False
        # NTN parameters:
        use_ntn=old_args.use_ntn,                               # default False
        k=old_args.k,                                           # default 4
        # Regularization parameters:
        penalize_emb_norm=old_args.penalize_emb_norm,           # default False
        penalize_emb_drift=old_args.penalize_emb_drift,         # default False
        emb_penalty=old_args.emb_penalty,                       # default 0.001
        penalize_activations=old_args.penalize_activations,     # default False
        act_penalty=old_args.act_penalty                        # default 500
    )

    logger.info("Set the learned weights...")
    with open('%s_best_weights.pkl' % de_model, 'rb') as handle:
        params = pkl.load(handle)
        lasagne.layers.set_all_param_values(model.l_out, params)
    with open('%s_best_M.pkl' % de_model, 'rb') as handle:
        M = pkl.load(handle)
        model.M.set_value(M)
    with open('%s_best_embed.pkl' % de_model, 'rb') as handle:
        em = pkl.load(handle)
        model.embeddings.set_value(em)

    if test:
        logger.info("Testing the model...")
        model.test()
    
    return model


def unique(alist, verbose=False):
    """
    Remove duplicate 'rows' from a 2D numpy array
    Remove duplicated utterances
    :return: array of unique utterances
    """
    if verbose: logger.debug("  removing duplicates...")
    unique = set(tuple(e) for e in alist)  # remove duplicates by converting to tuples
    return [list(e) for e in unique]  # convert each element back to an array
    '''# pad with -1 so that each utterance is of the same length
    if verbose: logger.debug("  pading with -1 so that each utterance is of the same length...")
    max_length = max([len(u) for u in alist])
    alist = np.asarray( [np.pad(u, (0, max_length-len(u)), 'constant', constant_values=(-1,-1)) for u in alist] )
    # remove duplicates
    if verbose: logger.debug("  removing duplicates...")
    order = np.lexsort(alist.T)
    alist = alist[order]
    diff = np.diff(alist, axis=0)
    ui = np.ones(len(alist), 'bool')
    ui[1:] = (diff != 0).any(axis=1) 
    alist = alist[ui]
    # remove the padded -1's from the utterances
    if verbose: logger.debug("  remove padded -1's from utterances...")
    stop = [np.where(u == -1)[0] for u in alist]
    stop = [end_idx[0] if len(end_idx)>0 else max_length for end_idx in stop]
    return [alist[idx, :end] for idx, end in enumerate(stop)]'''


def get_context_utterances(model):
    utterances = []
    for c_idx, c in enumerate(model.data['train']['c']):
        if c_idx % 100000 == 0:
            logger.debug("get context utterances progress: %d / %d" % (c_idx, len(model.data['train']['c'])))
        utt = np.split(c, np.where(np.asarray(c) == model.word2idx[EOT_TOKEN])[0]+1)  # list of utterances
        utt = unique(utt)
        utterances.extend([u for u in utt if len(u) > 0])
    logger.debug(" %d utterances" % len(utterances))
    utterances = unique(utterances, verbose=True)
    logger.debug("+ %d unique utterances from contexts" % len(utterances))
    return utterances


def parse_args():
    parser = argparse.ArgumentParser("Compute the ecodding of a list of responses with a given DE model")
    parser.add_argument("de_model", help="Path to the dual encoder model prefix")
    parser.add_argument("--responses_file", default="", help="Path to the text file of responses to encode")
    parser.add_argument("-v", "--verbose", action="store_true", help="be verbose")
    return parser.parse_args()


def main():
    logging.basicConfig(level=getattr(logging, 'DEBUG'), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s", stream=sys.stdout)

    args = parse_args()
    logger.info(args)

    logger.info("Loading %s_model..." % args.de_model)
    try:
        with open('%s_model.pkl' % args.de_model, 'rb') as handle:
            model = pkl.load(handle)
    except pkl.UnpicklingError:
        logger.error("cPickle.UnpicklingError: couldn't load the model")
        logger.info("Creating a new one...")
        model = create_model(args.de_model, test=False)

    with open('%s_timings.pkl' % args.de_model, 'rb') as handle:
        timings = pkl.load(handle)
        model.timings = timings  # load last timings (when no improvement was done)
    logger.info("Model loaded.")

    logger.info("Gathering responses to encode...")
    if args.responses_file == "":
        response_set = copy.deepcopy(model.data['train']['r'])  # get responses from model training set
        logger.debug("%d responses" % len(response_set))
        response_set = unique(response_set, verbose=True)  # remove duplicates
        logger.debug("= %d unique responses" % len(response_set))
        response_set.extend(get_context_utterances(model))  # get utterances seen in every training context
        logger.debug("= %d responses" % len(response_set))
        
        # convert response_set from idx to string
        response_set_str = []
        for r_id, r in enumerate(response_set):
            if r_id % 100000 == 0:
                logger.debug("index to string conversion progress: %d / %d" % (r_id, len(response_set)))
            response_set_str.append(model.indices2string(r))
    else:
        with open(args.responses_file, 'r') as handle:
            # this has to be strings formated in the same way as the DE model: BPE string if DE model has BPE dictionary, regular string otherwise
            response_set_str = handle.readlines()
            response_set = []
            for r_id, r in enumerate(response_set_str):
                if r_id % 100000 == 0:
                    logger.debug("string to idx conversion progress: %d / %d" % (r_id, len(response_set_str)))
                response_set.append(model.string2indices(r))

    logger.info("Computing response encodding of %d unique responses..." % len(response_set))
    # Pad response set to be divisible by batch_size
    length = len(response_set)
    while len(response_set) % model.batch_size != 0:
        response_set.append(response_set[0])
    # Compute embeddings
    n_batches = len(response_set) // model.batch_size
    response_embs = []  # list of response embeddings
    if n_batches > 100: bar = pyprind.ProgBar(n_batches, monitor=True, stream=sys.stdout)  # show a progression bar on the screen
    for i in xrange(n_batches):
        response_embs.extend(model.compute_response_embeddings(response_set, i))
        if n_batches > 100: bar.update()
    if verbose: print ""
    # Ignore padded embeddings
    response_embs = response_embs[:length]
    response_set = response_set[:length]
    assert len(response_set_str) == len(response_set) == len(response_embs)

    retrieved_data = {
        'c': [], 'c_embs': [],
        'r': response_set_str,
        'r_embs': response_embs,
        'r_retrieved': [],
        'r_retrieved_embs': [],
        'proba_retrieved': []
    }
    logger.info("Got %d encoddings" % len(retrieved_data['r_embs']))

    logger.info("Saving retrieved information to %s_r-encs.pkl..." % args.de_model)
    with open(args.de_model+'_r-encs.pkl', 'wb') as handle:
        pkl.dump(retrieved_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
    logger.info("saved.")

if __name__ == '__main__':
    main()
