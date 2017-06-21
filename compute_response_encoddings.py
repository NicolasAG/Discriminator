import argparse
import cPickle as pkl
import numpy as np
import logging
import copy
import sys
from model import Model as DE_Model

logger = logging.getLogger(__name__)


def create_model(de_model, test=False):
    # Loading old arguments
    with open('%s_args.pkl' % de_model, 'rb') as handle:
        old_args = pkl.load(handle)

    logger.info("Loading retriever data...")
    # data sets are dictionaries containing contexts, responses, flag
    with open('./%s/%s' % (old_args.data_path, old_args.dataset_fname), 'rb') as handle:
        train_data, val_data, test_data = pkl.load(handle)
    # w_emb is the word embedding matrix and word2idx, idx2word are dictionaries
    with open('./%s/%s' % (old_args.data_path, old_args.W_fname), 'rb') as handle:
        w_emb, word2idx, idx2word = pkl.load(handle)
    logger.info("w_emb.shape: %s" % (w_emb.shape,))  # (5092,300) = word embedding for each vocab word

    logger.info("Number of training examples: %d" % len(train_data['c']))
    logger.info("Number of validation examples: %d" % len(val_data['c']))
    logger.info("Number of test examples: %d" % len(test_data['c']))

    # Cap the number of training examples
    if old_args.train_examples:
        num_train_examples = old_args.train_examples + old_args.train_examples % old_args.batch_size
        train_data['c'] = train_data['c'][:500]
        train_data['r'] = train_data['r'][:500]
        train_data['y'] = train_data['y'][:500]
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


def get_context_utterances(model):
    utterances = []
    for c in model.data['train']['c']:
        utt = np.split(c, np.where(np.asarray(c) == model.word2idx['__eot__'])[0]+1)  # list of utterances
        utterances.extend([u for u in utt if len(u) > 0])
    logger.debug("+ %d utterances from contexts" % len(utterances))
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
        response_set = copy.deepcopy(model.data['train']['r'])
        logger.debug("%d responses" % len(response_set))
        response_set.extend(get_context_utterances(model))
        logger.debug("= %d responses" % len(response_set))
        uniq = []
        # convert response_set from idx to string!!
        response_set_str = []
        for r_id, r in enumerate(response_set):
            if r_id % 100000 == 0:
                logger.debug("%d / %d" % (r_id, len(response_set)))
            if r not in uniq:
                uniq.append(r)
                response_set_str.append(model.indices2string(r))
    else:
        with open(args.responses_file, 'r') as handle:
            # this has to be strings formated in the same way as the DE model: BPE string if DE model has BPE dictionary, regular string otherwise
            response_set_str = handle.readlines()

    logger.info("Computing response encodding of %d unique responses..." % len(response_set_str))
    # Create a dumb context, just to compute the response embeddings
    retrieved_data = model.retrieve(
        context_set=["**unknown**"], response_set=response_set_str,
        k=1, batch_size=1, verbose=args.verbose
    )
    logger.info("Got %d encoddings" % len(retrieved_data['r_embs']))

    logger.info("Saving retrieved information to %s_r-encs.pkl..." % args.de_model)
    with open(args.de_model+'_r-encs.pkl', 'wb') as handle:
        pkl.dump(retrieved_data, handle, protocol=pkl.HIGHEST_PROTOCOL)
    logger.info("saved.")

if __name__ == '__main__':
    main()
