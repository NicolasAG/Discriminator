import argparse
import cPickle
import random
import copy
import numpy.random as np_rnd
from datetime import datetime


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


def process_dialogues(dialogues, str_to_idx):
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


def print_k(elements, idx_2_str=None, ubuntu_bpe=None, k=10):
    # start = random.randint(0, len(elements)-k-1)
    start = 0
    if idx_2_str:
        top_list = map(lambda e : indices2string(e, idx_2_str, ubuntu_bpe), elements[start:start+k])
    else:
        top_list = elements[start:start+k]
    for e in top_list:
        print e


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() in ("yes", "true", "t", "1"))
    parser.add_argument('--data_dir', type=str, default='.', help='Input/Output directory to find original data and save new data')
    parser.add_argument('--data_fname_prefix', type=str, default='dataset', help='File name of new data')
    parser.add_argument('--data_embeddings_prefix', type=str, default='W', help='File name of new data embeddings')
    parser.add_argument('--embedding_size', type=int, default=300, help='Size of word embedding')
    parser.add_argument('--random_model', type='bool', default='True', help='Flag to add a random retrieval model as part of the new data')
    args = parser.parse_args()
    print "args: ", args

    print "\nLoading original ubuntu data..."
    ###
    # LOAD WORD DICTIONARIES: map word_indices/words - vocab ~ 20,000
    ###
    ubuntu_dict = cPickle.load(open('%s/words/Dataset.dict.pkl' % args.data_dir, 'rb'))
    ubuntu_str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in ubuntu_dict])
    ubuntu_idx_to_str = dict([(tok_id, tok) for tok, tok_id, _, _ in ubuntu_dict])
    print "Word dictionary length: ", len(ubuntu_dict)
    ###
    # Load the original Twitter dataset - only TRUE responses
    ###
    train_dialogues = cPickle.load(open('%s/words/Training.dialogues.pkl' % args.data_dir, 'rb'))
    train_contexts, train_true_responses = process_dialogues(train_dialogues, ubuntu_str_to_idx)  # get contexts and TRUE responses.
    print "number of train dialogues: %d - produced %d context-response pairs" % (len(train_dialogues), len(train_contexts))
    val_dialogues = cPickle.load(open('%s/words/Validation.dialogues.pkl' % args.data_dir, 'rb'))
    val_contexts, val_true_responses = process_dialogues(val_dialogues, ubuntu_str_to_idx)  # get contexts and TRUE responses.
    print "number of val dialogues: %d - produced %d context-response pairs" % (len(val_dialogues), len(val_contexts))
    test_dialogues = cPickle.load(open('%s/words/Test.dialogues.pkl' % args.data_dir, 'rb'))
    test_contexts, test_true_responses = process_dialogues(test_dialogues, ubuntu_str_to_idx)  # get contexts and TRUE responses.
    print "number of test dialogues: %d - produced %d context-response pairs" % (len(test_dialogues), len(test_contexts))

    print "\ntrain contexts:"
    print_k(train_contexts, ubuntu_idx_to_str)
    print "train true responses:"
    print_k(train_true_responses, ubuntu_idx_to_str)

    ###
    # CREATE THE DATA SET
    ###
    print "\nCreating new dataset..."
    data = {
        'train': {'c': [], 'r': [], 'y': [], 'id': []},
        'val': {'c': [], 'r': [], 'y': [], 'id': []},
        'test': {'c': [], 'r': [], 'y': [], 'id': []},
    }

    # add TRUE responses to the data train, validation, and test sets.
    data['train']['c'].extend(train_contexts)
    data['train']['r'].extend(train_true_responses)
    data['train']['y'].extend([1] * len(train_true_responses))
    data['train']['id'].extend(['true'] * len(train_true_responses))

    data['val']['c'].extend(val_contexts)
    data['val']['r'].extend(val_true_responses)
    data['val']['y'].extend([1] * len(val_true_responses))
    data['val']['id'].extend(['true'] * len(val_true_responses))

    data['test']['c'].extend(test_contexts)
    data['test']['r'].extend(test_true_responses)
    data['test']['y'].extend([1] * len(test_true_responses))
    data['test']['id'].extend(['true'] * len(test_true_responses))

    if args.random_model:
        # get the list of RANDOM responses.
        train_random_responses = random.sample(train_true_responses, len(train_true_responses))
        # random.shuffle(train_random_responses)
        val_random_responses = random.sample(val_true_responses, len(val_true_responses))
        # random.shuffle(val_random_responses)
        test_random_responses = random.sample(test_true_responses, len(test_true_responses))
        # random.shuffle(test_random_responses)
        
        print "\ntrain random responses:"
        print_k(train_random_responses, ubuntu_idx_to_str)
        print ""

        # add RANDOM responses to the data train, validation, and test sets.
        data['train']['c'].extend(train_contexts)
        data['train']['r'].extend(train_random_responses)
        data['train']['y'].extend([0] * len(train_random_responses))
        data['train']['id'].extend(['rand'] * len(train_random_responses))

        data['val']['c'].extend(val_contexts)
        data['val']['r'].extend(val_random_responses)
        data['val']['y'].extend([0] * len(val_random_responses))
        data['val']['id'].extend(['rand'] * len(val_random_responses))

        data['test']['c'].extend(test_contexts)
        data['test']['r'].extend(test_random_responses)
        data['test']['y'].extend([0] * len(test_random_responses))
        data['test']['id'].extend(['rand'] * len(test_random_responses))

    # Making sure each context has a unique response, flag, and model_id
    assert len(data['train']['c']) == len(data['train']['r']) == len(data['train']['y']) == len(data['train']['id'])
    assert len(data['val']['c']) == len(data['val']['r']) == len(data['val']['y']) == len(data['val']['id'])
    assert len(data['test']['c']) == len(data['test']['r']) == len(data['test']['y']) == len(data['test']['id'])

    print "New dataset created!"
    if args.random_model:
        print 'New number of training examples: true+rand = %d' % len(data['train']['c'])
        print 'New number of validation examples: true+rand = %d' % len(data['val']['c'])
        print 'New number of testing examples: true+rand = %d' % len(data['test']['c'])
    else:
        print 'New number of training examples: true = %d' % len(data['train']['c'])
        print 'New number of validation examples: true = %d' % len(data['val']['c'])
        print 'New number of testing examples: true = %d' % len(data['test']['c'])

    ###
    # SHUFFLE THE WHOLE TRAINING SET
    ###
    SEED = datetime.now()
    random.seed(SEED)
    random.shuffle(data['train']['c'])
    random.seed(SEED)
    random.shuffle(data['train']['r'])
    random.seed(SEED)
    random.shuffle(data['train']['y'])
    random.seed(SEED)
    random.shuffle(data['train']['id'])

    ###
    # SAVE THE RESULTING DATA
    # .pkl will have (data[train], data[val], data[test])
    ###
    print "\nSaving resulting dataset in %s/%s_ubuntu_words.pkl..." % (args.data_dir, args.data_fname_prefix)
    data_file = open("%s/%s_ubuntu_words.pkl" % (args.data_dir, args.data_fname_prefix), 'wb')
    cPickle.dump((data['train'], data['val'], data['test']), data_file, protocol=cPickle.HIGHEST_PROTOCOL)
    data_file.close()
    print "Saved."

    ###
    # SAVE RANDOM WORD EMBEDDINGS
    # .pkl will have (word embeddings, str_to_idx map)
    ###
    print "\nSaving random word embeddings in %s/%s_%s_ubuntu_words.pkl..." % (args.data_dir, args.data_embeddings_prefix, args.embedding_size)
    vocab_size = len(ubuntu_dict)
    random_word_embeddings = np_rnd.random((vocab_size, args.embedding_size))
    w_file = open("%s/%s_%d_ubuntu_words.pkl" % (args.data_dir, args.data_embeddings_prefix, args.embedding_size), 'wb')
    cPickle.dump((random_word_embeddings, ubuntu_str_to_idx, ubuntu_idx_to_str), w_file, protocol=cPickle.HIGHEST_PROTOCOL)
    w_file.close()
    print "Saved."


if __name__ == '__main__':
    main()
