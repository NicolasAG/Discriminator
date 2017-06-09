import cPickle
from collections import Counter


def main():
    print "\nLoading original ubuntu data..."
    ###
    # Load the original Ubuntu dataset - only TRUE responses
    ###
    with open('./big/dataset.pkl', 'rb') as handle:
        train_data, val_data, test_data = cPickle.load(handle)
    print "train: %d" % len(train_data['c'])
    print "val: %d" % len(val_data['c'])
    print "test: %d" % len(test_data['c'])

    with open('./big/vocab.pkl', 'rb') as handle:
        vocab = cPickle.load(handle)
    print "vocab length: %d" % len(vocab)

    with open('./big/W.pkl', 'rb') as handle:
        w, word_to_idx = cPickle.load(handle)
    print "w shape: %s" % (w.shape,)
    print "word_to_idx length: %d" % len(word_to_idx)

    # Add [id] key in datasets
    print "\nTRAINING SET"
    train_data['id'] = []
    for idx, context in enumerate(train_data['c']):
        if idx % 10000 == 0:
            indices = [i for i, c2 in enumerate(train_data['c']) if c2==context]
            print indices
            if len(indices) > 1:
                for jdx in indices:
                    print "  y=%s" % train_data['y'][jdx]
                    print "  same response" if train_data['r'][jdx] == train_data['r'][idx] else "  different response"
        if train_data['y'][idx] == '0':
            train_data['y'][idx] = 0  # set to int type
            train_data['id'].append('rand')  # add id = 'rand'
        elif train_data['y'][idx] == '1':
            train_data['y'][idx] = 1  # set to int type
            train_data['id'].append('true')  # add id = true
        else:
            print "ERROR: this should never happen"
            return

    print "\nVALIDATION SET"
    val_data['id'] = []
    for idx, context in enumerate(val_data['c']):
        if idx % 10000 == 0:
            indices = [i for i, c2 in enumerate(val_data['c']) if c2==context]
            print indices
            if len(indices) > 1:
                for jdx in indices:
                    print "  y=%s" % val_data['y'][jdx]
                    print "  same response" if val_data['r'][jdx] == val_data['r'][idx] else "  different response"
        if val_data['y'][idx] == '0':
            val_data['y'][idx] = 0  # set to int type
            val_data['id'].append('rand')  # add id = 'rand'
        elif val_data['y'][idx] == '1':
            val_data['y'][idx] = 1  # set to int type
            val_data['id'].append('true')  # add id = true
        else:
            print "ERROR: this should never happen"
            return

    print "\nTEST SET"
    test_data['id'] = []
    for idx, context in enumerate(test_data['c']):
        if idx % 10000 == 0:
            indices = [i for i, c2 in enumerate(test_data['c']) if c2==context]
            print indices
            if len(indices) > 1:
                for jdx in indices:
                    print "  y=%s" % test_data['y'][jdx]
                    print "  same response" if test_data['r'][jdx] == test_data['r'][idx] else "  different response"
        if test_data['y'][idx] == '0':
            test_data['y'][idx] = 0  # set to int type
            test_data['id'].append('rand')  # add id = 'rand'
        elif test_data['y'][idx] == '1':
            test_data['y'][idx] = 1  # set to int type
            test_data['id'].append('true')  # add id = true
        else:
            print "ERROR: this should never happen"
            return

    with open("./big/dataset_prepared.pkl", "wb") as handle:
        cPickle.dump((train_data, val_data, test_data), handle, protocol=cPickle.HIGHEST_PROTOCOL)
    print "Saved new dataset."

    print "\nBuilding reverse dictionary: idx -> word"
    # Build reverse dictionary
    idx_to_word = {}
    for word, idx in word_to_idx.iteritems():
        idx_to_word[idx] = word

    with open("./big/W_300_ubuntu_big.pkl", 'wb') as handle:
        cPickle.dump((w, word_to_idx, idx_to_word), handle, protocol=cPickle.HIGHEST_PROTOCOL)
    print "Saved new W file."


if __name__ == '__main__':
    main()
