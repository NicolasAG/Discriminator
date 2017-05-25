import numpy as np
import cPickle

from sklearn.feature_extraction.text import TfidfVectorizer

from bpe.apply_bpe import BPE


def process_dialogues(dialogues):
    ''' Removes </d> </s> at end, splits into contexts/ responses '''
    contexts = []
    responses = []
    for d in dialogues:
        d_proc = d[:-3]
        index_list = [i for i, j in enumerate(d_proc) if j == 1]
        split = index_list[-1] + 1
        context = filter(lambda idx: idx!=1, d_proc[:split])  # remove </s> from context
        contexts.append(context)
        response = filter(lambda idx: idx!=1, d_proc[split:])  # remove </s> from response
        responses.append(response)
    return contexts, responses

def strs_to_idxs(data, bpe, str_to_idx):
    ''' Encodes strings in BPE form '''
    out = []
    for row in data:
        bpe_segmented = bpe.segment(row.strip())
        # Note: there shouldn't be any unknown tokens with BPE!
        #out.append([str_to_idx[word] for word in bpe_segmented.split()])
        out.append([str_to_idx[word] for word in bpe_segmented.split() if word in str_to_idx])

    return out

def idxs_to_strs(data, bpe, idx_to_str):
    ''' Converts from BPE form to strings '''
    out = []
    for row in data:
        out.append(' '.join([idx_to_str[idx] for idx in row if idx in idx_to_str]).replace(bpe.separator+' ',''))
    return out

def idxs_to_bpestrs(data, idx_to_str):
    ''' Converts from BPE form to strings '''
    out = []
    for row in data:
        out.append(' '.join([idx_to_str[idx] for idx in row if idx in idx_to_str]))
    return out

def bpestrs_to_strs(data, bpe):
    out = []
    for row in data:
        out.append(row.replace(bpe.separator+' ',''))
    return out


def tfidf_c(contexts_str, responses_str, batch_size=100):
    """
    For each context, finds the most similar and return its corresponding answer.
    :param contexts_str: list of contexts
    :param responses_str: list of responses
    :param batch_size: number of contexts to consider at a time.
    :returns: list of retrieved responses.
    """
    vec = TfidfVectorizer()
    print "\nFitting vectorizer..."
    tfidf_contexts = vec.fit_transform(contexts_str)
    print "tfidf_contexts shape:", tfidf_contexts.shape  # (#_contexts, context-tfidf-vector-size)
    print "done."

    hit = 0
    preds = []
    print "\nRetrieving responses based on tfidf_contexts..."
    for i in range(0, len(contexts_str), batch_size):
        end = min(i+batch_size, len(contexts_str))
        print "%d-->%d / %d" % (i+1, end, len(contexts_str))
        tfidf = vec.transform([contexts_str[idx] for idx in range(i, end)])  # encodes this set of contexts into a tfidf vector
        print "  tfidf shape:", tfidf.shape  # (batch_size, context-tfidf-vector-size)
        dot_products = tfidf * tfidf_contexts.T
        print "  dot_products shape:", dot_products.shape  # (batch_size, #_context)
        dot_products = dot_products.toarray().T
        print "  dot_products shape:", dot_products.shape  # (#_context, batch_size)
        best_context_idx = np.argmax(dot_products, axis=0)  # index of most similar context for each context -- (batch_size,)
        # print " best_context_idx =", best_context_idx
        for j, jdx in enumerate(best_context_idx):
            if jdx == i+j:  # if retrieved the same context, take the second best.
                # print "   HIT! take second best..."
                hit += 1
                dot_products[jdx][j] = 0  # set vector[jdx][j] to 0 so that it's not picked again.
        best_context_idx = np.argmax(dot_products, axis=0)  # take the second best -- (batch_size,)
        # print " best_context_idx =", best_context_idx
        preds.extend([responses_str[jdx] for jdx in best_context_idx])  # add corresponding response
    print "done."
    print "tfidf_c hit %d " % hit
    print "Retrieved %d responses" % len(preds)
    return preds


def tfidf_r(contexts_str, responses_str, batch_size=100):
    """
    EXPERIMENTAL USE ONLY.
    For each response, find the most similar context and return its corresponding response.
    THIS RESULTS IN PRETTY BAD RESPONSES.
    :param contexts_str: list of contexts
    :param responses_str: list of responses
    :param batch_size: number of contexts to consider at a time.
    :returns: list of retrieved responses.
    """
    vec = TfidfVectorizer()
    print "\nFitting vectorizer..."
    tfidf_responses = vec.fit_transform(responses_str)
    print "tfidf_responses shape:", tfidf_responses.shape  # (#_response, response-tfidf-vector-size)
    print "done."

    hit = 0
    preds = []
    print "\nRetrieving responses based on tfidf_responses..."
    for i in range(0, len(contexts_str), batch_size):
        end = min(i+batch_size, len(contexts_str))
        print "%d-->%d / %d" % (i+1, end, len(contexts_str))
        tfidf = vec.transform([contexts_str[idx] for idx in range(i, end)])  # encodes this set of contexts into a response tfidf vector
        print "  tfidf shape:", tfidf.shape  # (batch_size, response-tfidf-vector-size)
        dot_products = tfidf * tfidf_responses.T
        print "  dot_products shape:", dot_products.shape  # (batch_size, #_response)
        dot_products = dot_products.toarray().T
        print "  dot_products shape:", dot_products.shape  # (#_response, batch_size)
        best_context_idx = np.argmax(dot_products, axis=0)  # index of most similar context for each context -- (batch_size,)
        print " best_context_idx =", best_context_idx
        for j, jdx in enumerate(best_context_idx):
            if jdx == i+j:  # if retrieved the same context, take the second best.
                print "   HIT! take second best..."
                hit += 1
                dot_products[jdx][j] = 0  # set vector[jdx][j] to 0 so that it's not picked again.
        best_context_idx = np.argmax(dot_products, axis=0)  # take the second best -- (batch_size,)
        print " best_context_idx =", best_context_idx
        preds.extend([responses_str[jdx] for jdx in best_context_idx])  # add corresponding response
    print "done."
    print "tfidf_r hit %d " % hit
    print "Retrieved %d responses" % len(preds)
    return preds


def vecs_to_textfile(vecs, filename):
    print "\nSaving results to %s" % filename
    with open(filename, 'w') as f:
        for vec in vecs:
            f.write(vec + "\n")
    print "done."


if __name__ == '__main__':
    print "Loading Twitter data..."
    twitter_bpe_dictionary = './bpe/Twitter_Codes_5000.txt'
    twitter_model_dictionary = './bpe/Dataset.dict-5k.pkl'

    # Load in Twitter dictionaries
    twitter_bpe = BPE(open(twitter_bpe_dictionary, 'rb').readlines())
    twitter_dict = cPickle.load(open(twitter_model_dictionary, 'rb'))
    twitter_str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in twitter_dict])
    twitter_idx_to_str = dict([(tok_id, tok) for tok, tok_id, _, _ in twitter_dict])

    # Get data for Twitter
    train_file = './bpe/Train.dialogues.pkl'
    # val_file = './bpe/Valid.dialogues.pkl'
    # test_file = './bpe/Test.dialogues.pkl'

    f1 = open(train_file, 'rb')
    train_data = cPickle.load(f1)
    f1.close()
    # f1 = open(val_file, 'rb')
    # val_data = cPickle.load(f1)
    # f1.close()
    # f1 = open(test_file, 'rb')
    # test_data = cPickle.load(f1)
    # f1.close()
    print "Data loaded."

    train_contexts, train_responses = process_dialogues(train_data)
    # val_contexts, val_responses = process_dialogues(val_data)
    # test_contexts, test_responses = process_dialogues(test_data)

    train_contexts_str = idxs_to_strs(train_contexts, twitter_bpe, twitter_idx_to_str)
    train_responses_str = idxs_to_strs(train_responses, twitter_bpe, twitter_idx_to_str)
    # val_contexts_str = idxs_to_strs(val_contexts, twitter_bpe, twitter_idx_to_str)
    # val_responses_str = idxs_to_strs(val_responses, twitter_bpe, twitter_idx_to_str)
    # test_contexts_str = idxs_to_strs(test_contexts, twitter_bpe, twitter_idx_to_str)
    # test_responses_str = idxs_to_strs(test_responses, twitter_bpe, twitter_idx_to_str)

    print "Number of contexts:", len(train_contexts_str)
    print "Number of responses:", len(train_responses_str)

    c_pred = tfidf_c(train_contexts_str, train_responses_str, batch_size=1000)
    vecs_to_textfile(c_pred, "./c_tfidf_responses.txt")

    # r_pred = tfidf_r(train_contexts_str, train_responses_str)
    # vecs_to_textfile(r_pred, "./r_tfidf_responses.txt")


