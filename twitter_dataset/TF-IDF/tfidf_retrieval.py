import numpy as np
import cPickle

from sklearn.feature_extraction.text import TfidfVectorizer

from apply_bpe import BPE



def preprocess_tweet(s):
    s = s.replace('@user', '<at>').replace('&lt;heart&gt;', '<heart>').replace('&lt;number&gt;', '<number>').replace('  ', ' </s> ').replace('  ', ' ')
    # Make sure we end with </s> token
    while s[-1] == ' ':
        s = s[0:-1]
    if not s[-5:] == ' </s>':
        s = s + ' </s>'
    return s

def process_dialogues(dialogues):
    ''' Removes </d> </s> at end, splits into contexts/ responses '''
    contexts = []
    responses = []
    for d in dialogues:
        d_proc = d[:-3]
        index_list = [i for i, j in enumerate(d_proc) if j == 1]
        split = index_list[-1] + 1
        contexts.append(d_proc[:split])
        responses.append(d_proc[split:] + [1])
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
        out.append(' '.join([idx_to_str[idx] for idx in row if idx in idx_to_str]).replace('@@ ',''))
    return out

def idxs_to_bpestrs(data, bpe, idx_to_str):
    ''' Converts from BPE form to strings '''
    out = []
    for row in data:
        out.append(' '.join([idx_to_str[idx] for idx in row if idx in idx_to_str]))
    return out

def bpestrs_to_strs(data):
    out = []
    for row in data:
        out.append(row.replace('@@ ',''))
    return out

def flatten_list(l1):
    return [i for sublist in l1 for i in sublist]

def brute_force_search(train_emb, query_emb):
    max_index = -1
    largest_product = -1e9
    for i in xrange(len(train_emb)):
        prod = np.dot(train_emb[i], query_emb)
        if prod > largest_product:
            largest_product = prod
            max_index = i
    return max_index, largest_product

def mat_vector_2norm(mat):
    '''
    Takes as input a matrix, and returns a vector correponding to the 2-norm
    of each row vector.
    '''
    norm_list = []
    for i in xrange(mat.shape[0]):
        norm_list.append(np.sqrt(np.dot(mat[0], mat[0].T)))
    return np.array(norm_list)

def mat_vector_2norm_squared(mat):
    '''
    Takes as input a matrix, and returns a vector correponding to the squared
    2-norm of each row vector.
    '''
    norm_list = []
    for i in xrange(mat.shape[0]):
        norm_list.append(np.dot(mat[0], mat[0].T))
    return np.array(norm_list)


def tfidf_retrieval(tfidf_vec, train_contexts_txt, train_responses_txt, output_file):
    # tfidf_vec = tfidf_vec.toarray()
    # print "type of tfidf_vec :", type(tfidf_vec)
    # print "tfidf_vec shape :", tfidf_vec.shape
    prod_mat = np.dot(tfidf_vec, tfidf_vec.T)
    print "prod_mat shape :", prod_mat.shape
    prod_mat = prod_mat / mat_vector_2norm_squared(tfidf_vec)
    print "prod_mat shape :", prod_mat.shape
    
    response_list = []
    for i in xrange(len(prod_mat)):
        row = np.array(prod_mat[i])
        # No idea what's going on here. See the following page:
        # stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
        ind = np.argpartition(row, -2)[-2:]
        ind = ind[np.argsort(row[ind])][0]
        response_list.append(train_responses_txt[ind])
        print train_contexts_txt[i]
        print "-->", response_list[i]
        print ""

    print "Saving responses..."
    with open(output_file, 'w') as f1:
        for response in response_list:
            f1.write(response)
    print "Saved."


def tfidf_c(contexts_str, responses_str):
    vec = TfidfVectorizer()
    print "\nFitting vectorizer..."
    tfidf_contexts = vec.fit_transform(contexts_str)
    print "tfidf_contexts shape:", tfidf_contexts.shape
    print "done."

    hit = 0
    preds = []
    print "\nRetrieving responses based on tfidf_contexts..."
    for i, context in enumerate(contexts_str):
        if i % 10 == 0:
            tfidf = vec.transform([context])
            dot_products = tfidf * tfidf_contexts.T
            dot_products = dot_products.toarray().T
            idx = np.argmax(dot_products)
            if idx == i / 10:
                dot_products[idx] = 0
                hit += 1
                idx = np.argmax(dot_products)
            preds.append(responses_str[idx])
    print "done."
    print "tfidf_c hit %d " % hit
    print "Retrieved %d responses" % len(preds)
    return preds


def tfidf_r(contexts_str, responses_str):
    vec = TfidfVectorizer()
    print "\nFitting vectorizer..."
    tfidf_responses = vec.fit_transform(responses_str)
    print "tfidf_responses shape:", tfidf_responses.shape
    print "done."

    hit = 0
    preds = []
    print "\nRetrieving responses based on tfidf_responses..."
    for i, context in enumerate(contexts_str):
        tfidf = vec.transform([context])
        dot_products = tfidf * tfidf_responses.T
        dot_products = dot_products.toarray().T
        idx = np.argmax(dot_products)
        if idx == i / 10:
            hit += 1
            dot_products[idx] = 0
            idx = np.argmax(dot_products)
        preds.append(responses_str[idx])
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
    twitter_bpe_dictionary = '../BPE/Twitter_Codes_5000.txt'
    twitter_model_dictionary = '../BPE/Dataset.dict-5k.pkl'

    # Load in Twitter dictionaries
    twitter_bpe = BPE(open(twitter_bpe_dictionary, 'r').readlines())
    twitter_dict = cPickle.load(open(twitter_model_dictionary, 'r'))
    twitter_str_to_idx = dict([(tok, tok_id) for tok, tok_id, _, _ in twitter_dict])
    twitter_idx_to_str = dict([(tok_id, tok) for tok, tok_id, _, _ in twitter_dict])

    # Get data for Twitter
    train_file = '../BPE/Train.dialogues.pkl'
    #val_file = '../BPE/Valid.dialogues.pkl'
    #test_file = '../BPE/Test.dialogues.pkl'

    f1 = open(train_file, 'rb')
    train_data = cPickle.load(f1)
    f1.close()
    #f1 = open(val_file, 'rb')
    #val_data = cPickle.load(f1)
    #f1.close()
    #f1 = open(test_file, 'rb')
    #test_data = cPickle.load(f1)
    #f1.close()
    print "Data loaded."

    train_contexts, train_responses = process_dialogues(train_data)
    #val_contexts, val_responses = process_dialogues(val_data)
    #test_contexts, test_responses = process_dialogues(test_data)

    train_contexts_txt = idxs_to_bpestrs(train_contexts, twitter_bpe, twitter_idx_to_str)
    train_responses_txt = idxs_to_bpestrs(train_responses, twitter_bpe, twitter_idx_to_str)
    #val_contexts_txt = idxs_to_bpestrs(val_contexts, twitter_bpe, twitter_idx_to_str)
    #val_responses_txt = idxs_to_bpestrs(val_responses, twitter_bpe, twitter_idx_to_str)
    #test_contexts_txt = idxs_to_strs(test_contexts, twitter_bpe, twitter_idx_to_str)
    #test_responses_txt = idxs_to_strs(test_responses, twitter_bpe, twitter_idx_to_str)

    c_pred = tfidf_c(train_contexts_txt, train_responses_txt)
    r_pred = tfidf_r(train_contexts_txt, train_responses_txt)

    vecs_to_textfile(c_pred, "./c_tfidf_responses.txt")
    vecs_to_textfile(r_pred, "./r_tfidf_responses.txt")

    '''
    print '\nFitting vectorizer...'
    vectorizer = TfidfVectorizer()
    vectorizer.fit(train_contexts_txt + train_responses_txt)
    c_vec = vectorizer.transform(train_contexts_txt)
    r_vec = vectorizer.transform(train_responses_txt)
    print "r_vec shape:", r_vec.shape
    print "c_vec shape:", c_vec.shape
    print "done."
    
    print "\nRetrieving responses based on r_vec..."
    tfidf_retrieval(r_vec, train_contexts_txt, train_responses_txt, './rtfidf_responses.txt')
    print "done."

    print "\nRetrieving responses based on c_vec..."
    tfidf_retrieval(c_vec, train_contexts_txt, train_responses_txt, './ctfidf_responses.txt')
    print "done."
    '''


