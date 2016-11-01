import sys, string, random, os, re
from optparse import OptionParser
import nltk
import numpy as np
from sklearn import linear_model
from dependency_decoder import DependencyDecoder
from scipy.sparse import csr_matrix

parser = OptionParser()
parser.add_option("-f","--file", dest="file", help="Input File")
parser.add_option("-v","--verbose", dest="verbose", help="Print lots of stuff")
parser.add_option("-t","--testfile", dest="testfile", help="Test File")
parser.add_option("-r","--trainfile", dest="trainfile", help="Train File")
parser.add_option("-c","--cutoff", dest="cut", help="Number of Tokens", default="10")
parser.add_option("-s","--iterations1", dest="iter1", help="Iteration Starts", default="0")
parser.add_option("-e","--iterations2", dest="iter2", help="Iteration Ends", default="20")
parser.add_option("-g","--gflweight", dest="gflw", help="Gfl Weight", default="10")
parser.add_option("-u","--ugweight", dest="ugw", help="UG Weight", default="10")
parser.add_option("-w","--withgfl", dest="wgfl", help="If Train with GFL", default="true")
(options, args) = parser.parse_args()

gfl_sentences = open(options.file).read().rstrip("\n").split("\n\n")
#train_sentences = [i for i in train_sentences if len(i.split('\n')) <= 10]
gfl_sentence_lens = [len(i.split('\n')) for i in gfl_sentences]
print(gfl_sentence_lens[-1])
print(sum([x*x for x in gfl_sentence_lens]))
#train_sentences = open("en-univiersal-train.conll").read().split("\n\n")

test_sentences = open(options.testfile).read().rstrip("\n").split("\n\n")
test_sentences = [i for i in test_sentences if len(i.split('\n')) <= int(options.cut)]
test_sentence_lens = [len(i.split('\n')) for i in test_sentences]
print(test_sentence_lens[-1])
print(sum([x*x for x in test_sentence_lens]))


y_test = np.load("test_y_init_array_np.npy")
test_row_ind = np.load("test_row_ind_array_np.npy")
test_col_ind = np.load("test_col_ind_array_np.npy")
test_x_data = np.load("test_data_array_np.npy")

train_sentences = open(options.trainfile).read().rstrip("\n").split("\n\n")
train_sentences = [i for i in train_sentences if len(i.split('\n')) <= 10]
train_sentence_lens = [len(i.split('\n')) for i in train_sentences]
print(train_sentence_lens[-1])
print(sum([x*x for x in train_sentence_lens]))

#x_test = coo_matrix((test_x_data, (test_row_ind, test_col_ind)), shape = (20483, 164640))

#anno_test = open("test_anno","r").read().split("\n")
#anno_test = [int(i) for i in anno_test]
#anno_test = np.load("test_anno_np.npy")


y_train = np.load("train_y_init_array_np.npy")
train_row_ind = np.load("train_row_ind_array_np.npy")
train_col_ind = np.load("train_col_ind_array_np.npy")
train_x_data = np.load("train_data_array_np.npy")

#x_test = coo_matrix((test_x_data, (test_row_ind, test_col_ind)), shape = (20483, 164640))

#anno_test = open("test_anno","r").read().split("\n")
#anno_test = [int(i) for i in anno_test]
anno_train = np.load("train_anno_np.npy")



y_gfl = np.load("gfl_y_init_array_np.npy")
gfl_row_ind = np.load("gfl_row_ind_array_np.npy")
gfl_col_ind = np.load("gfl_col_ind_array_np.npy")
gfl_x_data = np.load("gfl_data_array_np.npy")


#y_gfl_ug = np.load("gfl_ug_y_init_array_np.npy")
#gfl_ug_row_ind = np.load("gfl_ug_row_ind_array_np.npy")
#gfl_ug_col_ind = np.load("gfl_ug_col_ind_array_np.npy")
#gfl_ug_x_data = np.load("gfl_ug_data_array_np.npy")


anno_gfl = np.load("gfl_anno_gfl_np.npy")

anno_ug = np.load("gfl_anno_ug_np.npy")



if(options.wgfl == "true"):
    train_row_ind = train_row_ind + len(anno_gfl)
    anno = np.concatenate((anno_gfl*float(options.gflw) + anno_ug*float(options.ugw), anno_train), axis = 0)
    sen_length = gfl_sentence_lens + train_sentence_lens
    y = np.concatenate((y_gfl, y_train), axis = 0)
    x_row_ind = np.concatenate((gfl_row_ind, train_row_ind), axis = 0)
    x_col_ind = np.concatenate((gfl_col_ind, train_col_ind), axis = 0)
    x_data = np.concatenate((gfl_x_data, train_x_data), axis = 0)
    x = csr_matrix((x_data, (x_row_ind, x_col_ind)), shape = ((len(anno_gfl) + len(anno_train)), 172872))
    design_mat = x
else:
    anno = anno_train
    sen_length = train_sentence_lens
    y = y_train
    x_row_ind = train_row_ind
    x_col_ind = train_col_ind
    x_data = train_x_data
    x = csr_matrix((x_data, (x_row_ind, x_col_ind)), shape = (len(anno_train), 172872))
    design_mat = x
    



y_t = y
test_design_mat = csr_matrix((test_x_data, (test_row_ind, test_col_ind)), shape = (len(y_test), 172872))
total_num_arcs = len(y)



dd = DependencyDecoder()

#T = int(options.iter)
alpha = 0.001
miu = 0.1/total_num_arcs
#y_t = np.array(y)
#design_mat = np.matrix(x)
#test_design_mat = np.matrix(x_test)

print(design_mat.shape)
for t in xrange(int(options.iter1), int(options.iter2)):
    gamma_t = 2.0/(t+2.0)
    # w = minimizer for 1/2n||y_t - Xw|| + Lambda/2 ||w||
    clf = linear_model.SGDRegressor(alpha = alpha, penalty = 'elasticnet', l1_ratio = 0.0, n_iter = 10)
    clf.fit(design_mat, y_t)
    w_t = clf.coef_
    intercept_t = clf.intercept_
    # compute the gradient w.r.t. y
    test = design_mat * (np.mat(w_t).transpose()) + intercept_t
    test = y_t - np.asarray(test).reshape(-1)
    test = test/total_num_arcs
    test2 = miu*np.array(anno)
    g_t = test - miu * np.array(anno)
    # solve the linear program by minimum spanning tree
    # loop over all sentences
    #start.arc.index = 1
    #s_t = c()  # binary indicator whether in or not in parse tree
    s_t = np.zeros(total_num_arcs)
    pt = 0
    for lens in sen_length:
        scores = np.mat(np.zeros((lens+1,lens+1)))
        arc_idx = {}
        for i in range(0,lens+1):
            if i == 0:
                for j in range(1, lens+1):
                    scores[i,j] = -g_t[pt]
                    arc_idx[(i,j)] = pt
                    pt = pt+1
            else:
                for j in range(1, lens+1):
                    if i != j:
                        scores[i,j] = -g_t[pt]
                        arc_idx[(i,j)] = pt
                        pt = pt+1
        # minimum spanning tree 
        head_t = dd.parse_proj(scores)
        for i in range(1,lens):
            s_t[arc_idx[(head_t[i],i)]] = 1
    print(pt)
    y_t = gamma_t*s_t + (1 - gamma_t)*y_t
    ### round solution and evaluation
    pt = 0
    n_tokens = 0
    n_correct = 0
    sen_index = 0
    for lens in test_sentence_lens:
        scores = np.mat(np.zeros((lens+1,lens+1)))
        arc_idx = {}
        for i in range(0,lens+1):
            if i == 0:
                for j in range(1, lens+1):
                    scores[i,j] = (test_design_mat.getrow(pt).dot(np.array(w_t)))[0]
                    arc_idx[(i,j)] = pt
                    pt = pt+1
            else:
                for j in range(1, lens+1):
                    if i != j:
                        scores[i,j] = (test_design_mat.getrow(pt).dot(np.array(w_t)))[0]
                        arc_idx[(i,j)] = pt
                        pt = pt+1
        # minimum spanning tree 
        head_t = dd.parse_proj(scores)
        gold_head = [int(i.split("\t")[6]) for i in test_sentences[sen_index].split('\n')]
        gold_head = [-1] + gold_head
        for i in range(np.size(head_t)):
            if i!=0 and gold_head[i] <= 0:
                gold_head[i] = 0
            if head_t[i] == gold_head[i]:
                n_correct = n_correct + 1
            n_tokens = n_tokens + 1
        sen_index = sen_index + 1
    uas = 1.0*n_correct/(1.0*n_tokens)
    print(uas)

















