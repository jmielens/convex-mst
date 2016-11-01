import sys, string, random, os, re
from optparse import OptionParser
import nltk
import numpy as np
from scipy.sparse import coo_matrix
from sklearn import linear_model
import gflDeps
import math

parser = OptionParser()
parser.add_option("-f","--file", dest="file", help="Input File")
parser.add_option("-v","--verbose", dest="verbose", help="Print lots of stuff")
parser.add_option("-o","--output", dest="output", help="Output File", default="out")
parser.add_option("-i","--iterations", dest="iter", help="Number of Iterations", default="20")
parser.add_option("-l","--length", dest="length", help="Number of Tokens", default="200")
parser.add_option("-g","--gflfile", dest="gflfile", help="GFL Annotation File")
(options, args) = parser.parse_args()

sentences = open(options.file).read().rstrip("\n").split("\n\n")
sentences = [i for i in sentences if len(i.split('\n')) <= int(options.length)]

iftest = (options.output == "test")

#sentences = open('en-universal-test-strip.conll').read().split("\n\n")
#sentences = [i for i in sentences if len(i.split('\n')) <= 10]

### create feature sets
tags = ['NOUN','VERB','ADJ','ADV','PRON','DET','ADP','NUM','CONJ','PRT','X', 'ROOT']
cont_tags = ['NOUN','VERB','ADJ','ADV','PRON','DET','ADP','NUM','CONJ','PRT','X', 'STA','END','ROOT']
distances = ["-10","-9","-8","-7","-6","-5","-4","-3","-2","-1","1","2","3","4","5","6","7","8","9","10"]
features = list()
f_i_dict = dict()
f_index = 0
for tag in tags:
    for d in distances:
        features.append('i' + tag + 'd' + d)
        f_index = f_index + 1
        f_i_dict[features[-1]] = f_index

for tag in tags:
    for d in distances:
        features = features + ['j' + tag + 'd' + d]
        f_index = f_index + 1
        f_i_dict[features[-1]] = f_index

for tag1 in tags:
    for tag2 in tags:
        for d in distances:
            features = features + ['i' + tag1 + 'j' + tag2 + 'd' + d]
            f_index = f_index + 1
            f_i_dict[features[-1]] = f_index

for tag1 in tags:
    for tag2 in cont_tags:
        for tag3 in tags:
            for d in distances:
                features.append('i' + tag1 + 'iM1'+ tag2 + 'j' + tag3 + 'd' + d)
                f_index = f_index + 1
                f_i_dict[features[-1]] = f_index

for tag1 in tags:
    for tag2 in cont_tags:
        for tag3 in tags:
            for d in distances:
                features.append('i' + tag1 + 'iP1'+ tag2 + 'j' + tag3 + 'd' + d)
                f_index = f_index + 1
                f_i_dict[features[-1]] = f_index

for tag1 in tags:
    for tag2 in tags:
        for tag3 in cont_tags:
            for d in distances:
                features.append('i' + tag1 + 'j'+ tag2 + 'jM1' + tag3 + 'd' + d)
                f_index = f_index + 1
                f_i_dict[features[-1]] = f_index

for tag1 in tags:
    for tag2 in tags:
        for tag3 in cont_tags:
            for d in distances:
                features.append('i' + tag1 + 'j'+ tag2 + 'jP1' + tag3 + 'd' + d)
                f_index = f_index + 1
                f_i_dict[features[-1]] = f_index

for tag in tags:
    features = features + ['i' + tag]
    f_index = f_index + 1
    f_i_dict[features[-1]] = f_index

for tag in tags:
    features = features + ['j' + tag]
    f_index = f_index + 1
    f_i_dict[features[-1]] = f_index

for tag1 in tags:
    for tag2 in tags:
        features = features + ['i' + tag1 + 'j' + tag2]
        f_index = f_index + 1
        f_i_dict[features[-1]] = f_index

for tag1 in tags:
    for tag2 in cont_tags:
        for tag3 in tags:
            features.append('i' + tag1 + 'iM1'+ tag2 + 'j' + tag3)
            f_index = f_index + 1
            f_i_dict[features[-1]] = f_index

for tag1 in tags:
    for tag2 in cont_tags:
        for tag3 in tags:
            features.append('i' + tag1 + 'iP1'+ tag2 + 'j' + tag3)
            f_index = f_index + 1
            f_i_dict[features[-1]] = f_index

for tag1 in tags:
    for tag2 in tags:
        for tag3 in cont_tags:
            features.append('i' + tag1 + 'j'+ tag2 + 'jM1' + tag3)
            f_index = f_index + 1
            f_i_dict[features[-1]] = f_index

for tag1 in tags:
    for tag2 in tags:
        for tag3 in cont_tags:
            features.append('i' + tag1 + 'j'+ tag2 + 'jP1' + tag3)
            f_index = f_index + 1
            f_i_dict[features[-1]] = f_index

        

num_features = len(features)
print(num_features)
print(features[-1])


### create y-name, y value and x matrix, universal grammar sets
#universal_sets = set(['VERB VERB', 'VERB NOUN', 'VERB PRON', 'VERB ADV', 'VERB ADP', \
#                      'NOUN NOUN', 'NOUN ADJ', 'NOUN DET', 'NOUN NUM', 'NOUN CONJ', 'ADJ ADV', 'ADP NOUN'])
#f = open('en-univiersal-dev.sentences', 'r')
#outfile = open('arcs_list_dev','w')
if(not iftest):
    universal_sets = set(['VERB VERB', 'VERB NOUN', 'VERB PRON', 'VERB ADV', 'VERB ADP', \
                      'NOUN NOUN', 'NOUN ADJ', 'NOUN DET', 'NOUN NUM', 'NOUN CONJ', 'ADJ ADV', 'ADP NOUN'])
    (whitelist, blacklist) = gflDeps.getArcLists(options.gflfile)
    anno_gfl = list()
    anno_ug = list()

y = list()

#f1 = open("row_ind_list",'w')
#f2 = open("col_ind_list",'w')
#f3 = open("data_list",'w')
arc_index = 0
row_ind_list = []
col_ind_list = []
data_list = []
sen_length = list()
sen_index = 0
for sentence in sentences:
    if(not iftest):
        sen_whitelist = set(whitelist[sen_index])
        sen_blacklist = set(blacklist[sen_index])
    lines = sentence.split("\n")
    slength = len(lines)
    sen_length = sen_length + [slength]
    head_pos = "ROOT"
    head_index = "0"
    for line in lines:
        #temp_x = np.zeros(num_features)
        tokens = line.split("\t")
        if len(tokens) < 3:
            print('flag' + line)
            continue
        tail_index = tokens[0]
        tail_pos = tokens[3]
        if tail_index == "1":
            y = y + [0.6]
        else:
            y = y + [0.0]
        odist = int(head_index) - int(tail_index)
        if odist > 0:
            dist = min(odist, 10)
        else:
            dist = max(odist, -10)
        feature1 = 'i' + head_pos + 'd' + str(dist)
        feature2 = 'j' + tail_pos + 'd' + str(dist)
        feature3 = 'i' + head_pos + 'j' + tail_pos + 'd' + str(dist)
        feature11 = 'i' + head_pos
        feature22 = 'j' + tail_pos
        feature33 = 'i' + head_pos + 'j' + tail_pos
        if int(tail_index) == 1:
            tail_m_pos = 'STA'
        else:
            tail_m_pos = lines[int(tail_index)-2].split("\t")[3];
        if int(tail_index) == slength:
            tail_p_pos = 'END'
        else:
            tail_p_pos = lines[int(tail_index)].split("\t")[3];
        feature6 = 'i' + head_pos + 'j'+ tail_pos + 'jM1' + tail_m_pos + 'd' + str(dist)
        feature7  = 'i' + head_pos + 'j'+ tail_pos + 'jP1' + tail_p_pos + 'd' + str(dist)
        feature66 = 'i' + head_pos + 'j'+ tail_pos + 'jM1' + tail_m_pos 
        feature77  = 'i' + head_pos + 'j'+ tail_pos + 'jP1' + tail_p_pos 
        #temp_x[f_i_dict[feature1] - 1] = 1
        #temp_x[f_i_dict[feature2] - 1] = 1
        #temp_x[f_i_dict[feature3] - 1] = 1
        #temp_x[f_i_dict[feature6] - 1] = 1
        #temp_x[f_i_dict[feature7] - 1] = 1
        
        row_ind_list.extend([arc_index]*10)
        col_ind_list.extend([f_i_dict[feature1] - 1, f_i_dict[feature2] - 1, f_i_dict[feature3] - 1, \
               f_i_dict[feature6] - 1, f_i_dict[feature7] - 1, f_i_dict[feature11] - 1, f_i_dict[feature22] - 1, f_i_dict[feature33] - 1, f_i_dict[feature66] - 1, f_i_dict[feature77] - 1])
        data_list.extend([1]*10)

        arc_index = arc_index + 1

        if(not iftest):
            if (head_pos + ' ' + tail_pos) in universal_sets:
                anno_ug = anno_ug + [1]
            else:
                anno_ug = anno_ug + [0]
            if (head_index + ' -> ' + tail_index) in sen_whitelist:
                anno_gfl = anno_gfl + [1]
            elif (head_index + ' -> ' + tail_index) in sen_blacklist:
                anno_gfl = anno_gfl + [-1]
            else:
                anno_gfl = anno_gfl + [0]
        #f2.write("\t".join(map(str, temp_x)) + "\n")
        #x = x + [temp_x.tolist()]

    for line1 in lines:
        tokens1 = line1.split("\t")
        if len(tokens1) < 3:
            continue
        head_index = tokens1[0]
        head_pos = tokens1[3]
        for line2 in lines:
            tokens2 = line2.split("\t")
            if tokens2[0] != head_index:
                #temp_x = np.zeros(num_features)
                if len(tokens2) < 3:
                    continue
                tail_index = tokens2[0]
                tail_pos = tokens2[3]
                odist = int(head_index) - int(tail_index)
                if odist > 0:
                    dist = min(odist, 10)
                else:
                    dist = max(odist, -10)
                if (int(head_index) - int(tail_index)) == -1:
                    y = y + [0.6]
                else:
                    y = y + [0.0]

                if int(head_index) == 1:
                    head_m_pos = 'STA'
                else:
                    head_m_pos = lines[int(head_index)-2].split("\t")[3];
                if int(head_index) == slength:
                    head_p_pos = 'END'
                else:
                    head_p_pos = lines[int(head_index)].split("\t")[3];

                if int(tail_index) == 1:
                    tail_m_pos = 'STA'
                else:
                    tail_m_pos = lines[int(tail_index)-2].split("\t")[3];
                if int(tail_index) == slength:
                    tail_p_pos = 'END'
                else:
                    tail_p_pos = lines[int(tail_index)].split("\t")[3];
                feature1 = 'i' + head_pos + 'd' + str(dist)
                feature2 = 'j' + tail_pos + 'd' + str(dist)
                feature3 = 'i' + head_pos + 'j' + tail_pos + 'd' + str(dist)
                feature4 = 'i' + head_pos + 'iM1'+ head_m_pos + 'j' + tail_pos + 'd' + str(dist)
                feature5 = 'i' + head_pos + 'iP1'+ head_p_pos + 'j' + tail_pos + 'd' + str(dist)
                feature6 = 'i' + head_pos + 'j'+ tail_pos + 'jM1' + tail_m_pos + 'd' + str(dist)
                feature7  = 'i' + head_pos + 'j'+ tail_pos + 'jP1' + tail_p_pos + 'd' + str(dist)
                feature11 = 'i' + head_pos 
                feature22 = 'j' + tail_pos 
                feature33 = 'i' + head_pos + 'j' + tail_pos 
                feature44 = 'i' + head_pos + 'iM1'+ head_m_pos + 'j' + tail_pos 
                feature55 = 'i' + head_pos + 'iP1'+ head_p_pos + 'j' + tail_pos 
                feature66 = 'i' + head_pos + 'j'+ tail_pos + 'jM1' + tail_m_pos 
                feature77  = 'i' + head_pos + 'j'+ tail_pos + 'jP1' + tail_p_pos 
                #temp_x[f_i_dict[feature1] - 1] = 1
                #temp_x[f_i_dict[feature2] - 1] = 1
                #temp_x[f_i_dict[feature3] - 1] = 1
                #temp_x[f_i_dict[feature4] - 1] = 1
                #temp_x[f_i_dict[feature5] - 1] = 1
                #temp_x[f_i_dict[feature6] - 1] = 1
                #temp_x[f_i_dict[feature7] - 1] = 1

                row_ind_list.extend([arc_index]*14)
                col_ind_list.extend([f_i_dict[feature1] - 1, f_i_dict[feature2] - 1, f_i_dict[feature3] - 1, \
                               f_i_dict[feature4] - 1, f_i_dict[feature5] - 1, f_i_dict[feature6] - 1, f_i_dict[feature7] - 1, \
                                f_i_dict[feature11] - 1, f_i_dict[feature22] - 1, f_i_dict[feature33] - 1, \
                               f_i_dict[feature44] - 1, f_i_dict[feature55] - 1, f_i_dict[feature66] - 1, f_i_dict[feature77] - 1])
                data_list.extend([1]*14)

                arc_index = arc_index + 1
                if(not iftest):
                    if (head_pos + ' ' + tail_pos) in universal_sets:
                        anno_ug = anno_ug + [1]
                    else:
                        anno_ug = anno_ug + [0]
                    if (head_index + ' -> ' + tail_index) in sen_whitelist:
                        anno_gfl = anno_gfl + [1]
                    elif (head_index + ' -> ' + tail_index) in sen_blacklist:
                        anno_gfl = anno_gfl + [-1]
                    else:
                        anno_gfl = anno_gfl + [0]
                #f2.write("\t".join(map(str, temp_x)) + "\n")
                #x = x + [temp_x.tolist()]
    sen_index = sen_index + 1
#f1.write("\n".join(map(str, y)))
#f3.write("\n".join(map(str, anno)))
print(len(row_ind_list))
print(len(col_ind_list))
print(len(data_list))
print(len(y))
#print(len(anno))
total_num_arcs = len(y)
row_ind_list = np.array(row_ind_list)
np.save(options.output + "_row_ind_array_np", row_ind_list)
col_ind_list = np.array(col_ind_list)
np.save(options.output + "_col_ind_array_np", col_ind_list)
data_list = np.array(data_list)
np.save(options.output + "_data_array_np", data_list)
y = np.array(y)
np.save(options.output + "_y_init_array_np", y)
if(not iftest):
    anno_gfl = np.array(anno_gfl)
    np.save(options.output + "_anno_gfl_np", anno_gfl)
    anno_ug = np.array(anno_ug)
    np.save(options.output + "_anno_ug_np", anno_ug)











































