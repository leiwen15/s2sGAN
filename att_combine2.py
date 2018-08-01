 # -*- coding: utf-8 -*-
# import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numpy as np
from keras import losses
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, merge, Activation, Concatenate, Add
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.layers.core import Permute, Lambda, Activation, Reshape
from keras import backend as K
# from keras.engine.topology import Layer, InputSpec
# from keras.layers.merge import Dot, Multiply, dot
from keras.activations import softmax
from att import AttLayer
import math
from gensim.models import KeyedVectors

##  data init
# word to 128 dim                       词向量的维度 = 128
EMBEDDING_DIM = 200
# sents contain at most 30 words
MAX_SENT_LENGTH = 30
MAX_SENTS = 10
# max word possibility = 70000
MAX_NB_WORDS = 65535
alpha = 0.2  # 论文中超参数a的值

################################ 数据读入部分 开始 ####################################
## read data from file
infile = open('tcs_train.txt', 'r',encoding='utf-8')
sentenceList = []
labels = []
sampleNum = 0
timeList = []
contentList = []

times=[]
tscs=[]
contentLine = infile.readlines()
num=0
for line in contentLine:
    if 'type:' in line:
        if tscs!=[] and len(tscs)>10:#先填满tscs
            label = line.split(':')[1]
            labels.append(label)
            contentList.append(tscs[0:10])
            timeList.append(times[0:10])
            sentenceList.extend(tscs[0:10])
            tscs=[]
            times=[]
            sampleNum+=1
        else:
            tscs=[]
    else:
        tscs.append(line.split('  ')[1].strip('\n'))
        times.append(int(line.split('  ')[0]))
print('num',num)
print('labels',labels)
print('contentList',contentList)
print('timeList',timeList)
print('sentenceList',sentenceList)
print('samples num',sampleNum)# 样本数量
print('labes num',len(labels))# 用户情感标签数量
print('content num',len(contentList),len(contentList[5]))
print('time input num',len(timeList),len(timeList[7]))#时间戳
print('sentence num',len(sentenceList))# 弹幕内容


################################ 时间衰减函数矩阵计算 开始 ####################################
# do time operations
# samples * 10 -> samples * 10 * 10
timeDecayList = []
# 从timeList中读取100维的时间戳，生成100*100的时间衰减矩阵
for timeVec in timeList:
	#100 * 100
    timeDecayTemp = [[0.0] * MAX_SENTS for _ in range(MAX_SENTS)] #MAX_SENTS*MAX_SENTS
    for i in range(0,MAX_SENTS):
        for j in range(0,i):
				#时间衰减的公式
            continue
            # timeDecayTemp[i][j] =  math.exp(-alpha *(timeVec[i]-timeVec[j]))
##				print (timeVec[col],timeVec[row], math.exp(-(timeVec[col]-timeVec[row])) )
				#矩阵一半是全0，因为后面对前面的无影响
	# 最终得到的维度： 样本数 * 100 * 100




################################ 时间衰减函数矩阵计算 结束 ####################################

## data tokenize (use keras Tokenizer class)        用keras自带的Tokenizer库建立词典index
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(sentenceList)
# word dic
word_index = tokenizer.word_index
print('Total %s unique words.' % len(word_index))

# data to np array, mContent shape: samples x 200sent   将python数组转成npy数组
x_data = np.zeros((len(contentList), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
y_data = np.asarray(labels, dtype='float32')
time_data = np.asarray(timeDecayList, dtype='float32')

# 根据之前建立的词典word_index，建立弹幕内容的输入数据x_data，维度： 样本数 * 100 * 最大词数（20）
for i, sentences in enumerate(contentList):
    for j, sent in enumerate(sentences):
        if j < MAX_SENTS:
            # split word
            wordTokens = text_to_word_sequence(sent)
            k = 0
            for word in wordTokens:
                if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                    x_data[i, j, k] = tokenizer.word_index[word]
                    k += 1

print('Shape of data tensor:', x_data.shape)
print('Shape of label tensor:', y_data.shape)
print('Shape of time tensor:', time_data.shape)
#
## read pre-trained word2vec     读入word2vec预训练权重
embeddings_index = KeyedVectors.load_word2vec_format(r'danmu.model.bin',binary=True)
print('shape of word2vec data', embeddings_index.vectors.shape)

# a dic: word str -> 128 dim vec    将word_index中的词文本，根据w2v结果替换成128维向量
print('Begin to build embedding matrix')
wordIn = 0
wordNotIn = 0
embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    ################################################################## annotate to be faster
    if (word in embeddings_index.vocab):
        embedding_vector = embeddings_index[word]
        embedding_matrix[i] = embedding_vector
        wordIn = wordIn + 1
    else:
        ######################### ######################################### annotate to be faster
        embedding_matrix[i] = [0] * EMBEDDING_DIM
        wordNotIn = wordNotIn + 1
##        print (word)

print("word in w2v", wordIn)
print("word not in w2v", wordNotIn)

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)




































# distributed lstm
sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32', name='sentence_input')
print(sentence_input.shape)
embedded_sequences = embedding_layer(sentence_input)
print('embedded_sequence shape',embedded_sequences.shape)
lstm = Bidirectional(LSTM(64), name='word_lstm')(embedded_sequences)
print('lstm shape',lstm.shape)
l_att = AttLayer()(lstm)
sentEncoder = Model(sentence_input, l_att)

# 10 sentence, 30 length
review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32', name='review_input')
review_encoder = TimeDistributed(sentEncoder)(review_input)
print(review_encoder.shape)
sent_lstm = LSTM(EMBEDDING_DIM, input_shape=(MAX_SENTS, EMBEDDING_DIM), return_sequences=True, name='sent_lstm')(
    review_encoder)
# sent_lstm = Reshape((MAX_SENTS,EMBEDDING_DIM))(sent_lstm)
# print(sent_lstm.shape)
# ###################################  attention
# time_input = Input(shape=(MAX_SENTS, MAX_SENTS), name='time_input')
# semantic_permute = Permute((2, 1), name='semantic_permute')(review_encoder)
# semantic_merge = merge([review_encoder, semantic_permute], mode='dot', dot_axes=(2, 1), name='semantic_merge')
# semantic_softmax = Activation('softmax')(semantic_merge)
# weight_merge = merge([time_input, semantic_softmax], mode='mul', name='weight_merge')
# softmax_weight = Activation('softmax')(weight_merge)
# print(softmax_weight.shape, sent_lstm.shape)
# merge_lstm = merge([softmax_weight, sent_lstm], mode='dot', dot_axes=(2, 1), name='merge_lstm')
# decoder_lstm = LSTM(EMBEDDING_DIM, input_shape=(MAX_SENTS, EMBEDDING_DIM), name='decoder_lstm')(merge_lstm)
# # ###################################  attention

# result = merge([combine_user, combine_video], mode = 'dot')
# output_layer = Activation('softmax')(result)
#
# model = Model(inputs=[review_input, time_input], outputs=output_layer)
# ##print model.summary()
# model.compile(loss='categorical_crossentropy',
#               optimizer='Adam',
#               metrics=['acc'])
# ######################################################################################
# # start training
# print("model fitting - Our Model")
# print(model.summary())
# model.fit([x_data, time_data], y_data, epochs=10, batch_size=32)
# end training
# #######################################################################################
# video_ebd = model.get_layer('global_video').get_weights()
# video_ebd = video_ebd[0]
# print ('video_ebd',video_ebd.shape)
#
# user_ebd = model.get_layer('global_user').get_weights()
# user_ebd = user_ebd[0]
# print ('user_ebd',user_ebd.shape,type(user_ebd))

# ###########################################
# #                  test
# ###########################################
# testdata=open('testdata_new.txt','r')
# trueValue = [[-1] * totalVideo for _ in range(totalUser)]
# userFav = [0] * totalUser
# userTot = [0] * totalUser
# userInfo = [[] for _ in range(totalUser)]
#
# while True:
# 	contentLine=testdata.readline()
# 	if (contentLine and contentLine!='\n'):
# 		print (contentLine)
# 		content = contentLine.split('\t')
# 		if (dictUser.get(content[0]) is None):
# 			print('User:', content[0], 'is missing!')
# 			continue
# 		userID = dictUser[content[0]]
# 		if dictVideo.get(content[1] is None):
# 			print('Video:', content[1], 'is missing!')
# 			continue
# 		videoID = dictVideo[content[1]]
# 		res = int(content[2])
# 		trueValue[userID][videoID] = res
# 		if (res == 1):
# 			userFav[userID] += 1
# 		userTot[userID] += 1
# 		pred = float(np.vdot(video_ebd[videoID],user_ebd[userID]))
# 		userInfo[userID].append([videoID,pred,res])
# 	else:
# 		break
#
# tt = 0
# tp = 0
# for i in range(totalUser):
# 	print(i,userTot[i],userFav[i])
# 	tt += userTot[i]
# 	tp += userFav[i]
# print (tt,tp)
#
# trueTop5 = 0
# trueTop10 = 0
# trueTop20 = 0
# valTest = 0
# for i in range(totalUser):
# 	userInfo[i] = sorted(userInfo[i], reverse=True, key=lambda user: user[1])
# 	if (userTot[i] >= 20):
# 		valTest += 1
# 		for j in range(5):
# 			if (userInfo[i][j][2] == 1):
# 				trueTop5 += 1
# 		for j in range(5,10):
# 			if (userInfo[i][j][2] == 1):
# 				trueTop10 += 1
# 		for j in range(10,min(20, userTot[i])):
# 			if (userInfo[i][j][2] == 1):
# 				trueTop20 += 1
#
# trueTop10 += trueTop5
# trueTop20 += trueTop10
#
# print('Top5: ', 1.0 * trueTop5 / (5.0 * valTest), 1.0 * trueTop5 / tp)
# print('Top10: ', 1.0 * trueTop10 / (10.0 * valTest), 1.0 * trueTop10 / tp)
# print('Top20: ', 1.0 * trueTop20 / (20.0 * valTest), 1.0 * trueTop20 / tp)
