from keras.preprocessing.text import Tokenizer,text_to_word_sequence
import numpy as np
from gensim.models import KeyedVectors
from keras.layers import Embedding
from keras.layers import Dense,Input
from keras.layers import LSTM
from keras.layers.core import Activation
from keras.models import Model

##参数
EMBEDDING_DIM=200#单词64维

MAX_SENT_LENGTH=30#每个句子最多30个单词
MAX_SENTS=60#每分钟最多60个句子

MAX_NB_WORDS=70000

tcs=[]#弹幕列表
tcs_num=0#总弹幕数
min_num=0#总分钟数
min_tcss=[]#[[]]里面每一个小的[]包含同一分钟内的总弹幕 [['cc cc sc ','dd d db dd dddds  ds' ],[],[]]
################读入文本##############
print('正在读取数据......')
fr=open('total_tcs_test.txt','r',encoding='utf-8')#先用这个文本，比较小，方便，最后要改成total_tcs3.txt
all_lines=fr.readlines()
current_min_tcs=[]
last_min = 1  # 最开始的分钟数
for line in all_lines[0:9000]:#先拿0到9000试试，最后记得删掉[0:9000]
    current_min=line.split(':')[0]#标记分钟
    t=line.split(':')[1].strip('\n')
    tcs.append(t)#将弹幕加入弹幕列表
    if int(current_min) ==last_min:#将某一分钟内所有的弹幕与该分钟结合起来
        current_min_tcs.append(t)
    else:
        min_tcss.append(current_min_tcs)
        current_min_tcs=[]
        current_min_tcs.append(t)
        last_min+=1

print('总的分钟数：',len(min_tcss))

tokenizer=Tokenizer(num_words=MAX_NB_WORDS)#用keras自带的库建立词典index
tokenizer.fit_on_texts(tcs)
word_index=tokenizer.word_index#创建word dic
print('总的词汇数：',len(word_index))

#把数据转成array
x_data=np.zeros((len(min_tcss),MAX_SENTS,MAX_SENT_LENGTH),dtype='int32')
print('Shape of data tensor:', x_data.shape)
#根据之前建立的词典word_index,建立弹幕内容的输入数据x_data，维度数：样本数*100*最大词数
for i,sents in enumerate(min_tcss):
    for j,sent in enumerate(sents):
        if j <MAX_SENTS:
            word_tokens=text_to_word_sequence(sent)
            k=0
            for word in word_tokens:
                if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
                    x_data[i,j,k]=tokenizer.word_index[word]
                    k+=1

print(x_data)
print('shape of data tensor:',x_data.shape)

#读入word2vec预训练权重,虽然上面写的是embedding=128，其实是embedding=64
embeddings_index=KeyedVectors.load_word2vec_format(r'danmu.model.bin',binary=True)


print('begin to build embedding matrix')
word_in=0
word_not_in=0
embedding_matrix=np.random.random((len(word_index)+1,EMBEDDING_DIM))
for word,i in word_index.items():
    if(word in embeddings_index.vocab):
        embedding_vector=embeddings_index[word]
        embedding_matrix[i]=embedding_vector
        word_in+=1
    else:
        embedding_matrix[i]=[0]*EMBEDDING_DIM
        word_not_in+=1

print('word in w2v',word_in)
print('word not in w2v',word_not_in)

###################autoencoder#####################
embedding_layer=Embedding(len(word_index)+1,EMBEDDING_DIM,
                          weights=[embedding_matrix],
                          input_length=MAX_SENT_LENGTH,
                          trainable=True)

sentence_input=Input(shape=(MAX_SENT_LENGTH,),dtype='int32',name='sentence_input')
embedded_sequences=embedding_layer(sentence_input)
sent_lstm=LSTM(EMBEDDING_DIM,input_shape=(MAX_SENTS,EMBEDDING_DIM))(embedded_sequences)
generator=Dense(EMBEDDING_DIM)(sent_lstm)

abs_decoder=LSTM(EMBEDDING_DIM)(generator)
abs_softmax=Activation('softmax')(abs_decoder)
model=Model(sentence_input,abs_softmax)
model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['acc'])
print(model.summary())
model.fit(x_data,x_data,epochs=10,batch_size=32)

################################start training#####################


