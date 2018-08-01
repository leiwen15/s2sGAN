import numpy
import pickle
import os
import keras
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import word2vec
from keras.utils.np_utils import to_categorical
import gensim
from keras.models import Model
from keras import metrics
from keras import losses
from keras import backend as K
from keras import regularizers
from keras.backend.tensorflow_backend import set_session
from keras.layers import Embedding, LSTM, Dense, Activation, TimeDistributed,Bidirectional, Input,Reshape,Conv1D,Lambda
from keras.callbacks import TensorBoard
from keras import initializers
from keras.engine.topology import Layer

class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        print(len(input_shape))
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], 10)))
        self.b = K.variable(self.init((input_shape[-2], 10)))
        self.u = K.variable(self.init((input_shape[-2], 10)))
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W) # 对应公式(5)
        #uit = K.squeeze(uit, -1) # 对应公式(5)
        uit = uit + self.b # 对应公式(5)
        uit = K.tanh(uit) # 对应公式(5)
        ait = K.dot(uit , self.u) # 对应公式(6)
        ait = K.exp(ait) # 对应公式(6)
        # 对应公式(6)
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait) # 对应公式(7)
        weighted_input = x * ait # 对应公式(7)
        output = K.sum(weighted_input, axis=1) # 对应公式(7)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])