# from att import MultiHeadAttention
from keras_multi_head import MultiHeadAttention
import tensorflow as tf

from keras.layers import Input, Dense, Lambda, concatenate, LSTM, Activation, Flatten, MaxPooling2D, GlobalAveragePooling1D
from keras.layers.convolutional import Conv2D, Conv1D
from keras.models import Model
from keras import backend as K
from keras.layers.core import RepeatVector, Dropout, Layer
from keras.layers.wrappers import TimeDistributed
from keras import optimizers
from keras.losses import mse
from keras.models import Sequential
from tensorflow.keras import layers

import numpy as np

class LayerNormalization(Layer):
    """
    Implementation of Layer Normalization (https://arxiv.org/abs/1607.06450).
    "Unlike batch normalization, layer normalization performs exactly
    the same computation at training and test times."
    """
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['axis'] = self.axis
        return config

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        dim = input_shape[-1]
        self.gain = self.add_weight(
            name='gain',
            shape=(dim,),
            initializer='ones',
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(dim,),
            initializer='zeros',
            trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=self.axis, keepdims=True)
        variance = K.mean(
            K.square(inputs - mean), axis=self.axis, keepdims=True)
        epsilon = K.constant(1e-5, dtype=K.floatx())
        normalized_inputs = (inputs - mean) / K.sqrt(variance + epsilon)
        result = self.gain * normalized_inputs + self.bias
        return result


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        # self.att = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim), ]
        )

        self.layernorm1 = LayerNormalization()
        self.layernorm2 = LayerNormalization()
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

        self.input_query = Dense(ff_dim, activation='relu', name='input_query')
        self.input_key = Dense(ff_dim, activation='relu', name='input_key')
        self.input_value = Dense(ff_dim, activation='relu', name='input_value')
        self.att = MultiHeadAttention(head_num=num_heads, name='att_layer')
        self.reshape = Dense(ff_dim,activation='relu', name='input_original')
    def call(self, inputs, training):

        q = inputs#self.input_query(inputs)
        k = inputs#self.input_key(inputs)
        v = inputs#self.input_value(inputs)

        attn_output = self.att([v,k,q])
        print(attn_output.shape,'after multihead attention--------------------------------------')
        attn_output = self.dropout1(attn_output, training=training)

        print(inputs.shape,'input reshape shape-------------------------------------------------')
        # test = self.reshape(inputs)
        out1 = self.layernorm1(inputs + attn_output)
        print(out1.shape,'shape after layer normalization --------------------------------------')
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionEncoding(Layer):

    def __init__(self, model_dim, **kwargs):
        self._model_dim = model_dim
        super(PositionEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        seq_length = inputs.shape[1]

        position_encodings = np.zeros((seq_length, self._model_dim))
        for pos in range(seq_length):
            for i in range(self._model_dim):
                position_encodings[pos, i] = pos / np.power(10000, (i-i%2) / self._model_dim)
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2]) # 2i
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2]) # 2i+1
        position_encodings = K.cast(position_encodings, 'float32')
        return position_encodings

    def compute_output_shape(self, input_shape):
        return input_shape


class Add(Layer):

    def __init__(self, **kwargs):
        super(Add, self).__init__(**kwargs)

    def call(self, inputs):
        input_a, input_b = inputs
        return input_a + input_b

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Encoder(Layer):
    def __init__(self,embed_dim,num_heads,ff_dim,num_layers):
        self.num_layers = num_layers
        self.transformer_block_list = [TransformerBlock(embed_dim,num_heads, ff_dim)  for _ in range(self.num_layers)]
        self.reshape = Dense(embed_dim,activation='relu')

        self.pos_encoding = PositionEncoding(embed_dim)
        #
    def __call__(self,x):

        x = self.reshape(x)
        x_pos_enc = self.pos_encoding(x)
        x = Add()([x,x_pos_enc])
        x = Dropout(0.1)(x)

        for enc in self.transformer_block_list:
            x = enc(x,training=False)
        return x

