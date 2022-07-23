
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Input, Dropout

import tensorflow as tf
import numpy as np

class ebf(Model):
    """A Deep Recurrent Framework for Cleaning Motion Capture Data"""
    def __init__(self):
        super(ebf, self).__init__(dynamic=True)

        self.ebf_1 = Sequential([
            Dense(units=108, activation='relu'),
            Dense(units=110, activation='relu'),
            Dense(units=86, activation='relu'),
            Dense(units=64, activation='relu'),

            Bidirectional(LSTM(units=108, return_sequences=True)),
            Bidirectional(LSTM(units=108)),

            Dense(units=108, activation='relu'),
            Dense(units=108, activation='relu'),
            Dense(units=108, activation='relu'),
            Dense(units=108)
        ], name='EBF_1')


    def build(self, input_shape):
        self.build_graph(input_shape[1:])

    def build_graph(self, input_shape):
        inputs = Input(shape=input_shape)
        return Model(inputs=inputs, outputs=self.call(inputs))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        N, T, C = x.shape

        sigma = self.ebf_1(x)
        sigma = tf.exp(sigma)

        mol = np.zeros(shape=(T, C))
        for i in range(T):
            for j in range(C):
                mol[i, j] = np.square(i - 15)

        mol = tf.convert_to_tensor(mol, dtype=tf.float64)
        omega = - tf.einsum('tc, nc->ntc', mol, 1 / tf.square(sigma) / 2)

        omega = tf.nn.softmax(omega, axis=1)

        y = tf.einsum('ntc, ntc->nc', x, omega)

        return y