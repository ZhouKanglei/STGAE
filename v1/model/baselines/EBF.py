
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Input, BatchNormalization

import tensorflow as tf
import numpy as np

class ebf(Model):
    """A Deep Recurrent Framework for Cleaning Motion Capture Data"""
    def __init__(self):
        super(ebf, self).__init__(dynamic=True)

        self.e = Sequential([
            BatchNormalization(),
            Dense(units=108, activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),
            Dense(units=110, activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),
            Dense(units=86, activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),
            Dense(units=64, activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),
        ], name='E')

        self.b = Sequential([
            Bidirectional(LSTM(units=108, return_sequences=True)),
            Bidirectional(LSTM(units=108))
        ], name='B')

        self.f = Sequential([
            BatchNormalization(),
            Dense(units=108, activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),
            Dense(units=108, activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),
            Dense(units=108, activation='relu', kernel_initializer='he_uniform'),
            BatchNormalization(),
            Dense(units=108, kernel_initializer='he_uniform')
        ], name='F')

        self.bias = None

    def build(self, input_shape):
        N, T, C = input_shape
        self.bias = self.add_weight(name='bias',
                                    shape=(T, 1),
                                    initializer=tf.keras.initializers.zeros,
                                    trainable=True)
        self.build_graph(input_shape[1:])

    def build_graph(self, input_shape):
        inputs = Input(shape=input_shape)
        return Model(inputs=inputs, outputs=self.call(inputs))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        N, T, C = x.shape

        h = self.e(x)
        h = self.b(h)
        sigma = self.f(h)
        sigma = tf.exp(sigma)
        # print(sigma)

        mol = np.zeros(shape=(T, C))
        for i in range(T):
            for j in range(C):
                mol[i, j] = np.square(i - 15) / 255

        # print(mol)

        mol = tf.convert_to_tensor(mol, dtype=tf.float64)
        omega = - tf.einsum('tc, nc->ntc', mol, 1 / tf.square(sigma) / 2)
        # print(omega)
        # print(omega[0][15])
        omega = tf.nn.softmax(omega, axis=1)

        # print(omega)

        bias = tf.einsum('ntc, td->ncd', x, self.bias)
        bias = tf.reshape(bias, [-1, C])

        y = tf.einsum('ntc, ntc->nc', x, omega) - bias * 0

        return y
