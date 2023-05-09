from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, \
    Input, Conv1D, MaxPool1D, UpSampling1D, Dropout, Lambda, AveragePooling1D

import tensorflow as tf

class cnn(Model):
    """Learning Motion Manifolds with Convolutional Auto-encoders"""

    def __init__(self,
                 hidden_filters=[108, 128, 128, 128, 256],
                 dropout=0):
        super(cnn, self).__init__()

        self.hidden_filters = hidden_filters

        self.batch_in = BatchNormalization(name='batch_in')

        self.encoder = []
        self.en_res = []
        for i in range(len(self.hidden_filters)):
            self.encoder.append(Sequential([
                Conv1D(filters=self.hidden_filters[i],
                       kernel_size=15, use_bias=True,
                       padding='same'),
                BatchNormalization(),
                Activation('relu'),
                MaxPool1D(pool_size=2) if i % 2 == 0 else Lambda(lambda x: x),
                Dropout(dropout),
            ], name='Encoder_%s' % (i + 1)))

            self.en_res.append(Sequential([
                Conv1D(filters=self.hidden_filters[i],
                       kernel_size=1,
                       padding='same'),
                MaxPool1D(pool_size=2) if i % 2 == 0 else Lambda(lambda x: x),
            ], name='in_res_%d' % (i + 1)))

        self.decoder = []
        self.de_res = []
        for i in range(len(self.hidden_filters)):
            self.decoder.append(Sequential([
                Conv1D(filters=self.hidden_filters[len(self.hidden_filters) - i - 1],
                       kernel_size=15, use_bias=True,
                       padding='same'),
                BatchNormalization(),
                Activation('relu'),
                UpSampling1D(size=2) if i % 2 == 0 else Lambda(lambda x: x),
                Dropout(dropout),
            ], name='Decoder_%s' % (len(self.hidden_filters) - i)))

            self.de_res.append(Sequential([
                Conv1D(filters=self.hidden_filters[len(self.hidden_filters) - i - 1],
                       kernel_size=1,
                       padding='same'),
                UpSampling1D(size=2) if i % 2 == 0 else Lambda(lambda x: x),
            ], name='out_res_%d' % (len(self.hidden_filters) - i)))

    def build(self, input_shape):
        self.build_graph(input_shape[1:])

    def build_graph(self, input_shape):
        inputs = Input(shape=input_shape)
        outputs = tf.keras.backend.ones_like(inputs)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def call(self, inputs, training=None, mask=None):
        x = inputs

        h = self.batch_in(x)
        for i in range(len(self.hidden_filters)):
            h = self.en_res[i](h) + self.encoder[i](h)

        for i in range(len(self.hidden_filters)):
            h = self.de_res[i](h) + self.decoder[i](h)

        y =  x - h

        return y