import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, Activation, Dropout, Input, Add, BatchNormalization, Conv2D

from config.config import COMPARAION, STRATEGY
from model.gcn import gcn
from config.graph import *

class wavenet(Model):
    """WAVENET: A GENERATIVE MODEL FOR RAW AUDIO"""
    
    def __init__(self,
                 n_filters=32,
                 filter_width=2,
                 n_layers=7,
                 dropout=0.2):
        super(wavenet, self).__init__(dynamic=True)

        self.n_layers = n_layers
        self.in_filters = None
        self.dropout = dropout

        dilation_rates = [2 ** i for i in range(self.n_layers)] * 2

        self.conv_in = []
        self.conv_f = []
        self.conv_g = []
        self.conv_out = []
        for i in range(self.n_layers):
            self.conv_in.append(Conv1D(
                filters=128,
                kernel_size=1,
                padding='same',
                activation='relu',
                name='Conv1D_in_%d' % i
            ))

            self.conv_f.append(Conv1D(
                filters=n_filters,
                kernel_size=filter_width,
                padding='causal',
                dilation_rate=dilation_rates[i],
                activation='tanh',
                name='Conv1D_f_%d' % i
            ))

            self.conv_g.append(Conv1D(
                filters=n_filters,
                kernel_size=filter_width,
                padding='causal',
                dilation_rate=dilation_rates[i],
                activation='sigmoid',
                name='Conv1D_g_%d' % i
            ))

            self.conv_out.append(Conv1D(
                filters=128,
                kernel_size=1,
                padding='same',
                activation='relu',
                name='Conv1D_out_%d' % i
            ))

        self.add = Add()
        self.out = None

    def build(self, input_shape):
        self.in_filters = input_shape[-1]

        self.out = Sequential([
            Activation('relu'),
            Conv1D(filters=128, kernel_size=1, padding='same', activation='relu'),
            Dropout(self.dropout),
            Conv1D(filters=self.in_filters, kernel_size=1, padding='same')
        ], name='Out_layer')

        self.build_graph(input_shape[1:])

        # plot and save model
        tf.keras.utils.plot_model(self.build_graph(input_shape=input_shape[1:]),
                                  to_file='./output/plots/model_WaveNet.pdf',
                                  show_shapes=True)


    def build_graph(self, input_shape):
        inputs = Input(shape=input_shape)

        return Model(inputs=inputs, outputs=self.call(inputs))

    def compute_output_shape(self, input_shape):
        x_shape = input_shape
        return tf.TensorShape((x_shape))

    def call(self, inputs, training=None, mask=None):
        x = inputs
        h = x
        skips = []

        for i in range(self.n_layers):
            z_in = self.conv_in[i](h)

            z_f = self.conv_f[i](z_in)
            z_g = self.conv_g[i](z_in)

            z = tf.multiply(z_f, z_g) # combine filter and gating branches

            z_out = self.conv_out[i](z)

            h = z_in + z_out # residual connection

            skips.append(z_out)


        skips = self.add(skips)
        out = self.out(skips) # final time-distributed dense layers
        # y = out[:, -1:, :] # extract training target at end
        # print('WaveNet: ', y.shape)
        return out


class CAM_GCN(Model):
    """Stable Hand Pose Estimation under Tremor via Graph Neural Network"""

    def __init__(self):
        super(CAM_GCN, self).__init__(dynamic=True)

        # load graph
        graph = Graph(layout='nyu', strategy=STRATEGY)

        # Attention: initialize as A * 1
        self.A = self.add_weight(name='Learnable_adjacent_edge',
                                 shape=graph.A.shape,
                                 initializer=tf.keras.initializers.constant(value=graph.A),
                                 trainable=True)

        # WaveNet
        self.wavenet = wavenet(n_filters=256,
                               filter_width=2,
                               n_layers=7,
                               dropout=0.2)

        self.g1 = gcn(filters=32,
                       t_kernel_size=1,
                       layer_no=0,
                       A=self.A)

        self.g2 = gcn(filters=64,
                      t_kernel_size=1,
                      layer_no=0,
                      A=self.A)

        self.g3 = None

        self.batch_in = BatchNormalization()
        self.batch_out = BatchNormalization()

    def build(self, input_shape):
        self.g3 = gcn(filters=input_shape[-1],
                      t_kernel_size=1,
                      layer_no=0,
                      A=self.A)

        self.conv = Conv2D(filters=input_shape[-1],
                           kernel_size=(1, 1))

        self.build_graph(input_shape=input_shape[1:])

        # plot and save model
        tf.keras.utils.plot_model(self.build_graph(input_shape=input_shape[1:]),
                                  to_file='./output/plots/model_%s.pdf' % COMPARAION,
                                  show_shapes=True)

    def build_graph(self, input_shape):
        inputs = Input(shape=input_shape)
        outputs = tf.keras.backend.ones_like(inputs)
        return Model(inputs=inputs, outputs=outputs)


    def call(self, inputs, training=None, mask=None):
        x = inputs

        N, T, V, C = x.shape
        x_in = tf.reshape(x, shape=[-1, T, V * C])

        x_in = self.batch_in(x_in)
        h = x_in + self.wavenet(x_in)

        x_out = tf.reshape(h, shape=[-1, T, V, C])

        x_out = self.batch_out(x_out)
        g1 = self.g1(x_out)
        g2 = self.g2(g1)
        g3 = self.conv(x_out) + self.g3(g2)
        # print('GCN: ', g3.shape)

        y = x - g3

        return y

    def compute_output_shape(self, input_shape):
        x_shape = input_shape
        return tf.TensorShape((x_shape))