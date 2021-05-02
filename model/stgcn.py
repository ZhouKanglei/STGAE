# The based unit of spatial temporal module.
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Activation, Lambda, Input, \
    BatchNormalization, Dropout, Conv2D, UpSampling2D, AveragePooling2D

from model.agcn import agcn
from model.gcn import gcn
from model.tgcn import tgcn

from config.config import ATTENTION

class stgcn(Model):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        filters (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolution kernel
                                & graph convolution kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism.
                                Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format

        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`.

    """
    def __init__(self,
                 filters,
                 kernel_size,
                 stride=1,
                 A=None,
                 dropout=0,
                 layer_no=1,
                 in_batchnorm=True,
                 out_batchnorm=True,
                 residual=True):
        super(stgcn, self).__init__(dynamic=True)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        t_padding = (kernel_size[0] - 1) // 2

        self.residual = residual
        self.filters = filters
        self.kernel_size = kernel_size
        self.res = None
        self.stride = stride
        self.A = A
        self.layer_no = layer_no

        # C-B-A: gcn sub-block
        if ATTENTION == 'A+B+C' or ATTENTION == 'A+C'  or ATTENTION == 'B+C':
            self.gcn = agcn(filters=filters,
                           t_kernel_size=kernel_size[1],
                           layer_no=layer_no,
                           A=self.A)
        elif ATTENTION == 'A+B' or ATTENTION == 'A':
            self.gcn = gcn(filters=filters,
                            t_kernel_size=kernel_size[1],
                            layer_no=layer_no,
                            A=self.A)
        elif ATTENTION == 'A*M':
            self.gcn = tgcn(filters=filters,
                           t_kernel_size=kernel_size[1],
                           layer_no=layer_no,
                           A=self.A)

        if in_batchnorm:
            self.batch_1 = BatchNormalization()
        else:
            self.batch_1 = Lambda(lambda x: x, name='not_in_batchnorm')
        self.a_1 = Activation('relu')
        self.dropout_1 = Dropout(dropout)

        # C-B-A-P-D: tcn sub-block
        self.tcn = Conv2D(filters=filters,
                          kernel_size=(kernel_size[0], 1),
                          padding='same' if t_padding else 'valid',
                          strides=(1, 1),
                          name='tcn')
        if out_batchnorm:
            self.batch_2 = BatchNormalization()
        else:
            self.batch_2 = Lambda(lambda x: x, name='not_out_batchnorm')
        self.a_2 = Activation('relu')
        if stride == 1:
            self.up = Lambda(lambda x: x, name='not_UpSampling2D')
        elif stride == 2:
            self.up = AveragePooling2D(pool_size=(stride, 1), name='AveragePooling2D')
        elif stride == 0.5:
            self.up = UpSampling2D(size=(np.int(1 / stride), 1), name='UpSampling2D')
        self.dropout_2 = Dropout(dropout)

    def build(self, input_shape):
        input_channel = input_shape[-1]

        if not self.residual:
            self.res = Lambda(lambda x: 0, name='residual_0')
        elif input_channel == self.filters and self.stride == 1:
            self.res = Lambda(lambda x: x, name='residual_x')
        elif self.stride >= 1:
            self.res = Sequential([
                Conv2D(filters=self.filters,
                              kernel_size=(1, 1),
                              strides=(self.stride, 1),
                              name='residual'),
            ])
        else:
            self.res = Sequential([
                Conv2D(filters=self.filters,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       name='residual_UpSampling2D'),
                UpSampling2D(size=(np.int(1 / self.stride), 1), name='res_UpSampling2D'),
            ])


        # build graph
        input_shape = input_shape[1:]
        self.build_graph(input_shape)

        tf.keras.utils.plot_model(self.build_graph(input_shape=input_shape), dpi=300,
                                  to_file='./output/plots/model/%d-stgcn-block-%d.png' % (self.layer_no, self.filters),
                                  show_shapes=True)

    def call(self, inputs, training=None, mask=None):
        x = inputs

        res = self.res(x)

        g = self.gcn(x)
        x = self.batch_1(g)
        x = self.a_1(x)
        x = self.dropout_1(x)
        x = self.tcn(x)
        x = self.batch_2(x)
        x = self.a_2(x)
        x = self.up(x)
        x = self.dropout_2(x)

        y = x + res

        return y

    def build_graph(self, input_shape):
        inputs = Input(shape=input_shape)
        return Model(inputs=inputs, outputs=self.call(inputs))

    def compute_output_shape(self, input_shape):
        x_shape = input_shape
        return tf.TensorShape((x_shape[0], np.int(x_shape[1] / self.stride), x_shape[2], self.filters))