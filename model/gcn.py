# The based unit of graph convolution networks.
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Reshape, Input

from config.config import ATTENTION

class gcn(Model):
    r"""The basic module for applying a graph convolution.

    Args:
        filters (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolution kernel
        t_kernel_size (int): Size of the temporal convolution kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, T_{in}, V, in_channels)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, T_{out}, V, out_channels)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`.
    """

    def __init__(self,
                 filters,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True,
                 layer_no=1,
                 A=None):
        super(gcn, self).__init__(dynamic=True)

        self.layer_no = layer_no

        self.filters = filters
        self.t_kernel_size = t_kernel_size
        self.t_padding = t_padding
        self.t_stride = t_stride
        self.t_dilation = t_dilation
        self.bias = bias
        self.k_size = None
        self.A = A

        self.conv = None
        self.reshape = None

    def build(self, input_shape):
        # conv
        self.k_size = self.A.shape[0]
        self.conv = Conv2D(
            filters=self.filters * self.k_size,
            kernel_size=(self.t_kernel_size, 1),
            padding='same' if self.t_padding else 'valid',
            strides=(self.t_stride, 1),
            dilation_rate=(self.t_dilation, 1),
            use_bias=self.bias,
            input_shape=input_shape)

        # reshape
        _, t, v, c = self.conv.compute_output_shape(input_shape)
        self.reshape = Reshape([t, v, self.k_size, c // self.k_size])

        # build graph
        input_shape = input_shape[1:]
        self.build_graph(input_shape)

        tf.keras.utils.plot_model(self.build_graph(input_shape=input_shape), dpi=300,
                                  to_file='./output/plots/model/%d-gcn-%d.png'  % (self.layer_no, self.filters),
                                  show_shapes=True)

    def call(self, inputs, training=None, mask=None):
        x = inputs

        h = self.conv(x)
        h = self.reshape(h)
        y = tf.einsum('ntvkc, kvw->ntwc', h, self.A)

        return y

    def build_graph(self, input_shape):
        inputs = Input(shape=input_shape)
        return Model(inputs=inputs, outputs=self.call(inputs))

    def compute_output_shape(self, input_shape):
        x_shape = input_shape
        return tf.TensorShape((x_shape[0], x_shape[1], x_shape[2], self.filters))

