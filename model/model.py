# The whole end-to-end model.
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, BatchNormalization

from config.graph import *
from config.config import STRATEGY, ATTENTION
from model.stgcn import stgcn

class STGAE(Model):
    # End-to-end model.
    def __init__(self,
                 filters,
                 kernel_size=(9, 1),
                 edge_importance=ATTENTION):
        super(STGAE, self).__init__(dynamic=True)

        self.filters = filters

        # load graph
        graph = Graph(layout='nyu', strategy=STRATEGY)

        # Attention: initialize as A * 1
        if edge_importance == 'A+M' or edge_importance == 'A+B+C' or edge_importance == 'A*M':
            self.A = self.add_weight(name='Learnable_adjacent_edge',
                                     shape=graph.A.shape,
                                     initializer=tf.keras.initializers.constant(value=graph.A),
                                     trainable=True)

        else:
            self.A = tf.convert_to_tensor(graph.A)

        # build networks
        self.spatial_kernel_size = kernel_size[0]
        self.temporal_kernel_size = kernel_size[1]

        # BatchNormalization
        self.b = BatchNormalization()

        # encoder blocks
        self.g1 = stgcn(filters=self.filters, kernel_size=kernel_size, stride=1, A=self.A, layer_no=1)
        self.g2 = stgcn(filters=32, kernel_size=kernel_size, stride=2, A=self.A, layer_no=2)
        self.g3 = stgcn(filters=32, kernel_size=kernel_size, stride=1, A=self.A, layer_no=3)
        self.g4 = stgcn(filters=32, kernel_size=kernel_size, stride=2, A=self.A, layer_no=4)
        self.g5 = stgcn(filters=64, kernel_size=kernel_size, stride=1, A=self.A, layer_no=5)

        # decoder
        self.g6 = stgcn(filters=64, kernel_size=kernel_size, stride=1, A=self.A, layer_no=6)
        self.g7 = stgcn(filters=32, kernel_size=kernel_size, stride=0.5, A=self.A, layer_no=7)
        self.g8 = stgcn(filters=32, kernel_size=kernel_size, stride=1, A=self.A, layer_no=8)
        self.g9 = stgcn(filters=32, kernel_size=kernel_size, stride=0.5, A=self.A, layer_no=9)
        self.g10 = stgcn(filters=self.filters, kernel_size=kernel_size, stride=1, A=self.A, layer_no=10)


    def build(self, input_shape):
        # build graph
        input_shape = input_shape[1:]
        self.build_graph(input_shape=input_shape)

        # save model plot
        tf.keras.utils.plot_model(self.build_graph(input_shape=input_shape), dpi=300,
                                  to_file='./output/plots/model/stgcn-%d.pdf' % (self.filters),
                                  show_shapes=True)

    def call(self, inputs, training=None, mask=None):
        x_in = inputs

        x = self.b(x_in)

        x = self.g1(x)
        x = self.g2(x)
        x = self.g3(x)
        x = self.g4(x)
        x = self.g5(x)
        x = self.g6(x)
        x = self.g7(x)
        x = self.g8(x)
        x = self.g9(x)
        x_out = self.g10(x)

        y = x_in - x_out

        return y

    def build_graph(self, input_shape):
        inputs = Input(shape=input_shape)
        return Model(inputs=inputs, outputs=self.call(inputs))

    def compute_output_shape(self, input_shape):
        x_shape = input_shape
        return tf.TensorShape((x_shape[0], x_shape[1], x_shape[2], self.filters))