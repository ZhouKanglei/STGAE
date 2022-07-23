import tensorflow as tf


from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

class attention(Model):
    def __init__(self):
        super(attention, self).__init__(dynamic=True)
        self.output_dim = None
        self.kernel = None

    def build(self, input_shape):
        self.output_dim = input_shape[1:]
        self.kernel = self.add_weight(name='adjecent edge',
                                      shape=self.output_dim,
                                      initializer='uniform',
                                      trainable=True)

        # build graph
        # print('attention: ', input_shape)
        input_shape = input_shape[1:]
        self.build_graph(input_shape=input_shape)

        tf.keras.utils.plot_model(self.build_graph(input_shape=input_shape), dpi=300,
                                  to_file='./output/plots/model/attention.png',
                                  show_shapes=True)

    def call(self, inputs, **kwargs):
        x = inputs
        y = x * self.kernel

        return y

    def build_graph(self, input_shape):
        inputs = Input(shape=input_shape)

        return Model(inputs=[inputs], outputs=self.call(inputs))

    def compute_output_shape(self, input_shape):
        return input_shape