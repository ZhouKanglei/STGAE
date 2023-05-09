from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Input

class ResNet(Model):
    """Robust Solving of Optical Motion Capture Data by Denoising"""

    def __init__(self,
                 hidden_layer_num=7,
                 hidden_uints=512):
        super(ResNet, self).__init__()

        self.layers_num = hidden_layer_num

        self.dense_in = Sequential([
            BatchNormalization(),
            Dense(units=hidden_uints, kernel_initializer='he_uniform'),
        ])

        self.relu = []
        self.dense = []
        for l in range(self.layers_num):

            self.relu.append(Activation('relu'))
            self.dense.append(Dense(units=hidden_uints, kernel_initializer='he_uniform'))

        self.dense_out = None

    def build(self, input_shape):
        in_channel = input_shape[-1]

        self.dense_out = Sequential([
            Activation('relu'),
            Dense(units=in_channel, kernel_initializer='he_uniform'),
        ])

        self.build_graph(input_shape[1:])

    def build_graph(self, input_shape):
        inputs = Input(shape=input_shape)
        return Model(inputs=inputs, outputs=self.call(inputs))

    def call(self, inputs, training=None, mask=None):
        x = inputs

        h = self.dense_in(x)
        for i in range(self.layers_num):
            z = self.relu[i](h)
            h = z + self.dense[i](z)

        y = x - self.dense_out(h)

        return y