from utils import *
import mindspore

# ======================Embedding=========================
# item embedding
class ItemEmbedding(mindspore.nn.Cell):
    def __init__(self, n_layer, in_dim, embedding_dim, name, activation='sigmoid'):
        super(ItemEmbedding, self).__init__(auto_prefix=True)
        self.input_size = in_dim
        self.name = name
        fcs = []
        last_size = self.input_size
        hid_dim = int(self.input_size/2)

        for i in range(n_layer - 1):
            linear_model = mindspore.nn.Dense(last_size, hid_dim, bias_init=0)
            fcs.append(linear_model)
            last_size = hid_dim
            fcs.append(activation_func(activation))

        self.fc = mindspore.nn.SequentialCell(*fcs)

        finals = [mindspore.nn.Dense(last_size, embedding_dim), activation_func(activation)]
        self.final_layer = mindspore.nn.SequentialCell(*finals)

    def construct(self, x):
        x = self.fc(x)
        out = self.final_layer(x)
        return out

# user embedding
class UserEmbedding(mindspore.nn.Cell):
    def __init__(self, n_layer, in_dim, embedding_dim, name, activation='sigmoid'):
        super(UserEmbedding, self).__init__(auto_prefix=True)
        self.input_size = in_dim
        self.name = name

        fcs = []
        last_size = self.input_size
        hid_dim = int(self.input_size / 2)

        for i in range(n_layer - 1):
            linear_model = mindspore.nn.Dense(last_size, hid_dim, bias_init=0)
            fcs.append(linear_model)
            last_size = hid_dim
            fcs.append(activation_func(activation))

        self.fc = mindspore.nn.SequentialCell(*fcs)

        finals = [mindspore.nn.Dense(last_size, embedding_dim), activation_func(activation)]
        self.final_layer = mindspore.nn.SequentialCell(*finals)

    def construct(self, x):
        x = self.fc(x)
        out = self.final_layer(x)
        return out
