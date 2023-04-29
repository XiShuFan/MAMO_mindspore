from utils import *
import mindspore

class RecMAM(mindspore.nn.Cell):
    def __init__(self, embedding_dim, n_y, n_layer, name, activation='sigmoid', classification=True):
        super(RecMAM, self).__init__(auto_prefix=True)
        self.input_size = embedding_dim * 2
        self.name = name
        self.mem_layer = mindspore.nn.Dense(self.input_size, self.input_size)

        fcs = []
        last_size = self.input_size

        for i in range(n_layer - 1):
            out_dim = int(last_size / 2)
            linear_model = mindspore.nn.Dense(last_size, out_dim)
            fcs.append(linear_model)
            last_size = out_dim
            fcs.append(activation_func(activation))

        self.fc = mindspore.nn.SequentialCell(*fcs)

        if classification:
            finals = [mindspore.nn.Dense(last_size, n_y), activation_func('softmax')]
        else:
            finals = [mindspore.nn.Dense(last_size, 1)]
        self.final_layer = mindspore.nn.SequentialCell(*finals)

    def construct(self, x1, x2):
        x = mindspore.ops.operations.Concat(axis=1)([x1, x2])
        out0 = self.mem_layer(x)
        out = self.fc(out0)
        out = self.final_layer(out)
        return out
