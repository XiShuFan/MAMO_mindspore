import mindspore.ops

from utils import *
import mindspore

class FeatureMem:
    def __init__(self, n_k, u_emb_dim, base_model, device):
        self.n_k = n_k
        self.base_model = base_model
        self.p_memory = mindspore.ops.StandardNormal()((n_k, u_emb_dim))
        
        u_param, _, _ = base_model.get_weights()
        self.u_memory = []
        for i in range(n_k):
            bias_list = []
            for param in u_param:
                bias_list.append(param)
            self.u_memory.append(bias_list)
        self.att_values = mindspore.ops.Zeros()((n_k), mindspore.float32)
        
        self.device = device

    def read_head(self, p_u, alpha, train=True):
        # get personalized mu
        att_model = Attention(self.n_k)
        attention_values = att_model(p_u, self.p_memory)
        personalized_mu = get_mu(attention_values, self.u_memory, self.base_model, self.device)
        # update mp
        transposed_att = attention_values.reshape(self.n_k, 1)
        product = mindspore.ops.MatMul()(transposed_att, p_u)
        if train:
            self.p_memory = alpha * product + (1-alpha) * self.p_memory
        self.att_values = attention_values
        return personalized_mu, attention_values

    def write_head(self, u_grads, lr):
        update_mu(self.att_values, self.u_memory, u_grads, lr)


class TaskMem:
    def __init__(self, n_k, emb_dim, device):
        self.n_k = n_k
        self.memory_UI = mindspore.ops.UniformReal()((n_k, emb_dim *2, emb_dim*2))
        self.att_values = mindspore.ops.Zeros()((n_k), mindspore.float32)

    def read_head(self, att_values):
        self.att_values = att_values
        return get_mui(att_values, self.memory_UI, self.n_k)

    def write_head(self, u_mui, lr):
        update_values = update_mui(self.att_values, self.n_k, u_mui)
        self.memory_UI = lr* update_values + (1-lr) * self.memory_UI


def cosine_similarity(input1, input2):
    query_norm = mindspore.ops.Sqrt()(mindspore.ops.ReduceSum()(input1**2+0.00001, 1))
    doc_norm = mindspore.ops.Sqrt()(mindspore.ops.ReduceSum()(input2**2+0.00001, 1))

    prod = mindspore.ops.ReduceSum()(mindspore.ops.Mul()(input1, input2), 1)
    norm_prod = mindspore.ops.Mul()(query_norm, doc_norm)

    cos_sim_raw = mindspore.ops.div(prod, norm_prod)
    return cos_sim_raw


class Attention(mindspore.nn.Cell):
    def __init__(self, n_k, activation='relu'):
        super(Attention, self).__init__()
        self.n_k = n_k
        self.fc_layer = mindspore.nn.Dense(self.n_k, self.n_k, activation=activation_func(activation))
        self.soft_max_layer = mindspore.nn.Softmax()

    def construct(self, pu, mp):
        expanded_pu = mindspore.ops.Reshape()(mindspore.ops.tile(pu, (1, len(mp))), (len(mp), -1))
        inputs = cosine_similarity(expanded_pu, mp)
        inputs_len = len(inputs)
        # add one dimension
        inputs = mindspore.ops.reshape(inputs, (1, inputs_len))
        fc_layers = self.fc_layer(inputs)
        # remove one dimension
        # fc_layers = mindspore.ops.reshape(fc_layers, (inputs_len,))
        attention_values = self.soft_max_layer(fc_layers)
        return attention_values


def get_mu(att_values, mu, model, device):
    mu0,_,_ = model.get_zero_weights()
    attention_values = att_values.reshape(len(mu),1)
    for i in range(len(mu)):
        for j in range(len(mu[i])):
            mu0[j] += attention_values[i] * mu[i][j]
    return mu0


def update_mu(att_values, mu, grads, lr):
    att_values = att_values.reshape(len(mu), 1)
    for i in range(len(mu)):
        for j in range(len(mu[i])):
            mu[i][j] = lr * att_values[i] * grads[j] + (1-lr) * mu[i][j]


def get_mui(att_values, mui, n_k):
    attention_values = att_values.reshape(n_k, 1, 1)
    attend_mui = mindspore.ops.Mul()(attention_values, mui)
    u_mui = attend_mui.sum(axis=0)
    return u_mui


def update_mui(att_values, n_k, u_mui):
    repeat_u_mui = mindspore.ops.tile(u_mui.unsqueeze(0), (n_k, 1, 1))
    att_values = att_values.unsqueeze(0)
    attention_tensor = mindspore.ops.Reshape()(att_values, (n_k, 1, 1))
    attend_u_mui = mindspore.ops.Mul()(attention_tensor, repeat_u_mui)
    return attend_u_mui
