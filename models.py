from utils import *
import mindspore
from mindspore.dataset import GeneratorDataset
from typing import Tuple


class BASEModel(mindspore.nn.Cell):
    def __init__(self, input1_module, input2_module, embedding1_module, embedding2_module, rec_module):
        super(BASEModel, self).__init__(auto_prefix=True)

        self.input_user_loading = input1_module
        self.input_item_loading = input2_module
        self.user_embedding = embedding1_module
        self.item_embedding = embedding2_module
        self.rec_model = rec_module

    def construct(self, x1_x2: Tuple):
        x1, x2 = x1_x2
        pu, pi = self.input_user_loading(x1), self.input_item_loading(x2)
        eu, ei = self.user_embedding(pu), self.item_embedding(pi)
        rec_value = self.rec_model(eu, ei)
        return rec_value

    def get_weights(self):
        u_emb_params = get_params(self.user_embedding.trainable_params())
        i_emb_params = get_params(self.item_embedding.trainable_params())
        rec_params = get_params(self.rec_model.trainable_params())
        return u_emb_params, i_emb_params, rec_params

    def get_zero_weights(self):
        zeros_like_u_emb_params = get_zeros_like_params(self.user_embedding.trainable_params())
        zeros_like_i_emb_params = get_zeros_like_params(self.item_embedding.trainable_params())
        zeros_like_rec_params = get_zeros_like_params(self.rec_model.trainable_params())
        return zeros_like_u_emb_params, zeros_like_i_emb_params, zeros_like_rec_params

    def init_weights(self, u_emb_para, i_emb_para, rec_para):
        init_params(self.user_embedding.trainable_params(), u_emb_para)
        init_params(self.item_embedding.trainable_params(), i_emb_para)
        init_params(self.rec_model.trainable_params(), rec_para)

    def get_grad(self):
        u_grad = get_grad(self.user_embedding.trainable_params())
        i_grad = get_grad(self.item_embedding.trainable_params())
        r_grad = get_grad(self.rec_model.trainable_params())
        return u_grad, i_grad, r_grad

    def init_u_mem_weights(self, u_emb_para, mu, tao, i_emb_para, rec_para):
        init_u_mem_params(self.user_embedding.trainable_params(), u_emb_para, mu, tao)
        init_params(self.item_embedding.trainable_params(), i_emb_para)
        init_params(self.rec_model.trainable_params(), rec_para)

    def init_ui_mem_weights(self, att_values, task_mem):
        # init the weights only for the mem layer
        u_mui = task_mem.read_head(att_values)
        init_ui_mem_params(self.rec_model.mem_layer.trainable_params(), u_mui)

    def get_ui_mem_weights(self):
        return get_params(self.rec_model.mem_layer.trainable_params())


class LOCALUpdate:
    def __init__(self, your_model, u_idx, dataset, sup_size, que_size, bt_size, n_loop, update_lr, top_k, device):
        self.s_x1, self.s_x2, self.s_y, self.s_y0, self.q_x1, self.q_x2, self.q_y, self.q_y0 = load_user_info(u_idx,
                                                                                                              dataset,
                                                                                                              sup_size,
                                                                                                              que_size,
                                                                                                              device)
        self.user_data = UserDataLoader(self.s_x1, self.s_x2, self.s_y, self.s_y0)
        # self.user_data_loader = GeneratorDataset(self.user_data,
        #                                          ["user_info", "item_info", "ratings", "cold_labels"], shuffle=False)
        self.model = your_model

        self.update_lr = update_lr
        self.optimizer = mindspore.nn.Adam(self.model.trainable_params(), lr=self.update_lr)

        self.loss_fn = mindspore.nn.CrossEntropyLoss()

        self.n_loop = n_loop
        self.top_k = top_k

        self.device = device

        self.s_x1, self.s_x2, self.s_y = self.s_x1, self.s_x2, self.s_y
        self.q_x1, self.q_x2, self.q_y = self.q_x1, self.q_x2, self.q_y


    def train(self):
        # net_with_loss = mindspore.nn.WithLossCell(self.model, self.loss_fn)
        # train_network = mindspore.nn.TrainOneStepCell(net_with_loss, self.optimizer)
        # train_network.set_train()
#
        # for i in range(self.n_loop):
        #     # on support set
        #     idx = 0
        #     for x1, x2, y, y0 in self.user_data:
        #         print(f'data: {idx}')
        #         idx += 1
        #         # 在前面添加一个维度
        #         x1 = mindspore.ops.reshape(x1, (1, len(x1)))
        #         x2 = mindspore.ops.reshape(x2, (1, len(x2)))
#
        #         y = y.view(1)
#
        #         # 计算loss的时候，需要把y变成one-hot编码
        #         label_y = mindspore.ops.OneHot()(y, 5,
        #                                          mindspore.Tensor(1.0, mindspore.float32),
        #                                          mindspore.Tensor(0.0, mindspore.float32))
#
        #         loss = train_network((x1, x2), label_y)
#
        # loss = train_network((self.q_x1, self.q_x2), self.q_y.astype('int32'))
#
        # u_grad, i_grad, r_grad = self.model.get_grad()
        # return u_grad, i_grad, r_grad

        def forward_fn(data, label):
            logits = self.model(data)
            loss = self.loss_fn(logits, label)
            return loss, logits

        grad_fn = mindspore.value_and_grad(forward_fn, None, self.optimizer.parameters, has_aux=True)

        def train_step(data, label):
            (loss, _), grads = grad_fn(data, label)
            self.optimizer(grads)
            return loss, grads



        for i in range(self.n_loop):
            for x1, x2, y, y0 in self.user_data:
                x1 = mindspore.ops.reshape(x1, (1, len(x1)))
                x2 = mindspore.ops.reshape(x2, (1, len(x2)))

                y = y.view(1)

                # 计算loss的时候，需要把y变成one-hot编码
                label_y = mindspore.ops.OneHot()(y, 5,
                                                mindspore.Tensor(1.0, mindspore.float32),
                                                mindspore.Tensor(0.0, mindspore.float32))

                loss, grads = train_step((x1, x2), label_y)

        # 最后需要梯度
        loss, grads = train_step((self.q_x1, self.q_x2), self.q_y.astype('int32'))

        def get_param_list(start, end):
            params = []
            count = 0
            while start <= end:
                if count % 2 == 0:
                    value = deepcopy(grads[start])
                    params.append(value)
                    del value
                count += 1
                start += 1
            return params

        u_emb_params = get_param_list(start=9, end=12)
        i_emb_params = get_param_list(start=13, end=16)
        rec_params = get_param_list(start=17, end=22)

        return u_emb_params, i_emb_params, rec_params

    def test(self):
        for i in range(self.n_loop):
            for x1, x2, y, y0 in self.user_data:
                x1 = mindspore.ops.reshape(x1, (1, len(x1)))
                x2 = mindspore.ops.reshape(x2, (1, len(x2)))

                y = y.view(1)

                # 计算loss的时候，需要把y变成one-hot编码
                label_y = mindspore.ops.OneHot()(y, 5,
                                                mindspore.Tensor(1.0, mindspore.float32),
                                                mindspore.Tensor(0.0, mindspore.float32))

                logits = self.model((x1, x2))
                loss = self.loss_fn(logits, label_y)


def maml_train(raw_phi_u, raw_phi_i, raw_phi_r, u_grad_list, i_grad_list, r_grad_list, global_lr):
    phi_u = update_parameters(raw_phi_u, u_grad_list, global_lr)
    phi_i = update_parameters(raw_phi_i, i_grad_list, global_lr)
    phi_r = update_parameters(raw_phi_r, r_grad_list, global_lr)
    return phi_u, phi_i, phi_r


def user_mem_init(u_id, dataset, device, feature_mem, loading_model, alpha):
    path = 'data_processed/' + dataset + '/raw/'
    u_x1_data = pickle.load(open('{}sample_{}_x1.p'.format(path, str(u_id)), 'rb'))
    u_x1 = to_tensor([u_x1_data])
    pu = loading_model(u_x1)
    personalized_bias_term, att_values = feature_mem.read_head(pu, alpha)
    del u_x1_data, u_x1, pu
    return personalized_bias_term, att_values
