import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv,SAGEConv


from .layers import MultiHeadGraphAttention, GraphConvolution

PRIMITIVES = [
    # 'none',
    #'SAGE',
    'GCN',
    'GAT',
    #'SGC'
]

OPS = {
    'none': lambda args: Zero_graph(),
    'GCN': lambda args: GCN_darts(args),
    'SGC': lambda args: SGCNet(),
    'SAGE': lambda args: GraphSAGE(),
    'GAT': lambda args: GAT_darts(n_units=[300, 300, 300], n_heads=[2, 2], dropout=0.0, attn_dropout=0.0,
                                  instance_normalization=False, diag=True)#todo

}

ATTR = [
    'linear',
    'gauss',
    'linear_bert'
]

IMG = [
    # 'none',
    'linear_img',
    'gauss'
]

REL = [
    # 'none',
    'linear_rel',
    'gauss'

]

OPS_rel = {
    'none': lambda args, rel: Zero(rel),
    'linear_rel': lambda args, input_dim: linear_rel(input_dim),
    'gauss': lambda args, input_dim: gaussian_kernel(input_dim),
}

OPS_img = {
    # 'none': lambda args, img: Zero(img),
    'linear_img': lambda args, input_dim: linear_img(input_dim),
    'gauss': lambda args, input_dim: gaussian_kernel(input_dim),
}

OPS_attr = {
    'linear': lambda args,att1,att2, input_dim: Linear(args,att1,input_dim),
    'gauss': lambda args,att1,att2, input_dim: gaussian_kernel_att(att1,input_dim),
   'linear_bert': lambda args,att1,att2, input_dim: Linearbert(args,att2,768)
}

FUSION = [
    'mlcea',
    'concat',
    #'sum',
    #'self_attention'
]

OPS_fusion = {
    'concat': lambda args: Concat(args),
    'sum': lambda args: Sum(args),
    'mlcea': lambda args: Mclea(),
    #'self_attention': lambda args: SelfAttention()
}


def kernel_sigmas(n_kernels):
    l_sigma = [2]  # for exact match.
    # small variance -> exact match
    if n_kernels == 1:
        return l_sigma
    l_sigma += [2] * (n_kernels - 1)
    return torch.FloatTensor(l_sigma)


def kernel_mus(n_kernels):
    l_mu = [1]
    if n_kernels == 1:
        return l_mu
    bin_size = 1.0 / (n_kernels - 1)  # score range from [0, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in range(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return torch.FloatTensor(l_mu)



class ConvolutionalModel(nn.Module):
    def __init__(self, input_dim, output_dim=100):
        super(ConvolutionalModel, self).__init__()
        self.conv_layer = nn.Conv1d(input_dim, output_dim)
        self.pooling = nn.MaxPool1d(kernel_size=2)  # 可以选择最大池化或平均池化

    def forward(self, x):
        # 输入形状：(batch_size, input_dim, sequence_length)

        x = self.conv_layer(x)
        x = self.pooling(x)
        return x

class gaussian_kernel_att(nn.Module):
    def __init__(self,att, input_dim):
        super(gaussian_kernel_att, self).__init__()
        self.att = att
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, 500)#TODO
        self.fc1 = nn.Linear(500,300)

    def forward(self):
        x= self.att
        # 计算欧几里得距离的平方
        sigmas = kernel_sigmas(self.input_dim)
        mus = kernel_mus(self.input_dim)

        sigmas = sigmas.view(1, -1).cuda()
        mus = mus.view(1, -1).cuda()

        # 计算高斯核矩阵
        kernel_matrix = torch.exp((- ((x - mus) ** 2) / (sigmas ** 2) / 2))

        x1 = self.fc(kernel_matrix)
        x = self.fc1(F.sigmoid(x1))


        return x

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, 500)#TODO
        self.fc1 = nn.Linear(500,300)

    def forward(self, x):
        x1 = self.fc(x)
        x = self.fc1(F.sigmoid(x1))


        return x

class gaussian_kernel(nn.Module):
    def __init__(self, input_dim):
        super(gaussian_kernel, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Linear(input_dim, 500)#TODO
        self.fc1 = nn.Linear(500,300)

    def forward(self, x):
        # 计算欧几里得距离的平方
        sigmas = kernel_sigmas(self.input_dim)
        mus = kernel_mus(self.input_dim)

        sigmas = sigmas.view(1, -1).cuda()
        mus = mus.view(1, -1).cuda()

        # 计算高斯核矩阵
        kernel_matrix = torch.exp((- ((x - mus) ** 2) / (sigmas ** 2) / 2))

        x1 = self.fc(kernel_matrix)
        x = self.fc1(F.relu(x1))


        return x


class Mclea(nn.Module):
    def __init__(self):
        super().__init__()
        self.modal_num = 4 #todo
        self.requires_grad = True if 1 > 0 else False
        self.weight = nn.Parameter(torch.ones((self.modal_num, 1)),
                                   requires_grad=self.requires_grad)

    def forward(self, x1, x2, x3, x4, x5, x6):
        # -------------修改部分 ---------- #
        # x5 = None
        # x6 = None
        # -------------修改部分 ---------- #
        if x5 is not None:
              embs = [x1, x2, x3, x4, x5, x6]
              weight_norm = F.softmax(self.weight, dim=0)
              embs = [weight_norm[idx] * F.normalize(embs[idx]) for idx in range(self.modal_num) if embs[idx] is not None]
              joint_emb = torch.cat(embs, dim=1)
        else:
            embs = [x1, x2, x3, x4]
            weight_norm = F.softmax(self.weight, dim=0)
            embs = [weight_norm[idx] * F.normalize(embs[idx]) for idx in range(self.modal_num) if embs[idx] is not None]
            joint_emb = torch.cat(embs, dim=1)


        return joint_emb


class Concat(nn.Module):
    def __init__(self, args):
        super(Concat, self).__init__()
        self.args = args
        self.fc = nn.Linear(600, 300)
        self.dropout = nn.Dropout(args.drpt)

    # -------------修改部分 ---------- #
    # def forward(self, x1, x2, x3, x4):
    #     out_1 = torch.cat([x1, x2, x3, x4], dim=1)
    #     out = F.normalize(out_1)
    #     out = self.dropout(out)
    #     return out
    # -------------修改部分 ---------- #
    def forward(self, x1, x2, x3, x4, x5, x6):
        if x5 is not None:
            out_1 = torch.cat([x1, x2, x3, x4, x5, x6], dim=1)
        else:
            out_1 = torch.cat([x1, x2, x3, x4], dim=1)
        out = F.normalize(out_1)
        out = self.dropout(out)
        return out

class Sum(nn.Module):
    def __init__(self, args):
        super(Sum, self).__init__()
        self.args = args
        self.fc = nn.Linear(300,1200)

    def forward(self, x1, x2, x3, x4,x5,x6):
        out = x1 + x2 + x3 + x4
        out_1 = self.fc(out)
        return out_1






class Linear(nn.Module):
    def __init__(self, args,att,input_dim):
        super(Linear, self).__init__()
        self.att = att

        self.att_fc = nn.Linear(input_dim, 300)

    def forward(self):
        out = self.att_fc(self.att)
        return out

class Linearbert(nn.Module):
    def __init__(self, args,att,input_dim):
        super(Linearbert, self).__init__()
        self.att = att

        self.att_fc = nn.Linear(input_dim, 300)

    def forward(self):
        out = self.att_fc(self.att)
        return out


class linear_img(nn.Module):
    def __init__(self, input_dim):
        super(linear_img, self).__init__()
        self.img_fc = nn.Linear(input_dim, 300)

    def forward(self, emd):
        out = self.img_fc(emd)
        return out


class linear_rel(nn.Module):
    def __init__(self, input_dim):
        super(linear_rel, self).__init__()

        self.att_fc = nn.Linear(input_dim, 300)

    def forward(self, emd):
        out = self.att_fc(emd)
        return out


class Zero(nn.Module):
    def __init__(self, gcn):
        super(Zero, self).__init__()
        self.gcn = gcn

    def forward(self):
        # x = self.gcn[x]
        # out = x.mul(0.)
        # return out[27793, :100]
        return torch.zeros(27793, 100).cuda()


class Zero_graph(nn.Module):
    def __init__(self):
        super(Zero_graph, self).__init__()

    def forward(self, x, adj):
        # out = torch.zeros((len(ent), 100))

        # return out.cuda()

        return torch.zeros(27793, 100).cuda()


class GCN_darts(nn.Module):
    def __init__(self, args):
        super(GCN_darts, self).__init__()
        self.gc1 = GraphConvolution(300, 300)
        self.gc2 = GraphConvolution(300, 300)
        self.dropout = 0.0

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # change to leaky relu
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        out = x

        return out.cuda()


class GAT_darts(nn.Module):
    def __init__(self, n_units, n_heads, dropout, attn_dropout, instance_normalization, diag):
        super(GAT_darts, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        for i in range(self.num_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                MultiHeadGraphAttention(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False))

    def forward(self, x, adj):
        if self.inst_norm:
            x = self.norm(x)
        for i, gat_layer in enumerate(self.layer_stack):
            if i + 1 < self.num_layer:
                x = F.dropout(x, self.dropout, training=self.training)

            x = gat_layer(x, adj)

            if self.diag:
                x = x.mean(dim=0)
            if i + 1 < self.num_layer:
                if self.diag:
                    x = F.elu(x)
                else:
                    x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))
        if not self.diag:
            x = x.mean(dim=0)

        return x

class SGCNet(nn.Module):
    def __init__(self):
        super(SGCNet, self).__init__()
        self.conv1 = SGConv(300, 300)  # Define SGC layer


    def forward(self,  x, adj):
        x1, edge_index = x, adj  # Get node features and edge indices
        x2 = self.conv1(x1, edge_index)  # Apply SGC layer
        x3 = F.relu(x2)  # Apply ReLU activation
        x = F.dropout(x3, p=0.0, training=self.training)  # Apply dropout for regularization

        return F.normalize(x)


class GraphSAGE(torch.nn.Module):
    def __init__(self):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(300, 150)
        self.sage2 = SAGEConv(150, 300)

    def forward(self, features, edges):
        features1 = self.sage1(features, edges)
        features2 = F.relu(features1)
        features3 = F.dropout(features2, p=0.0, training=self.training)
        features4 = self.sage2(features3, edges)
        return features4



class EncodingMixedOp_graph(nn.Module):

    def __init__(self, args, adj):
        super().__init__()
        self.adj = adj

        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            op = OPS[primitive](args)
            self._ops.append(op)

    def forward(self, emd, weights):
        op_result = [op(emd, self.adj) for w, op in zip(weights, self._ops)]
        out = sum(w * result for w, result in zip(weights, op_result))
        oy = op_result[:]
        return out,oy


class EncodingMixedOp_attr(nn.Module):

    def __init__(self, args, att_1,att_2,input_dim):
        super().__init__()


        self._ops = nn.ModuleList()
        for primitive in ATTR:
            op = OPS_attr[primitive](args,att_1,att_2, input_dim)
            self._ops.append(op)

    def forward(self,  weights):
        op_result = [op() for w, op in zip(weights, self._ops)]
        out = sum(w * result for w, result in zip(weights, op_result))
        oy = op_result[:]
        return out, oy


class EncodingMixedOp_img(nn.Module):

    def __init__(self, args, input_dim):
        super().__init__()

        self._ops = nn.ModuleList()
        for primitive in IMG:
            op = OPS_img[primitive](args, input_dim)
            self._ops.append(op)

    def forward(self, emd, weights):
        op_result = [op(emd) for w, op in zip(weights, self._ops)]
        out = sum(w * result for w, result in zip(weights, op_result))
        oy = op_result[:]
        return out, oy


class EncodingMixedOp_rel(nn.Module):
    def __init__(self, args, input_dim):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in REL:
            op = OPS_rel[primitive](args, input_dim)
            self._ops.append(op)

    def forward(self, emd, weights):
        op_result = [op(emd) for w, op in zip(weights, self._ops)]
        out = sum(w * result for w, result in zip(weights, op_result))
        oy = op_result[:]
        return out, oy


class EncodingMixedOp_fusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self._ops = nn.ModuleList()
        for primitive in FUSION:
            op = OPS_fusion[primitive](args)
            self._ops.append(op)

    def forward(self, x, y, z, m, weights):
        out = sum(w * op(x, y, z, m,x5=None,x6=None) for w, op in zip(weights, self._ops))


        return out
