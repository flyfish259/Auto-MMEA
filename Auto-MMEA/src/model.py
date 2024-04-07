import os
import pickle

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import src.aux_models as aux
from .operations import *
import math
from collections import Counter
from collections import namedtuple
from src.search_darts import *
import torch
from transformers import BertTokenizer, BertModel
from .common import *


Genotype = namedtuple('Genotype', 'edges')

def icl_loss(zis, output_zis, output_zjs):
    n_view = 2
    temperature = 0.1
    LARGE_NUM = 1e9
    num_classes = len(zis) * n_view
    hidden1, hidden2 = output_zis, output_zjs
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = F.one_hot(torch.arange(start=0, end=len(zis), dtype=torch.int64),
                       num_classes=num_classes).float()
    labels = labels.cuda()

    masks = F.one_hot(torch.arange(start=0, end=len(zis), dtype=torch.int64),
                      num_classes=len(zis))
    masks = masks.cuda().float()
    logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large, 0, 1)) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(hidden2, torch.transpose(hidden2_large, 0, 1)) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large, 0, 1)) / temperature
    logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large, 0, 1)) / temperature
    logits_a = torch.cat([logits_ab, logits_bb], dim=1)
    logits_b = torch.cat([logits_ba, logits_aa], dim=1)
    loss_a = softXEnt(labels, logits_a).cuda()
    loss_b = softXEnt(labels, logits_b).cuda()
    loss = 0.5 * loss_a + loss_b * 0.5
    return loss

def ial_loss(src_zis, src_zjs, tar_zis, tar_zjs):
    n_view = 2
    temperature = 4
    LARGE_NUM = 1e9
    alpha = 0.5
    batch_size = len(src_zjs)
    masks = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size)
    masks = masks.cuda().float()
    p_ab = torch.matmul(src_zis, torch.transpose(src_zjs, 0, 1)) / temperature
    p_ba = torch.matmul(src_zjs, torch.transpose(src_zis, 0, 1)) / temperature
    q_ab = torch.matmul(tar_zis, torch.transpose(tar_zjs, 0, 1)) / temperature
    q_ba = torch.matmul(tar_zjs, torch.transpose(tar_zis, 0, 1)) / temperature
    # add self-contrastive
    p_aa = torch.matmul(src_zis, torch.transpose(src_zis, 0, 1)) / temperature
    p_bb = torch.matmul(src_zjs, torch.transpose(src_zjs, 0, 1)) / temperature
    q_aa = torch.matmul(tar_zis, torch.transpose(tar_zis, 0, 1)) / temperature
    q_bb = torch.matmul(tar_zjs, torch.transpose(tar_zjs, 0, 1)) / temperature
    p_aa = p_aa - masks * LARGE_NUM
    p_bb = p_bb - masks * LARGE_NUM
    q_aa = q_aa - masks * LARGE_NUM
    q_bb = q_bb - masks * LARGE_NUM
    p_ab = torch.cat([p_ab, p_aa], dim=1)
    p_ba = torch.cat([p_ba, p_bb], dim=1)
    q_ab = torch.cat([q_ab, q_aa], dim=1)
    q_ba = torch.cat([q_ba, q_bb], dim=1)

    # param 1 need to log_softmax, param 2 need to softmax
    # loss_a = F.kl_div(F.log_softmax(p_ab, dim=1), F.softmax(q_ab.detach(), dim=1), reduction="none")
    # loss_b = F.kl_div(F.log_softmax(p_ba, dim=1), F.softmax(q_ba.detach(), dim=1), reduction="none")

    margin = 0.5
    beta = 10

    euclidean_distance = torch.sqrt(torch.sum(torch.pow(p_ab - q_ab, 2), dim=1))

    # 计算CMD损失
    contrastive_loss = torch.mean((1 - euclidean_distance) ** 2)
    discriminative_loss = torch.mean(torch.relu(margin - euclidean_distance))
    loss_a = contrastive_loss + beta * discriminative_loss

    euclidean_distance = torch.sqrt(torch.sum(torch.pow(p_ba - q_ba, 2), dim=1))

    # 计算CMD损失
    contrastive_loss = torch.mean((1 - euclidean_distance) ** 2)
    discriminative_loss = torch.mean(torch.relu(margin - euclidean_distance))
    loss_b = contrastive_loss + beta * discriminative_loss

    loss_a = loss_a.mean()
    loss_b = loss_b.mean()

    return (alpha * loss_a + (1 - alpha) * loss_b) * 0.1

def dynamic_loss_weight(loss_now,loss_t_1,loss_t_2):
    """

    :param loss_t_1: 每个task上一轮的loss列表，并且为标量
    :param loss_t_2:
    :return:
    """
    T=10
    assert len(loss_t_1) == len(loss_t_2)
    task_n = len(loss_t_1)



    w = [l_1 / l_2 for l_1, l_2 in zip(loss_t_1, loss_t_2)]

    lamb = [math.exp(v / T) for v in w]

    lamb_sum = sum(lamb)

    a = [task_n * l / lamb_sum for l in lamb]
    loss_all = sum(a[it] * loss_now[it] for it in range(len(loss_now)))
    return  loss_all

def dynamic_loss_weight_1(loss_now,loss_t_1,loss_t_2):
    """

    :param loss_t_1: 每个task上一轮的loss列表，并且为标量
    :param loss_t_2:
    :return:
    """
    T=10
    assert len(loss_t_1) == len(loss_t_2)
    task_n = len(loss_t_1)



    w = [l_1 / l_2 for l_1, l_2 in zip(loss_t_1, loss_t_2)]

    lamb = [math.exp(v / T) for v in w]

    lamb_sum = sum(lamb)

    a = [task_n * l / lamb_sum for l in lamb]
    loss_all = sum(a[it] * loss_now[it] for it in range(len(loss_now)))
    return  loss_all,[a[it] * loss_now[it] for it in range(len(loss_now))]

def softXEnt(target, logits, replay=False, neg_cross_kg=False):
    # torch.Size([2239, 4478])

    logprobs = F.log_softmax(logits, dim=1)
    loss = -(target * logprobs).sum() / logits.shape[0]
    if replay:
        logits = logits
        idx = torch.arange(start=0, end=logprobs.shape[0], dtype=torch.int64).cuda()
        stg_neg = logits.argmax(dim=1)
        new_value = torch.zeros(logprobs.shape[0]).cuda()
        index = (
            idx,
            stg_neg,
        )
        logits = logits.index_put(index, new_value)
        stg_neg_2 = logits.argmax(dim=1)
        tmp = idx.eq_(stg_neg)
        neg_idx = stg_neg - stg_neg * tmp + stg_neg_2 * tmp
        return loss, neg_idx

    return loss


def load_relation(e, KG, topR=1000):
    # (39654, 1000)
    rel_mat = np.zeros((e, topR), dtype=np.float32)
    rels = np.array(KG)[:, 1]
    top_rels = Counter(rels).most_common(topR)
    rel_index_dict = {r: i for i, (r, cnt) in enumerate(top_rels)}
    for tri in KG:
        h = tri[0]
        r = tri[1]
        o = tri[2]
        if r in rel_index_dict:
            rel_mat[h][rel_index_dict[r]] += 1.
            rel_mat[o][rel_index_dict[r]] += 1.
    return np.array(rel_mat)


def load_attr(fns, e, ent2id, topA=1000):
    cnt = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    attr2id = {}
    # pdb.set_trace()

    for i in range(min(topA, len(fre))):
        attr2id[fre[i][0]] = i
    attr = np.zeros((e, topA), dtype=np.float32)
    # todo
    # attr_bert = np.zeros((e, 768), dtype=np.float32)
    #
    # model_name = 'bert-base-uncased'
    # tokenizer = BertTokenizer.from_pretrained(model_name)
    # model = BertModel.from_pretrained(model_name)
    # op= 0

    for fn in fns:
        # op+=1
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                # it = str(' ')
                th = line[:-1].split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        # it = it + th[i].split('/')[-1].split('>')[0].split('.')[-1] + ' '

                        if th[i] in attr2id:
                            attr[ent2id[th[0]]][attr2id[th[i]]] = 1.0
        #             tokens = tokenizer.tokenize(it)
        #             tokens = ['[CLS]'] + tokens + ['[SEP]']
        #             input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #             input_ids = torch.tensor(input_ids).unsqueeze(0)  # 添加batch维度
        #             with torch.no_grad():
        #                 outputs = model(input_ids)
        #                 sentence_vector = outputs.last_hidden_state[:, 0, :]  # 获取[CLS]对应的向量
        #                 attr_bert[ent2id[th[0]]] = sentence_vector
        # name = 'att_{}.pkl'.format(op)
        # with open(name,'wb') as f:
        #     pickle.dump(attr_bert,f)

    return attr

def load_att_bert(fns, emd,e, ent2id):

    attr_s = torch.zeros((e, 768))
    leng_s = torch.ones((e,1))

    for fn in range(len(emd)):
        with open(emd[fn], 'rb') as f:
            fs = pickle.load(f)
        with open(fns[fn], 'r', encoding='utf-8') as f:
            for line in f:
                th = line.strip('\n').split('\t')
                the = th[0]
                attr = th[1:]
                ioe = 0
                emd_1 = torch.zeros(768)
                for it in attr:
                    try:
                        #attr_emd[ent2id[the]].append(fs[it])
                        emd_1+= fs[it]

                        ioe+=1
                    except:
                        print(it)
                attr_s[ent2id[the]] = emd_1
                leng_s[ent2id[the]] = ioe
    return attr_s/leng_s


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


def dispose_emd_list(emd_list,zis,zjs):
    loss = 0
    for emd_diff in emd_list:
        for it in emd_diff:
            emd_zis = F.normalize(it[zis])
            emd_zjs = F.normalize(it[zjs])
            loss  += icl_loss(zis,emd_zis,emd_zjs)
    return loss*0.005



class Architect(object):
    def __init__(self, model, args, criterion, optimizer):
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = optimizer
        self.loss_before_1 = []
        self.loss_before_2 = []
        self.loss_before_two = []
        self.loss_before_one = []

    def step(self, data_emd,input_valid, target_valid,epoch,writer):
        self.optimizer.zero_grad()
        self._backward_step(data_emd,input_valid, target_valid,epoch,writer)
        self.optimizer.step()


    def _backward_step(self, data_emd,zis, zjs,epoch,writer):

        graph_emb, att_emb, img_emb, rel_emb, out_hid_emb,emd_list= data_emd
        graph_emb = F.normalize(graph_emb, dim=1)
        att_emb = F.normalize(att_emb, dim=1)
        img_emb = F.normalize(img_emb, dim=1)
        rel_emb = F.normalize(rel_emb, dim=1)
        out_hid_emb = F.normalize(out_hid_emb, dim=1)
        s_zis, att_zis, img_zis, rel_zis, out_hid_zis = graph_emb[zis], att_emb[
            zis], \
            img_emb[zis], rel_emb[zis], out_hid_emb[zis]
        s_zjs, att_zjs, img_zjs, rel_zjs, out_hid_zjs = graph_emb[zjs], att_emb[
            zjs], \
            img_emb[zis], rel_emb[zjs], out_hid_emb[zjs]
        loss_joi = icl_loss(zis, out_hid_zis, out_hid_zjs)



        loss_gcn = icl_loss(zis, s_zis, s_zjs)
        loss_att = icl_loss(zis, att_zis, att_zjs)
        loss_img = icl_loss(zis, img_zis, img_zjs)
        loss_rel = icl_loss(zis, rel_zis, rel_zjs)

        loss_now = [loss_gcn, loss_att, loss_img, loss_rel]



        loss_gcn_o = ial_loss(s_zis, s_zjs, out_hid_zis, out_hid_zjs)
        loss_att_o = ial_loss(att_zis, att_zjs, out_hid_zis, out_hid_zjs)
        loss_img_o = ial_loss(img_zis, img_zjs, out_hid_zis, out_hid_zjs)
        loss_rel_o = ial_loss(rel_zis, rel_zjs, out_hid_zis, out_hid_zjs)

        loss_momnet = [loss_gcn_o, loss_att_o, loss_img_o, loss_rel_o]

        if epoch>1:
            loss_in,temi = dynamic_loss_weight_1(loss_now,self.loss_before_1,self.loss_before_2)
            loss_out = dynamic_loss_weight(loss_momnet, self.loss_before_one, self.loss_before_two)
            writer.add_scalar("gcn_1", temi[0], epoch)
            writer.add_scalar("att_1",temi[1], epoch)
            writer.add_scalar("img_1", temi[2], epoch)
            writer.add_scalar("rel_1",temi[3] , epoch)
        else:
            loss_in = loss_gcn + loss_att + loss_img + loss_rel
            loss_out = loss_gcn_o + loss_att_o + loss_img_o + loss_rel_o

        loss_emd = dispose_emd_list(emd_list,zis,zjs)

        if epoch >-1:
            loss = loss_joi + loss_in + loss_out + loss_emd
        else:
            loss =  loss_joi


        loss.backward()
        if epoch==0:
            self.loss_before_2 = [loss_gcn.item(), loss_att.item(), loss_img.item(), loss_rel.item()]
            self.loss_before_two = [loss_gcn_o.item(), loss_att_o.item(), loss_img_o.item(), loss_rel_o.item()]
        elif epoch==1:
            self.loss_before_1 = [loss_gcn.item(), loss_att.item(), loss_img.item(), loss_rel.item()]
            self.loss_before_one = [loss_gcn_o.item(), loss_att_o.item(), loss_img_o.item(), loss_rel_o.item()]
        else:
            self.loss_before_2 = self.loss_before_1
            self.loss_before_1 = loss_now
            self.loss_before_two = self.loss_before_one
            self.loss_before_one = loss_momnet
        del loss_momnet
        del loss_now
        del loss

        torch.cuda.empty_cache()



class EncodingNetwork(nn.Module):
    def __init__(self, args, criterion, KGs):
        super().__init__()
        self.args = args
        self.kgs = KGs
        self.criterion = criterion

        self.input_idx = KGs["input_idx"].cuda()
        self.adj = KGs["adj"].cuda()
        self.dropout = nn.Dropout(args.drpt)
        self.img_mask = KGs["img_mask"]

        # 图结构嵌入
        self._ops_graph = nn.ModuleList()
        self.entity_emb = nn.Embedding(30355, 300)#30355 27793
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(30355))#TODO
        self.entity_emb.requires_grad = True
        self.entity_emb = self.entity_emb.cuda()

        op = EncodingMixedOp_graph(self.args, self.adj)
        self._ops_graph.append(op)
        self._initialize_graph()

        # 属性嵌入
        self._ops_attr = nn.ModuleList()
        a1 = os.path.join(self.args.datadir, 'training_attrs_1')
        a2 = os.path.join(self.args.datadir, 'training_attrs_2')
        att_features_1 = load_attr([a1, a2], self.kgs['ent_num'], self.kgs['ent2id_dict'], 1000)
        self.att_features_1 = torch.Tensor(att_features_1).cuda()
        self.att_features_2 = load_att_bert([a1, a2], ["/home/u2022171243/EA_MMKG/"+self.args.datadir + '/attr_name_1.pkl',
                                                       "/home/u2022171243/EA_MMKG/"+self.args.datadir + '/attr_name_2.pkl'], self.kgs['ent_num'],
                                            self.kgs['ent2id_dict'])
        op = EncodingMixedOp_attr(self.args, self.att_features_1, self.att_features_2.cuda(), input_dim=1000)
        self._ops_attr.append(op)
        self._initialize_attr()

        # 图像嵌入
        self._ops_img = nn.ModuleList()
        self.img_features = F.normalize(torch.FloatTensor(KGs["images_list"])).cuda()
        op = EncodingMixedOp_img(self.args, input_dim=4096)
        self._ops_img.append(op)
        self._initialize_img()

        # 关系嵌入
        self._ops_rel = nn.ModuleList()
        rel_fe = load_relation(KGs['ent_num'], KGs['triples'], 1000)
        self.rel_features = torch.Tensor(rel_fe).cuda()
        op = EncodingMixedOp_rel(self.args, input_dim=1000)
        self._ops_rel.append(op)
        self._initialize_rel()

        # 多模态融合
        self._ops_fusion = nn.ModuleList()
        op = EncodingMixedOp_fusion(self.args)
        self._ops_fusion.append(op)
        self._initialize_fusion()

        self._initialize_grad()




        self._arch_parameters = [self.graph, self.attr, self.img, self.rel, self.fusion,self.grad]





    def _initialize_grad(self):
        num_ops = 1
        # beta controls node cell arch
        self.grad= Variable(1e-3 *torch.randn(num_ops), requires_grad=True)

    def _initialize_fusion(self):
        num_ops = 2
        # beta controls node cell arch
        self.fusion = Variable(1e-3 * torch.randn(num_ops), requires_grad=True)



    def _initialize_graph(self):
        num_ops = 2
        # beta controls node cell arch
        self.graph = Variable(1e-3 * torch.randn(num_ops), requires_grad=True)

    def _initialize_attr(self):
        num_ops = 3
        # beta controls node cell arch
        self.attr = Variable(1e-3 * torch.randn(num_ops), requires_grad=True)

    def _initialize_img(self):
        num_ops = 2
        # beta controls node cell arch
        self.img = Variable(1e-3 * torch.randn(num_ops), requires_grad=True)

    def _initialize_rel(self):
        num_ops = 2
        # beta controls node cell arch
        self.rel = Variable(1e-3 * torch.randn(num_ops), requires_grad=True)

    def forward(self, is_test=False):
        graph_weights = F.softmax(self.graph+torch.randn(self.graph.size()) * 0.001, dim=-1)
        attr_weights = F.softmax(self.attr+torch.randn(self.attr.size()) * 0.001, dim=-1)
        img_weights = F.softmax(self.img+torch.randn(self.img.size()) * 0.001, dim=-1)
        rel_weights = F.softmax(self.rel+torch.randn(self.rel.size()) * 0.001, dim=-1)
        fusion_weights = F.softmax(self.fusion+torch.randn(self.fusion.size()) * 0.001, dim=-1)

        grad_weights = F.softmax(self.grad+torch.randn(self.grad.size()) * 0.001,dim=-1)


        # TODO
        struct_emd,strct_list = self._ops_graph[0](self.entity_emb(self.input_idx), graph_weights)
        attr_emd,attr_lsit = self._ops_attr[0]( attr_weights)

        rel_emd,rel_list = self._ops_rel[0](self.rel_features, rel_weights)

        img_emd,img_list = self._ops_img[0](self.img_features, img_weights)

        out_hid = self._ops_fusion[0](struct_emd,attr_emd,rel_emd,img_emd,fusion_weights)

        return struct_emd, attr_emd, img_emd, rel_emd, out_hid,[strct_list,attr_lsit,rel_list,img_list]

    def arch_parameters(self):
        return self._arch_parameters




class CustomMultiLossLayer(nn.Module):
    """
    Inspired by
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf
    """

    def __init__(self, loss_num, device=None):
        super(CustomMultiLossLayer, self).__init__()
        self.loss_num = loss_num
        self.log_vars = nn.Parameter(torch.zeros(self.loss_num, ), requires_grad=True)

    def forward(self, loss_list):
        assert len(loss_list) == self.loss_num
        precision = torch.exp(-self.log_vars)
        loss = 0
        for i in range(self.loss_num):
            loss += precision[i] * loss_list[i] + self.log_vars[i]
        return loss


class CMDLoss(nn.Module):
    def __init__(self, margin=1.0, alpha=32.0):
        super(CMDLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, view1, view2):
        # 计算欧几里得距离
        euclidean_distance = torch.sqrt(torch.sum(torch.pow(view1 - view2, 2), dim=1))

        # 计算CMD损失
        contrastive_loss = torch.mean((1 - euclidean_distance) ** 2)
        discriminative_loss = torch.mean(torch.relu(self.margin - euclidean_distance))
        cmd_loss = contrastive_loss + self.alpha * discriminative_loss

        return cmd_loss

class Searchable_EA_Net(nn.Module):
    def __init__(self, args, criterion, KGs):
        super().__init__()

        self.args = args
        self.criterion = criterion

        self.steps = args.steps
        self.Kgs = KGs


        self._criterion = criterion

        self.encoding_net = EncodingNetwork(args=self.args, criterion=self.criterion, KGs=self.Kgs)

        # self.multi_loss_layer = CustomMultiLossLayer(loss_num=3).cuda()
        # self.similar_loss_layer = CustomMultiLossLayer(loss_num=6).cuda()
        # self.similar_loss_lay2 = CustomMultiLossLayer(loss_num=6).cuda()
        self.align_multi_loss_layer = CustomMultiLossLayer(loss_num=4).cuda()

    def forward(self):
        out = self.encoding_net()

        return out

    def forward(self, is_test=False):
        out = self.encoding_net(is_test)
        return out


    def central_params(self):
        central_parameters = [
            {'params': self.encoding_net.parameters()},
            # {'params': self.multi_loss_layer.parameters()},
            # {'params': self.align_multi_loss_layer.parameters()}
        ]
        return central_parameters

    def _loss(self, input_features, labels):
        logits = self(input_features)
        return self._criterion(logits, labels)

    def arch_parameters(self):
        return self.encoding_net.arch_parameters()
