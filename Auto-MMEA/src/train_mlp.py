import pickle

import math

import numpy as np
import torch
from torch.cuda.amp import GradScaler

import src.scheduler as sc
import copy
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
from IPython import embed
import torch.optim as op

from src.search_darts import icl_loss
from src.utils import count_parameters, save, save_pickle
import torch.nn.functional as F
import traceback
import logging
from common import *

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    distance = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(distance, 0.0, np.inf)

def test(data_emd, test_left, test_right,last_epoch=False, save_name="", loss=None):
    with open('../img_data.pkl','rb')as f:
        emd_img = pickle.load(f)
    emd_graph = data_emd['graph']
    emd_att = data_emd['att']
    emd_rel = data_emd['rel']
    model = MLP(
        input_dim=900, output_dim=4096,
        dnn_units=[512], dropout_rate=0.0
    )
    model = model.cuda()
    with open('../mask.pkl', 'rb') as f:
        mask_data = pickle.load(f)
    with open('../train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    z_all = np.array(
        [i for i in np.concatenate((train_data[:,0], train_data[:,1])) if mask_data[i] == 1])
    img_optimizer = op.Adam(model.parameters(),
                            lr=0.001, betas=(0.5, 0.999))

    emd_i = emd_img[z_all]
    emd_a = emd_att[z_all]
    emd_r = emd_rel[z_all]
    emd_g = emd_graph[z_all]
    for it in range(10):
        img_optimizer.zero_grad()
        img_synthesis = torch.cat([emd_g, emd_r,emd_a],
                                  dim=-1)
        loss_img_mlp = icl_loss(z_all, model(img_synthesis),
                                emd_i)
        loss_img_mlp.backward(retain_graph=True)
        print(f'第{it}轮的loss为：{loss_img_mlp}')
        img_optimizer.step()


    if True:
       index = mask_data== 0
       emd_img[index] = model(torch.cat([emd_graph,emd_rel,emd_att],dim=-1))[index]
       emd_img = F.normalize(emd_img).cuda()




    top_k = [1, 10, 50]
    acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
    acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
    test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.
    distance = pairwise_distances(emd_left, emd_right)
    not_img = 0
    not_i = 0

    for idx in range(test_left.shape[0]):
        values, indices = torch.sort(distance[idx, :], descending=False)
        rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()


        mean_l2r += (rank + 1)
        mrr_l2r += 1.0 / (rank + 1)
        for i in range(len(top_k)):
            if rank < top_k[i]:
                acc_l2r[i] += 1
        if last_epoch:
            indices = indices.cpu().numpy()

    for idx in range(test_right.shape[0]):
        _, indices = torch.sort(distance[:, idx], descending=False)
        rank = (indices == idx).nonzero(as_tuple=False).squeeze().item()
        mean_r2l += (rank + 1)
        mrr_r2l += 1.0 / (rank + 1)
        for i in range(len(top_k)):
            if rank < top_k[i]:
                acc_r2l[i] += 1
    mean_l2r /= test_left.size(0)
    mean_r2l /= test_right.size(0)
    mrr_l2r /= test_left.size(0)
    mrr_r2l /= test_right.size(0)
    for i in range(len(top_k)):
        acc_l2r[i] = round(acc_l2r[i] / test_left.size(0), 4)
        acc_r2l[i] = round(acc_r2l[i] / test_right.size(0), 4)
    Loss_out = ""

    print(
        f"l2r: acc of top {top_k} = {acc_l2r}, mr = {mean_l2r:.3f}, mrr = {mrr_l2r:.3f}{Loss_out}")
    print(
        f"r2l: acc of top {top_k} = {acc_r2l}, mr = {mean_r2l:.3f}, mrr = {mrr_r2l:.3f}{Loss_out}")

    print(f"  l2r: acc of top {top_k} = {acc_l2r}, mr = {mean_l2r:.3f}, mrr = {mrr_l2r:.3f}{Loss_out}")
    print(f"  r2l: acc of top {top_k} = {acc_r2l}, mr = {mean_r2l:.3f}, mrr = {mrr_r2l:.3f}{Loss_out}")
    print('没有匹配上中,它的第一位有{}个没有图片,{}'.format(not_img,not_i))

    return acc_l2r[0]


with open('../data.pkl','rb')as f:
    data_emd = pickle.load(f)


with open('../test.pkl','rb')as f:
    test_data = pickle.load(f)

test(data_emd=data_emd, test_left=test_data['test_eft'], test_right=test_data['test_right'],save_name="")