import pickle
import random

import math

from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
from torch.cuda.amp import GradScaler

import src.scheduler as sc
import copy
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
from IPython import embed

from src.common import MLP
from src.model import dispose_emd_list
from src.utils import count_parameters, save, save_pickle
import torch.nn.functional as F
import traceback
import logging

import torch.optim as op


def caculate_loss(model,a1,a2,a3,a4,b1,b2,b3,b4):
    loss_1 = cmd_loss(a1,a2)
    loss_2 =cmd_loss(a1,a3)
    loss_3 = cmd_loss(a1,a4)
    loss_4 = cmd_loss(a2,a3)
    loss_5 = cmd_loss(a2,a4)
    loss_6 = cmd_loss(a3,a4)
    loss_a = model.similar_loss_layer([loss_1,loss_2,loss_3,loss_4,loss_5,loss_6])

    loss_1 = cmd_loss(b1, b2)
    loss_2 = cmd_loss(b1,b3)
    loss_3 = cmd_loss(b1, b4)
    loss_4 = cmd_loss(b2, b3)
    loss_5 = cmd_loss(b2, b4)
    loss_6 = cmd_loss(b3, b4)
    loss_b = model.similar_loss_lay2([loss_1, loss_2, loss_3, loss_4, loss_5, loss_6])

    return 0.5*loss_a+0.5*loss_b



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


def test(epoch,model, test_left, test_right, logger,last_epoch=False, save_name="", loss=None):
    model.eval()
    true_list=[]
    with torch.no_grad():
        model.eval()
        _, _, _, _, emd,_= model(is_test=False)#todo

    emd_left = emd[test_left]
    emd_right = emd[test_right]




    top_k = [1,5, 10, 50]
    acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
    acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
    test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.
    distance = pairwise_distances(emd_left, emd_right)



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

        #case study
        if rank ==0 and epoch==799:
            true_list.append(idx)

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

    logger.info(f"{epoch}  l2r: acc of top {top_k} = {acc_l2r}, mr = {mean_l2r:.3f}, mrr = {mrr_l2r:.3f}{Loss_out}")
    logger.info(f"{epoch}  r2l: acc of top {top_k} = {acc_r2l}, mr = {mean_r2l:.3f}, mrr = {mrr_r2l:.3f}{Loss_out}")

    if epoch==799:
        data={}
        data['left_emd']=emd_left
        data['right_emd']=emd_right
        for it in range(len(true_list)):
            true_list[it]=(test_left[true_list[it]].item(),test_right[true_list[it]].item())
        data['true_list']=true_list
        file_path = 'data_yago.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)




    return acc_l2r[0]


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

def iml_loss(zis, output_zis, output_zjs):
    tar=len(zis)
    target = torch.ones(tar).cuda()
    loss = F.cosine_embedding_loss(output_zjs,output_zis,target)
    return loss


def cmd_loss(p_ab,q_ab):
    margin = 0.5
    beta = 10

    euclidean_distance = torch.sqrt(torch.sum(torch.pow(p_ab - q_ab, 2), dim=1))

    # 计算CMD损失
    contrastive_loss = torch.mean((1 - euclidean_distance) ** 2)
    discriminative_loss = torch.mean(torch.relu(margin - euclidean_distance))
    loss_a = contrastive_loss + beta * discriminative_loss
    return  loss_a
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


def semi_supervised_learning(graph_emb,img_emb,rel_emb,att_emb,out_hid_emb,non_train):
    with torch.no_grad():
        gph_emb, img_emb, rel_emb, att_emb, joint_emb = graph_emb,img_emb,rel_emb,att_emb,out_hid_emb

        final_emb = F.normalize(joint_emb)

    left_non_train = non_train['left']

    right_non_train = non_train['right']

    distance_list = []
    for i in np.arange(0, len(left_non_train), 1000):
        d = pairwise_distances(final_emb[left_non_train[i:i + 1000]], final_emb[right_non_train])
        distance_list.append(d)
    distance = torch.cat(distance_list, dim=0)
    preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
    preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
    del distance_list, distance, final_emb
    return preds_l, preds_r

def distance(emb,length):
    dist = 0
    for it in range(length):
        dist += emb[it][it]
    return dist


def dynamic_loss_weight(loss_now,loss_t_1,loss_t_2):
    """

    :param loss_t_1: 每个task上一轮的loss列表，并且为标量
    :param loss_t_2:
    :return:
    """
    T = 10
    assert len(loss_t_1) == len(loss_t_2)
    task_n = len(loss_t_1)
    w = [l_1 / l_2 for l_1, l_2 in zip(loss_t_1, loss_t_2)]

    lamb = [math.exp(v/ T) for v in w]

    lamb_sum = sum(lamb)

    a = [task_n * l / lamb_sum for l in lamb]
    loss_all = sum(a[it] * loss_now[it] for it in range(len(loss_now)))
    return  loss_all

def train_track(model, architect,
                non_train, optimizer,dataloaders_train, dataloaders_dev, dataloaders_test, dev, train,
                dataset_sizes, device, num_epochs, args,
                f1_type='weighted', init_f1=0.0, th_fscore=0.3,
                status='search', ):
    best_genotype = None
    best_f1 = init_f1
    best_epoch = 0

    best_test_genotype = None
    best_test_f1 = init_f1
    best_test_epoch = 0

    new_links = []

    # 配置日志记录
    logging.basicConfig(filename='print_log_{}.log'.format(args.trainrate), level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 创建logger对象
    logger = logging.getLogger('print_logger')

    failsafe = True
    cont_overloop = 0
    loss_now = []
    loss_momnet = []
    loss_before_1 = []
    loss_before_2 = []

    loss_before_one = []
    loss_before_two = []
    logger.info(f"dataset:{args.datadir},train_rate:{args.trainrate}")
    print(f"dataset:{args.datadir},train_rate:{args.trainrate}")
    writer = SummaryWriter('./logs')




    while failsafe:
        for epoch in range(num_epochs):


            print('Epoch: {}'.format(epoch))
            print("EXP: {}".format(args.save))


            phases = []
            if status == 'search':
                phases = ['train', 'dev']
            else:
                # while evaluating, add dev set to train also
                phases = ['train', 'dev', 'test']

            # Each epoch has a training and validation phase
            for phase in phases:
                if phase == 'train':
                    data_loaders = dataloaders_train
                    model.train()  # Set model to training mode
                    list_preds = []
                    list_label = []
                elif phase == 'dev':
                    model.train()
                    data_loaders = dataloaders_dev
                    list_preds = []
                    list_label = []
                else:
                    model.eval()  # Set model to evaluate mode
                    list_preds = []
                    list_label = []

                running_loss = 0.0
                running_f1 = init_f1

                # zero the parameter gradients
                optimizer.zero_grad()

                graph_emb, att_emb, img_emb, rel_emb, out_hid_emb ,emd_list = model()



                if True:
                    if status == 'search' and phase == 'train':

                        graph_emb = F.normalize(graph_emb, dim=1)
                        att_emb = F.normalize(att_emb, dim=1)
                        img_emb = F.normalize(img_emb, dim=1)
                        rel_emb = F.normalize(rel_emb, dim=1)
                        out_hid_emb = F.normalize(out_hid_emb, dim=1)



                        if True:
                            # Iterate over data.
                            dynasty = 0
                            ave_gcn = 0
                            ave_att = 0
                            ave_rel = 0
                            ave_img = 0

                            ave_gcn_o = 0
                            ave_att_o = 0
                            ave_rel_o = 0
                            ave_img_o = 0
                            for si in np.arange(0, data_loaders.shape[0], args.batchsize):
                                # get the inputs
                                dynasty+=1
                                zis = data_loaders[si:si + args.batchsize][:, 0]
                                zjs = data_loaders[si:si + args.batchsize][:, 1]




                                with torch.set_grad_enabled(
                                        phase == 'train'):

                                    s_zis, att_zis, img_zis, rel_zis, out_hid_zis = graph_emb[zis], att_emb[
                                        zis], \
                                        img_emb[zis], rel_emb[zis], out_hid_emb[zis]
                                    s_zjs, att_zjs, img_zjs, rel_zjs, out_hid_zjs = graph_emb[zjs], att_emb[
                                        zjs], \
                                        img_emb[zis], rel_emb[zjs], out_hid_emb[zjs]



                                    loss_joi = icl_loss(zis,out_hid_zis,out_hid_zjs)




                                    loss_gcn = icl_loss(zis, s_zis, s_zjs)
                                    loss_att = icl_loss(zis, att_zis, att_zjs)
                                    loss_img = icl_loss(zis, img_zis, img_zjs)
                                    loss_rel = icl_loss(zis, rel_zis, rel_zjs)
                                    loss_now=[loss_gcn,loss_att,loss_img,loss_rel]
                                    ave_gcn += loss_gcn.item()
                                    ave_att += loss_att.item()
                                    ave_rel += loss_rel.item()
                                    ave_img += loss_img.item()



                                    loss_gcn_o = ial_loss(s_zis, s_zjs, out_hid_zis, out_hid_zjs)
                                    loss_att_o = ial_loss(att_zis, att_zjs, out_hid_zis, out_hid_zjs)
                                    loss_img_o = ial_loss(img_zis, img_zjs, out_hid_zis, out_hid_zjs)
                                    loss_rel_o = ial_loss(rel_zis, rel_zjs, out_hid_zis, out_hid_zjs)
                                    loss_momnet = [loss_gcn_o, loss_att_o, loss_img_o, loss_rel_o]

                                    ave_gcn_o += loss_gcn_o.item()
                                    ave_att_o += loss_att_o.item()
                                    ave_rel_o += loss_rel_o.item()
                                    ave_img_o += loss_img_o.item()

                                    loss_emd = dispose_emd_list(emd_list, zis, zjs)


                                    if epoch > 1 :
                                        loss_in = dynamic_loss_weight(loss_now,loss_before_1,loss_before_2)
                                        loss_out = dynamic_loss_weight(loss_momnet,loss_before_one,loss_before_two)


                                    else:
                                        loss_in = loss_gcn+loss_att+loss_img+loss_rel
                                        loss_out = loss_gcn_o+loss_att_o+loss_img_o+loss_rel_o

                                    #loss = grad_weights[0]*loss_joi*3 + grad_weights[2]*loss_joi_3*3 + loss_in + loss_out + grad_weights[1]*loss_joi_2*3 #todo-
                                    if epoch >-1 :
                                          loss = loss_joi + loss_in + loss_out +loss_emd
                                    else:
                                        loss = loss_joi




                                    # backward + optimize only if in training phase
                                    if phase == 'train' or (phase == 'dev' and status == 'eval'):
                                        loss.backward(retain_graph=True)


                            optimizer.step()
                            if epoch==0:
                                loss_before_2=[ave_gcn/dynasty,ave_att/dynasty,ave_img/dynasty,ave_rel/dynasty]
                                loss_before_two=[ave_gcn_o/dynasty,ave_att_o/dynasty,ave_img_o/dynasty,ave_rel_o/dynasty]

                            if epoch==1:
                                loss_before_1=[ave_gcn/dynasty,ave_att/dynasty,ave_img/dynasty,ave_rel/dynasty]
                                loss_before_one = [ave_gcn_o / dynasty, ave_att_o / dynasty,
                                                   ave_img_o/ dynasty, ave_rel_o / dynasty]

                            if epoch>1:
                                loss_before_2 = loss_before_1
                                loss_before_1 = [ave_gcn/dynasty,ave_att/dynasty,ave_img/dynasty,ave_rel/dynasty]

                                loss_before_two = loss_before_one
                                loss_before_one = [ave_gcn_o/ dynasty, ave_att_o/ dynasty,
                                                   ave_img_o/ dynasty, ave_rel_o/ dynasty]

                            del loss
                            del loss_in
                            del loss_out
                            del loss_joi






                    if status == 'search' and (phase == 'dev' or phase == 'test'):
                        zis = dev[:, 0]
                        zjs = dev[:, 1]




                        data_emd = [graph_emb, att_emb, img_emb, rel_emb, out_hid_emb,emd_list]
                        architect.step(data_emd,zis, zjs,epoch,writer)  # 从这进入

                    # forward
                    # track history if only in train


                torch.cuda.empty_cache()

                # num_params = 0
                # num_params = count_parameters(model.encoding_net)
                # print("Fusion Model Params: {}".format(num_params))



                if epoch >= args.il_start and (epoch + 1) % args.semi_learn_step == 0 and args.il:
                    # predict links
                    preds_l, preds_r = semi_supervised_learning(graph_emb,img_emb,rel_emb,att_emb,out_hid_emb,non_train)
                    left_non_train = non_train['left']
                    right_non_train = non_train['right']

                    # if args.csls is True:
                    #    distance = 1 - csls_sim(1 - distance, args.csls_k)
                    # print (len(preds_l), len(preds_r))

                    if (epoch + 1) % (args.semi_learn_step * 10) == args.semi_learn_step:
                        new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l)
                                     if preds_r[p] == i]  # Nearest neighbors
                    else:
                        new_links = [(left_non_train[i], right_non_train[p]) for i, p in enumerate(preds_l)
                                     if (preds_r[p] == i)
                                     and ((left_non_train[i], right_non_train[p]) in new_links)]
                    print("[epoch %d] #links in candidate set: %d" % (epoch, len(new_links)))

                if epoch >= args.il_start and (epoch + 1) % (args.semi_learn_step * 10) == 0 and len(
                        new_links) != 0 and args.il:
                    new_links_elect = new_links
                    print("\n#new_links_elect:", len(new_links_elect))

                    # if len(new_links) >= 5000: new_links = random.sample(new_links, 5000)
                    dataloaders_train = np.vstack((dataloaders_train, np.array(new_links_elect)))
                    print("train_ill.shape:", dataloaders_train.shape)

                    num_true = len([nl for nl in new_links_elect if nl in np.array(dataloaders_test)])
                    print("#true_links: %d" % num_true)
                    print("true link ratio: %.1f%%" % (100 * num_true / len(new_links_elect)))

                    # remove from left/right_non_train
                    for nl in new_links_elect:
                        left_non_train.remove(nl[0])
                        right_non_train.remove(nl[1])
                    print("#entity not in train set: %d (left) %d (right)" % (
                        len(left_non_train), len(right_non_train)))

                    new_links = []





                if phase == 'dev' and status == 'search' and (epoch + 1) %  50 == 0 :
                    test_left = []
                    test_right = []



                    test_left = torch.LongTensor(dataloaders_test[:, 0].squeeze()).cuda()

                    test_right = torch.LongTensor(dataloaders_test[:, 1].squeeze()).cuda()





                    # TODO batchsize
                    epoch_hit1 = test(epoch=epoch,model=model, test_left=test_left, test_right=test_right, logger=logger,save_name="")
                    print('{} hits@1: {:.4f}'.format(
                        phase, epoch_hit1))





                torch.cuda.empty_cache()



            torch.cuda.empty_cache()



        cont_overloop += 1
        writer.close()



        return  0,0


