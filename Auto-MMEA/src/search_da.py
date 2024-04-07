import math

import numpy as np
import torch
import src.scheduler as sc
import copy
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
from IPython import embed
from src.utils import count_parameters, save, save_pickle
import torch.nn.functional as F

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    distance = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return torch.clamp(distance, 0.0, np.inf)

def test(model, test_left, test_right, last_epoch=False, save_name="", loss=None):
    with torch.no_grad():
        _,_,_,_,_, emd_left = model(test_left)
        _,_,_,_,_, emd_right = model(test_right)



    # pdb.set_trace()
    top_k = [1, 10, 50]
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

def icl_loss(zis,output_zis,output_zjs):
    n_view = 2
    temperature = 0.1
    LARGE_NUM = 1e9
    num_classes = len(zis) * n_view
    labels = F.one_hot(torch.arange(start=0, end=len(zis), dtype=torch.int64),
                       num_classes=num_classes).float()
    labels = labels.cuda()

    masks = F.one_hot(torch.arange(start=0, end=len(zis), dtype=torch.int64),
                      num_classes=len(zis))
    masks = masks.cuda().float()
    logits_aa = torch.matmul(output_zis, torch.transpose(output_zis, 0, 1)) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM

    logits_bb = torch.matmul(output_zjs, torch.transpose(output_zjs, 0, 1)) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(output_zis, torch.transpose(output_zjs, 0, 1)) / temperature
    logits_ba = torch.matmul(output_zjs, torch.transpose(output_zis, 0, 1)) / temperature
    logits_a = torch.cat([logits_ab, logits_bb], dim=1)
    logits_b = torch.cat([logits_ba, logits_aa], dim=1)
    loss_a = softXEnt(labels, logits_a).cuda()
    loss_b = softXEnt(labels, logits_b).cuda()
    loss = loss_a + loss_b
    return loss

def train_track(model, architect,
                          criterion, optimizer, scheduler, dataloaders_train,dataloaders_dev,dataloaders_test,dev,
                          dataset_sizes, device, num_epochs, args,
                          f1_type='weighted', init_f1=0.0, th_fscore=0.3,
                          status='search',):
    best_genotype = None
    best_f1 = init_f1
    best_epoch = 0

    best_test_genotype = None
    best_test_f1 = init_f1
    best_test_epoch = 0

    failsafe = True
    cont_overloop = 0
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
                    data_loaders=dataloaders_train
                    if not isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                        scheduler.step()
                    model.train()  # Set model to training mode
                    list_preds = []
                    list_label = []
                elif phase == 'dev':
                    if status == 'eval':
                        if not isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                            scheduler.step()
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


                with tqdm(data_loaders) as t:
                    # Iterate over data.
                    for data in data_loaders:

                        # get the inputs
                        zis = data[:, 0]
                        zjs = data[:,1]



                        if status == 'search' and (phase == 'dev' or phase == 'test'):
                            architect.step(zis,zjs)  # 从这进入

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train') :#or (phase == 'dev' and status == 'eval')):
                            s_zis,att_zis,img_zis,rel_zis,out_hid_zis = model(zis)
                            s_zjs,att_zjs,img_zjs,rel_zjs,out_hid_zjs = model(zjs)

                            loss_2 = icl_loss(zis,s_zis,s_zjs)
                            loss_3 = icl_loss(zis,att_zis,att_zjs)
                            loss_4 = icl_loss(zis,img_zis,img_zjs)
                            loss_5 = icl_loss(zis,rel_zis,rel_zjs)
                            loss_6 = icl_loss(zis, out_hid_zis,out_hid_zjs)
                            loss = 0.1*loss_2+0.1*loss_3+0.1*loss_4+loss_5+loss_6*0.5

                            # backward + optimize only if in training phase
                            if phase == 'train' or (phase == 'dev' and status == 'eval'):
                                if isinstance(scheduler, sc.LRCosineAnnealingScheduler):
                                    scheduler.step()
                                    scheduler.update_optimizer(optimizer)
                                loss.backward(retain_graph=True)
                                optimizer.step()

                            # statistics
                        running_loss += loss.item() * len(zis)
                        if math.isnan(loss):
                            print("loss is NaN")
                        postfix_str = 'batch_loss: {:.03f}'.format(loss.item())
                        t.set_postfix_str(postfix_str)
                        t.update()

                epoch_loss = running_loss / dataset_sizes

                print('{} Loss: {:.4f}'.format(
                    phase, epoch_loss))



                num_params = 0
                num_params = count_parameters(model.encoding_net)
                print("Fusion Model Params: {}".format(num_params))

                genotype = model.genotype()  # 剔除多余杂边


                if phase == 'train' and status == 'search':
                    test_left = []
                    test_right = []

                    test_left = torch.LongTensor(dev[:, 0].squeeze()).cuda()
                    test_right = torch.LongTensor(dev[:, 1].squeeze()).cuda()

                    # TODO batchsize
                    epoch_hit1 = test(model=model,test_left=test_left, test_right=test_right, save_name="")
                    print('{} hits@1: {:.4f}'.format(
                        phase, epoch_hit1))


                if phase == 'train' and epoch_loss != epoch_loss:
                    print("Nan loss during training, escaping")
                    model.eval()
                    return best_f1

                if phase == 'dev' and status == 'search':
                    test_left = []
                    test_right = []

                    # test_left = torch.LongTensor(data_loaders[:, 0].squeeze()).cuda()
                    # test_right = torch.LongTensor(data_loaders[:, 1].squeeze()).cuda()

                    test_left = torch.LongTensor(dataloaders_test[:, 0].squeeze()).cuda()
                    test_right = torch.LongTensor(dataloaders_test[:, 1].squeeze()).cuda()

                    # TODO batchsize
                    epoch_hit1 = test(model=model, test_left=test_left, test_right=test_right, save_name="")
                    print('{} hits@1: {:.4f}'.format(
                        phase, epoch_hit1))
                    if epoch_hit1 > best_f1:
                        best_f1 = epoch_hit1
                        best_genotype = copy.deepcopy(genotype)
                        best_epoch = epoch
                        # best_model_sd = copy.deepcopy(model.state_dict())


                        save(model, os.path.join(args.save, 'best', 'best_model.pt'))

                        best_genotype_path = os.path.join(args.save, 'best', 'best_genotype.pkl')
                        save_pickle(best_genotype, best_genotype_path)

                if phase == 'test':
                    if epoch_hit1 > best_test_f1:
                        best_test_f1 = epoch_hit1
                        best_test_genotype = copy.deepcopy(genotype)
                        best_test_epoch = epoch


                        save(model, os.path.join(args.save, 'best', 'best_test_model.pt'))

                        best_test_genotype_path = os.path.join(args.save, 'best', 'best_test_genotype.pkl')
                        save_pickle(best_test_genotype, best_test_genotype_path)

            file_name = "epoch_{}".format(epoch)
            file_name = os.path.join(args.save, "architectures", file_name)
            #plotter.plot(genotype, file_name, task='mmimdb')  # 画图

            print("Current best dev {} F1: {}, at training epoch: {}".format(f1_type, best_f1, best_epoch))
            print(
                "Current best test {} F1: {}, at training epoch: {}".format(f1_type, best_test_f1, best_test_epoch))

        if best_f1 != best_f1 and num_epochs == 1 and cont_overloop < 1:
            failsafe = True
        else:
            failsafe = False

        cont_overloop += 1

    if best_f1 != best_f1:
        best_f1 = 0.0

    if status == 'search':
        return best_f1, best_genotype
    else:
        return best_test_f1, best_test_genotype
