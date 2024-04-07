import torch
import random
import json
import numpy as np
import pdb
import torch.distributed as dist
import os
import os.path as osp
from collections import Counter
import pickle
import torch.nn.functional as F
from transformers import BertTokenizer
import torch.distributed
from tqdm import tqdm
import scipy.sparse as sp


class EADataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def read_raw_data(file_dir, lang=[1, 2]):
    """
    Read DBP15k/DWY15k dataset.
    Parameters
    ----------
    file_dir: root of the dataset.
    Returns
    -------
    ent2id_dict : A dict mapping from entity name to ids
    ills: inter-lingual links (specified by ids)
    triples: a list of tuples (ent_id_1, relation_id, ent_id_2)
    r_hs: a dictionary containing mappings of relations to a list of entities that are head entities of the relation
    r_ts: a dictionary containing mappings of relations to a list of entities that are tail entities of the relation
    ids: all ids as a list
    """
    print('loading raw data...')

    def read_file(file_paths):
        tups = []
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    tups.append(tuple([int(x) for x in params]))
        return tups

    def read_dict(file_paths):
        ent2id_dict = {}
        ids = []
        for file_path in file_paths:
            id = set()
            with open(file_path, "r", encoding="utf-8") as fr:
                for line in fr:
                    params = line.strip("\n").split("\t")
                    ent2id_dict[params[1]] = int(params[0])
                    id.add(int(params[0]))
            ids.append(id)
        return ent2id_dict, ids

    ent2id_dict, ids = read_dict([file_dir + "/ent_ids_" + str(i) for i in lang])
    ills = read_file([file_dir + "/ill_ent_ids"])
    triples = read_file([file_dir + "/triples_" + str(i) for i in lang])
    r_hs, r_ts = {}, {}
    for (h, r, t) in triples:
        if r not in r_hs:
            r_hs[r] = set()
        if r not in r_ts:
            r_ts[r] = set()
        r_hs[r].add(h)
        r_ts[r].add(t)
    assert len(r_hs) == len(r_ts)
    return ent2id_dict, ills, triples, r_hs, r_ts, ids


def load_img(e_num, path):
    img_dict = pickle.load(open(path, "rb"))
    # init unknown img vector with mean and std deviation of the known's
    imgs_np = np.array(list(img_dict.values()))
    mean = np.mean(imgs_np, axis=0)
    std = np.std(imgs_np, axis=0)
    # img_embd = np.array([np.zeros_like(img_dict[0]) for i in range(e_num)]) # no image
    # img_embd = np.array([img_dict[i] if i in img_dict else np.zeros_like(img_dict[0]) for i in range(e_num)])

    img_embd = np.array(
        [img_dict[i] if i in img_dict else np.random.normal(mean, std, mean.shape[0]) for i in range(e_num)])
    img_mask = np.array(
        [1 if i in img_dict else 0 for i in range(e_num)])
    # print(img_id)
    print('{}% entities have images'.format(100 * len(img_dict) / e_num))
    return img_embd, img_mask


def get_ids(fn):
    ids = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            ids.append(int(th[0]))
    return ids


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.FloatTensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def get_adjr(ent_size, triples, norm=False):
    print('getting a sparse tensor r_adj...')
    M = {}
    for tri in triples:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 0
        M[(tri[0], tri[2])] += 1
    ind, val = [], []
    for (fir, sec) in M:
        ind.append((fir, sec))
        ind.append((sec, fir))
        val.append(M[(fir, sec)])
        val.append(M[(fir, sec)])

    for i in range(ent_size):
        ind.append((i, i))
        val.append(1)

    if norm:
        ind = np.array(ind, dtype=np.int32)
        val = np.array(val, dtype=np.float32)
        adj = sp.coo_matrix((val, (ind[:, 0], ind[:, 1])), shape=(ent_size, ent_size), dtype=np.float32)
        # 1. normalize_adj
        # 2. Convert a scipy sparse matrix to a torch sparse tensor
        # pdb.set_trace()
        return sparse_mx_to_torch_sparse_tensor(normalize_adj(adj))
    else:
        M = torch.sparse_coo_tensor(torch.LongTensor(ind).t(), torch.FloatTensor(val), torch.Size([ent_size, ent_size]))
        return M


def load_data(args):
    lang_list = [1, 2]
    ent2id_dict, ills, triples, r_hs, r_ts, ids = read_raw_data(args.datadir, lang_list)
    e1 = os.path.join(args.datadir, 'ent_ids_1')
    e2 = os.path.join(args.datadir, 'ent_ids_2')
    left_ents = get_ids(e1)
    right_ents = get_ids(e2)
    ENT_NUM = len(ent2id_dict)
    REL_NUM = len(r_hs)

    np.random.shuffle(ills)
    img_vec_path = osp.join(args.datadir, 'FBYG15K_id_img_feature_dict.pkl')
    img_features,img_mask = load_img(ENT_NUM, img_vec_path)
    numb = 0
    nv =0
    with open('mask.pkl','wb')as f:
        pickle.dump(img_mask, f)


    train_ill = np.array(ills[:int(len(ills) // 1 * args.trainrate + args.devrate)], dtype=np.int32) # todo
    dev_ill = np.array(ills[int(len(ills) // 1 * args.trainrate):int(len(ills) // 1 * (args.trainrate + args.devrate))],
                       dtype=np.int32)
    #dev_ill = np.array(ills[:int(len(ills) // 1 * args.trainrate + args.devrate)], dtype=np.int32)
    test_ill = np.array(ills[int(len(ills) // 1 * (args.trainrate + args.devrate)):], dtype=np.int32)
    test_left = torch.LongTensor(test_ill[:, 0].squeeze())
    test_right = torch.LongTensor(test_ill[:, 1].squeeze())



    left_non_train = list(set(left_ents) - set(train_ill[:, 0].tolist() + dev_ill[:, 0].tolist()))

    right_non_train = list(set(right_ents) - set(train_ill[:, 1].tolist() + dev_ill[:, 1].tolist()))

    for it in left_non_train:
        if img_mask[it]==0:
            numb+=1

    for it in right_non_train:
        if img_mask[it]==0:
            numb+=1

    for it in test_left :
        if img_mask[it]==0:
            nv+=1

    for it in test_right:
        if img_mask[it]==0:
            nv+=1

    input_idx = torch.LongTensor(np.arange(ENT_NUM))
    adj = get_adjr(ENT_NUM, triples, norm=True)
    train_ill = EADataset(train_ill)
    dev_ill = EADataset(dev_ill)
    test_ill = EADataset(test_ill)
    return {
        'ent2id_dict': ent2id_dict,
        'triples': triples,
        'ent_num': ENT_NUM,
        'rel_num': REL_NUM,
        'images_list': img_features,
        'input_idx': input_idx,
        'img_mask': img_mask,
        'adj': adj
    }, {"left": left_non_train, "right": right_non_train}, train_ill, dev_ill, test_ill
