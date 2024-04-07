
import torch
import argparse
import time
from src.data import load_data
from src.search import EA_Searcher
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
import shutil

import glob
import logging
import sys
import os



def data_init(args):
    KGs, non_train, train_set, dev_set,test_set  = load_data(args)

    return  KGs, non_train, train_set, dev_set,test_set

class Collator_base(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, batch):
        # pdb.set_trace()

        return np.array(batch)


def parse_args():

    parser = argparse.ArgumentParser(description='BM-NAS Configuration')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # experiment directory
    parser.add_argument('--save', type=str, default='EXP', help='where to save the experiment')

    # dataset and data parallel
    parser.add_argument('--datadir', type=str, help='data directory',
                        default='data/FBYA15K/')
    parser.add_argument('--small_dataset', action='store_true', default=False, help='use mini dataset for debugging')
    parser.add_argument('--workers', type=int, help='dataloader CPUs', default=32)
    # parser.add_argument('--use_dataparallel', help='use several GPUs', action='store_true', default=False)
    parser.add_argument('--parallel', help='use several GPUs', action='store_true', default=False)
    # basic learning settings
    parser.add_argument('--batchsize', type=int, help='batch size', default=4000)
    parser.add_argument('--trainrate', type=float, help='train rate', default=0.18)
    parser.add_argument('--devrate', type=float, help='dev rate', default=0.02)
    parser.add_argument('--epochs', type=int, help='training epochs', default=1000)
    parser.add_argument("--drpt", action="store", default=0.1, dest="drpt", type=float, help="dropout")



    # for cells and steps and inner representation size
    parser.add_argument('--C', type=int, help='channels for conv layer', default=192)
    parser.add_argument('--L', type=int, help='length after conv and pool', default=16)
    parser.add_argument('--multiplier', type=int, help='cell output concat', default=2)
    parser.add_argument('--steps', type=int, help='cell steps', default=2)
    parser.add_argument('--node_steps', type=int, help='inner node steps', default=1)
    parser.add_argument('--node_multiplier', type=int, help='inner node output concat', default=1)
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument("--only_test", default=0, type=int, choices=[0, 1])

    # number of classes
    parser.add_argument('--num_outputs', type=int, help='output dimension', default=23)
    parser.add_argument('--f1_type', type=str, help="use 'weighted' or 'macro' F1 Score", default='weighted')

    # archtecture optimizer
    parser.add_argument('--arch_learning_rate', type=float, default=3e-3, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

    # network optimizer and scheduler
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--eta_max', type=float, help='max learning rate', default=0.0005)
    parser.add_argument('--eta_min', type=float, help='min laerning rate', default=0.0001)
    parser.add_argument('--Ti', type=int, help='for cosine annealing scheduler, epochs Ti', default=1)
    parser.add_argument('--Tm', type=int, help='for cosine annealing scheduler, epochs multiplier Tm', default=2)
    parser.add_argument("--csls", action="store_true", default=False, help="use CSLS for inference")
    parser.add_argument("--csls_k", type=int, default=3, help="top k for csls")
    parser.add_argument("--il", action="store_true", default=True, help="Iterative learning?")
    parser.add_argument("--semi_learn_step", type=int, default=5, help="If IL, what's the update step?")
    parser.add_argument("--il_start", type=int, default=500, help="If Il, when to start?")

    parser.add_argument("--use_project_head", action="store_true", default=False, help="use projection head")
    parser.add_argument("--zoom", type=float, default=0.1, help="narrow the range of losses")
    parser.add_argument("--reduction", type=str, default="mean", help="[sum|mean]", choices=["sum", "mean"])

    # --------- MEAformer -----------
    parser.add_argument("--hidden_size", type=int, default=300, help="the hidden size of MEAformer")
    parser.add_argument("--intermediate_size", type=int, default=400, help="the hidden size of MEAformer")
    parser.add_argument("--num_attention_heads", type=int, default=5,
                        help="the number of attention_heads of MEAformer")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="the number of hidden_layers of MEAformer")
    parser.add_argument("--position_embedding_type", default="absolute", type=str)
    parser.add_argument("--use_intermediate", type=int, default=1, help="whether to use_intermediate")
    parser.add_argument("--replay", type=int, default=0, help="whether to use replay strategy")
    parser.add_argument("--neg_cross_kg", type=int, default=0,
                        help="whether to force the negative samples in the opposite KG")


    return parser.parse_args()


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

    os.mkdir(os.path.join(path, 'architectures'))
    os.mkdir(os.path.join(path, 'best'))

def _dataloader(args,train_set, batch_size, collator):
        train_dataloader = DataLoader(
            train_set,
            num_workers=args.workers,
            persistent_workers=True,  # True
            shuffle=(args.only_test == 0),
            # drop_last=(self.args.only_test == 0),
            drop_last=False,
            batch_size=batch_size,
            collate_fn=collator
        )
        return train_dataloader

def dataloader_init(args,train_set=None, dev_set=None):
        bs = args.batchsize
        collator = Collator_base(args)
        args.workers = min([os.cpu_count(), args.batchsize])
        if train_set is not None:
            train_dataloader = _dataloader(args,train_set, bs, collator)

        if dev_set is not None:
            dev_dataloader = _dataloader(args,dev_set, bs, collator)
        return train_dataloader,dev_dataloader

def main():
    args = parse_args()
    args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
    args.save = os.path.join('final_exp/ea_darts', args.save)
    create_exp_dir(args.save, scripts_to_save=None)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)




    KGs, non_train, train_set, dev_set, test_set = data_init(args)
    tr, dev = dataloader_init(args, train_set, dev_set)
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")
    EA_search = EA_Searcher(args, device, KGs, non_train, tr, train_set, dev, test_set, dev_set)

    best_acc, best_genotype = EA_search.search()

if __name__ == "__main__":

    # sweep_configuration = {
    #     'method': 'random',
    #     'name': 'sweep',
    #     'metric': {'goal': 'maximize', 'name': 'test_acc'},
    #     'parameters':
    #         {
    #             'a1': {'max': 10.0, 'min': 0.0001},
    #             'a2': {'max': 10.0, 'min': 0.0001},
    #             'a3': {'max': 10.0, 'min': 0.0001},
    #             'a4': {'max': 10.0, 'min': 0.0001},
    #         }
    # }
    #
    # sweep_id = wandb.sweep(
    #     sweep=sweep_configuration,
    #     project='my-first-sweep'
    # )
    #
    # wandb.agent(sweep_id, function=main, count=4)
    main()



























