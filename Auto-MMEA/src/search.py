import torch
import numpy as np
import torch.nn as nn
import torch.optim as op
import src.scheduler as sc
import src.aux_models as aux
import src.search_darts as S
import src.model as M


class EA_Searcher():
    def __init__(self, args, device, KGs, non_train, train_set, train, dev_set, test_set, dev):
        self.args = args
        self.device = device
        self.KGs = KGs
        self.non_train = non_train
        self.train_set = np.array(train, dtype=np.int32)
        self.dev_set = dev_set
        self.test_set = test_set
        self.dev = dev
        self.train = train

    def search(self):
        data_size = len(self.train_set)

        criterion = torch.nn.BCEWithLogitsLoss()
        model = M.Searchable_EA_Net(self.args, criterion, self.KGs)
        params = model.central_params()

        # optimizer and scheduler
        optimizer = op.AdamW(params, lr=self.args.eta_max)


        arch_optimizer = op.Adam(model.arch_parameters(),
                                 lr=self.args.arch_learning_rate, betas=(0.5, 0.999),
                                 weight_decay=self.args.arch_weight_decay)

        model.to(self.device)
        architect = M.Architect(model, self.args, criterion, arch_optimizer)





        best_f1, best_genotype = S.train_track(model, architect,
                                               self.non_train, optimizer, self.train_set, self.dev_set, self.test_set,
                                               self.dev, self.train,
                                               data_size,
                                               device=self.device,
                                               num_epochs=self.args.epochs,
                                               args=self.args,
                                               f1_type=self.args.f1_type,
                                               init_f1=0.0, th_fscore=0.3)

        return best_f1, best_genotype
