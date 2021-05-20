import tqdm
import torch
import numpy as np
from torch import nn
from torch import optim
from numpy import random
from models import KBCModel, SimCLR
from regularizers import Regularizer


class KBCOptimizer(object):
    def __init__(
            self, model_name, model: KBCModel, regularizer: list, optimizer: optim.Optimizer, batch_size: int = 256,
            rank: int = 2000, temperature: float = 1.0, hidden_size: int = 2048,
            a_h: float = 0, a_t: float = 0, a_hr: float = 0, a_tr: float = 0, verbose: bool = True
    ):
        self.model_name = model_name
        self.model = model
        self.regularizer = regularizer[0]
        self.regularizer2 = regularizer[1]
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.temperature = temperature
        self.rank = rank
        self.hidden_size = hidden_size
        self.a_h = a_h
        self.a_t = a_t
        self.a_hr = a_hr
        self.a_tr = a_tr

    def get_pos_t(self, actual_examples, dic_t=None):
        p_t = []
        for i in actual_examples:
            t_sample = dic_t[(i[0].item(), i[1].item())]
            random.shuffle(t_sample)
            p_t.append(t_sample[0])
        return p_t

    def get_pos_h(self, actual_examples, dic_h=None):
        p_h = []
        for i in actual_examples:
            h_sample = dic_h[(i[2].item(), i[1].item())]
            random.shuffle(h_sample)
            p_h.append(h_sample[0])
        return p_h

    def get_pos_tr(self, actual_examples, dic_tr=None):
        p_tr = []
        for i in actual_examples:
            tr_sample = dic_tr[i[0].item()]
            random.shuffle(tr_sample)
            p_tr.append(tr_sample[0])
        return p_tr

    def get_pos_hr(self, actual_examples, dic_hr=None):
        p_hr = []
        for i in actual_examples:
            hr_sample = dic_hr[i[2].item()]
            random.shuffle(hr_sample)
            p_hr.append(hr_sample[0])
        return p_hr

    def epoch(self, examples: torch.LongTensor, e=0, weight=None,  dic_hr=None, dic_tr=None, dic_t=None, dic_h=None):
        self.model.train()
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean', weight=weight)
        p_h = None
        p_t = None
        p_hr = None
        p_tr = None
        if self.a_h != 0:
            p_h = self.get_pos_h(actual_examples, dic_h=dic_h)
            p_h = torch.tensor(p_h)
        if self.a_t != 0:
            p_t = self.get_pos_t(actual_examples, dic_h=dic_h)
            p_t = torch.tensor(p_t)
        if self.a_h != 0:
            p_hr = self.get_pos_h(actual_examples, dic_h=dic_hr)
            p_hr = torch.tensor(p_hr)
        if self.a_t != 0:
            p_tr = self.get_pos_t(actual_examples, dic_h=dic_tr)
            p_tr = torch.tensor(p_tr)

        if self.model_name == 'ComplEx':
            net = SimCLR(self.rank * 2, self.temperature, self.hidden_size).cuda()
        else:
            net = SimCLR(self.rank, self.temperature, self.hidden_size).cuda()
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                              b_begin:b_begin + self.batch_size
                              ].cuda()
                pos_h_loss = 0
                pos_t_loss = 0
                pos_hr_loss = 0
                pos_tr_loss = 0
                if p_h is not None:
                    p_h_pos = p_h[b_begin:b_begin + self.batch_size].cuda()
                    p_h_e = self.model.embeddings[0](p_h_pos)
                    self_h_e = self.model.embeddings[0](input_batch[:, 0])
                    pos_h_loss = net(self_h_e, p_h_e)
                if p_t is not None:
                    p_t_pos = p_t[b_begin:b_begin + self.batch_size].cuda()
                    self_t_e = self.model.embeddings[0](input_batch[:, 2])
                    p_t_e = self.model.embeddings[0](p_t_pos)
                    pos_t_loss = net(self_t_e, p_t_e)
                if p_hr is not None:
                    p_tail = p_hr[b_begin:b_begin + self.batch_size].cuda()
                    self_hr, pos_t_predictions, pos_t_factor = self.model.forward(input_batch, p_tail=p_tail, mod=0)
                    pos_hr_loss = net(self_hr, pos_t_predictions)
                if p_tr is not None:
                    p_head = p_tr[b_begin:b_begin + self.batch_size].cuda()
                    self_tr, pos_h_predictions, pos_h_factor = self.model.forward(input_batch, p_head=p_head, mod=0)
                    pos_tr_loss = net(self_tr, pos_h_predictions)

                predictions, factors = self.model.forward(input_batch, mod=1)
                truth = input_batch[:, 2]
                l_fit = loss(predictions, truth)
                l_reg = self.regularizer.forward(factors)
                l = l_fit + l_reg + self.a_hr * pos_hr_loss + self.a_tr * pos_tr_loss + self.a_t * pos_t_loss + self.a_h * pos_h_loss
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.1f}')

        return l






