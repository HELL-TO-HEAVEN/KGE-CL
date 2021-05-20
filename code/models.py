from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
from torch.nn import functional as F, Parameter
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

import numpy as np

class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :return:
        """
        ranks = torch.ones(len(queries))
        # with tqdm(total=queries.shape[0], unit='ex') as bar:
        # bar.set_description(f'Evaluation')
        with torch.no_grad():
            b_begin = 0
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size]
                target_idxs = these_queries[:, 2].cpu().tolist()
                scores, _ = self.forward(these_queries)
                targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)

                for i, query in enumerate(these_queries):
                    filter_out = filters[(query[0].item(), query[1].item())]
                    filter_out += [queries[b_begin + i, 2].item()]  # Add the tail of this (b_begin + i) query
                    scores[i, torch.LongTensor(filter_out)] = -1e6
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()
                b_begin += batch_size
                # bar.update(batch_size)
        return ranks


class RESCAL(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(RESCAL, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], rank, sparse=True),
            nn.Embedding(sizes[1], rank * rank, sparse=True),
        ])

        nn.init.xavier_uniform_(tensor=self.embeddings[0].weight)
        nn.init.xavier_uniform_(tensor=self.embeddings[1].weight)

        self.lhs = self.embeddings[0]
        self.rel = self.embeddings[1]
        self.rhs = self.embeddings[0]

    def forward(self, x, p_tail=None, p_rel=None, p_head=None, p_h_r=None, mod=None):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1]).reshape(-1, self.rank, self.rank)
        rhs = self.rhs(x[:, 2])
        self_hr = (torch.bmm(lhs.unsqueeze(1), rel)).squeeze()
        if mod == 0:
            if p_tail is not None:
                h = self.lhs(p_tail[:, 0])
                r = self.rel(p_tail[:, 1]).reshape(-1, self.rank, self.rank)
                hr = torch.bmm(h.unsqueeze(1), r).squeeze()
                return self_hr, hr, hr
            if p_rel is not None:
                h = self.embeddings[0](p_rel[:, 0])
                t = self.embeddings[0](p_rel[:, 1])
                ht = h * t
                self_ht = rhs * lhs
                return self_ht, ht, ht
            if p_head is not None:
                t = self.embeddings[0](p_head[:, 0])
                r = self.embeddings[1](p_head[:, 1]).reshape(-1, self.rank, self.rank)
                tr = torch.bmm(t.unsqueeze(1), r).squeeze()
                self_tr = torch.bmm(rhs.unsqueeze(1), rel).squeeze()
                return self_tr, tr, tr
        else:
            return self_hr @ self.rhs.weight.t(), [(lhs, rel, rhs)]


class CP(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(CP, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True)
            for s in sizes[:3]
        ])

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        self.lhs = self.embeddings[0]
        self.rel = self.embeddings[1]
        self.rhs = self.embeddings[2]

    def forward(self, x, p_tail=None, p_rel=None, p_head=None, p_h_r=None, mod=None):
        lhs = self.lhs(x[:, 0])
        rel = self.rel(x[:, 1])
        rhs = self.rhs(x[:, 2])
        if mod == 0:
            if p_tail is not None:
                h = self.embeddings[0](p_tail[:, 0])
                r = self.embeddings[1](p_tail[:, 1])
                hr = h * r
                self_hr = lhs * rel
                return self_hr, hr, torch.sqrt(h ** 2 + r**2)
            if p_rel is not None:
                h = self.embeddings[0](p_rel[:, 0])
                t = self.embeddings[2](p_rel[:, 1])
                ht = h * t
                self_ht = rhs * lhs
                return self_ht, ht, torch.sqrt(h ** 2 + t ** 2)
            if p_head is not None:
                t = self.embeddings[2](p_head[:, 0])
                r = self.embeddings[1](p_head[:, 1])
                tr = t * r
                self_tr = rhs * rel
                return self_tr, tr, torch.sqrt(t ** 2 + r ** 2)
        else:
            return (lhs * rel) @ self.rhs.weight.t(), [(lhs, rel, rhs)], rhs


class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])

        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def forward(self, x, p_tail=None, p_rel=None, p_head=None, p_h_r=None, mod=None, r_weight=None):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        self_hr_R = lhs[0] * rel[0] - lhs[1] * rel[1]
        self_hr_I = lhs[0] * rel[1] + lhs[1] * rel[0]
        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        if mod == 0:
            if p_tail is not None:
                h = self.embeddings[0](p_tail[:, 0])
                r = self.embeddings[1](p_tail[:, 1])
                h = h[:, :self.rank], h[:, self.rank:]
                r = r[:, :self.rank], r[:, self.rank:]
                hr_R = h[0] * r[0] - h[1] * r[1]
                hr_I = h[0] * r[1] + h[1] * r[0]
                return torch.cat((self_hr_R, self_hr_I), 1), torch.cat((hr_R, hr_I), 1), torch.sqrt(hr_R ** 2 + hr_I ** 2)
            if p_head is not None:
                t = self.embeddings[0](p_head[:, 0])
                r = self.embeddings[1](p_head[:, 1])
                t = t[:, :self.rank], t[:, self.rank:]
                r = r[:, :self.rank], r[:, self.rank:]
                tr_R = t[0] * r[0] + t[1] * r[1]
                tr_I = t[0] * r[1] - t[1] * r[0]
                self_tr_R = rhs[0] * rel[0] + rhs[1] * rel[1]
                self_tr_I = rhs[0] * rel[1] - rhs[1] * rel[0]
                return torch.cat((self_tr_R, self_tr_I), 1), torch.cat((tr_R, tr_I), 1), torch.sqrt(tr_R ** 2 + tr_I ** 2)

        else:
            return (
                           self_hr_R @ to_score[0].transpose(0, 1) + self_hr_I @ to_score[1].transpose(0, 1)
                   ), [
                (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
                   ]


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim=4000):
        super().__init__()
        # hidden_dim1 = 1024
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_size, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SimCLR(nn.Module):

    def __init__(self, rank, temperature, hidden_size):
        super().__init__()

        self.projection = projection_MLP(rank, hidden_size)
        self.temperature = temperature


    def nt_xentloss(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N, Z = z1.shape
        device = z1.device
        representations = torch.cat([z1, z2], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=-1)
        l_pos = torch.diag(similarity_matrix, N)
        r_pos = torch.diag(similarity_matrix, -N)
        positives = torch.cat([l_pos, r_pos]).view(2 * N, 1)

        diag = torch.eye(2 * N, dtype=torch.bool, device=device)
        diag[N:, :N] = diag[:N, N:] = diag[:N, :N]

        negatives = similarity_matrix[~diag].view(2 * N, -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * N, device=device, dtype=torch.int64)

        loss = F.cross_entropy(logits, labels, reduction='sum')
        return loss / (2 * N)

    def forward(self, x1, x2):
        x1 = self.projection(x1)
        x2 = self.projection(x2)
        loss = self.nt_xentloss(x1, x2)
        return loss

