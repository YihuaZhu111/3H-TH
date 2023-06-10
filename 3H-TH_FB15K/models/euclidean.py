"""Euclidean Knowledge Graph embedding models where embeddings are in real space."""
import numpy as np
import torch
from torch import nn

from models.base import KGModel
from utils.euclidean import euc_sqdistance, givens_rotations, givens_reflection

EUC_MODELS = ["TransE", "QuatE", "ThreeE_TE", "TwoE_TE"]


class BaseE(KGModel):
    """Euclidean Knowledge Graph Embedding models.

    Attributes:
        sim: similarity metric to use (dist for distance and dot for dot product)
    """

    def __init__(self, args):
        super(BaseE, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        if self.sim == "dot":
            if eval_mode:
                score = lhs_e @ rhs_e.transpose(0, 1)
            else:
                score = torch.sum(lhs_e * rhs_e, dim=-1, keepdim=True)
        else:
            score = - euc_sqdistance(lhs_e, rhs_e, eval_mode)
        return score


class TransE(BaseE):
    """Euclidean translations https://www.utc.fr/~bordesan/dokuwiki/_media/en/transe_nips13.pdf"""

    def __init__(self, args):
        super(TransE, self).__init__(args)
        self.sim = "dist"

    def get_queries(self, queries):
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        lhs_e = head_e + rel_e
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases


class QuatE(BaseE):          #  (QuatE)
    """Euclidean 2x2 Givens rotations"""

    def __init__(self, args):
        super(QuatE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0     # (num_relations, rank)

        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""

        def _3d_rotation(relation, entity):    # entity:  (batch_size, rank)     relation: (batch_size, rank)
            relation_re, relation_i, relation_j, relation_k = torch.chunk(relation, 4, dim = -1)        # (batch_size, rank/4)
            entity_re, entity_i, entity_j, entity_k = torch.chunk(entity, 4, dim = -1)                  # (batch_size, rank/4)

            # normalize relation
            denominator_relation = torch.sqrt(relation_re ** 2 + relation_i ** 2 + relation_j ** 2 + relation_k **2)
            relation_re = relation_re / denominator_relation
            relation_i  = relation_i  / denominator_relation
            relation_j  = relation_j  / denominator_relation
            relation_k  = relation_k  / denominator_relation

            # do 3d rotation
            re = entity_re * relation_re - entity_i * relation_i - entity_j * relation_j - entity_k * relation_k
            i  = entity_re * relation_i  + entity_i * relation_re+ entity_j * relation_k - entity_k * relation_j
            j  = entity_re * relation_j  - entity_i * relation_k + entity_j * relation_re+ entity_k * relation_i
            k  = entity_re * relation_k  + entity_i * relation_j - entity_j * relation_i + entity_k * relation_re

            return torch.cat([re, i, j, k], dim = -1)             # (batch_size, rank)

        lhs_e = _3d_rotation(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0]))
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases
    

class ThreeE_TE(BaseE):          # 3E-TE
    """Euclidean 2x2 Givens rotations"""

    def __init__(self, args):
        super(ThreeE_TE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0     # (num_relations, rank)
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""

        def _3d_rotation(relation, entity):    # entity:  (batch_size, rank)     relation: (batch_size, rank)
            relation_re, relation_i, relation_j, relation_k = torch.chunk(relation, 4, dim = -1)        # (batch_size, rank/4)
            entity_re, entity_i, entity_j, entity_k = torch.chunk(entity, 4, dim = -1)                  # (batch_size, rank/4)

            # normalize relation
            denominator_relation = torch.sqrt(relation_re ** 2 + relation_i ** 2 + relation_j ** 2 + relation_k **2)
            relation_re = relation_re / denominator_relation
            relation_i  = relation_i  / denominator_relation
            relation_j  = relation_j  / denominator_relation
            relation_k  = relation_k  / denominator_relation

            # do 3d rotation
            re = entity_re * relation_re - entity_i * relation_i - entity_j * relation_j - entity_k * relation_k
            i  = entity_re * relation_i  + entity_i * relation_re+ entity_j * relation_k - entity_k * relation_j
            j  = entity_re * relation_j  - entity_i * relation_k + entity_j * relation_re+ entity_k * relation_i
            k  = entity_re * relation_k  + entity_i * relation_j - entity_j * relation_i + entity_k * relation_re

            return torch.cat([re, i, j, k], dim = -1)             # (batch_size, rank)

        lhs_e = _3d_rotation(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0])) + self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases


class TwoE_TE(BaseE):           #   2E-TE
    """Euclidean 2x2 Givens rotations"""

    def __init__(self, args):
        super(TwoE_TE, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.sim = "dist"

    def get_queries(self, queries: torch.Tensor):
        """Compute embedding and biases of queries."""
        lhs_e = givens_rotations(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0])) + self.rel(queries[:, 1])
        lhs_biases = self.bh(queries[:, 0])
        return lhs_e, lhs_biases