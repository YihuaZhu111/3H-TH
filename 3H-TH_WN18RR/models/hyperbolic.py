"""Hyperbolic Knowledge Graph embedding models where all parameters are defined in tangent spaces."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c

HYP_MODELS = ["RotH", "TH", "ThreeH", "ThreeH_TH", "ThreeE_TE_ThreeH_TH", "TwoE_TE_TwoH_TH"]

class BaseH(KGModel):
    """Trainable curvature for each relationship."""

    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank), dtype=self.data_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.multi_c = args.multi_c
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2


class RotH(BaseH):
    """Hyperbolic 2x2 Givens rotations"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        head = expmap0(self.entity(queries[:, 0]), c)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel1 = expmap0(rel1, c)
        rel2 = expmap0(rel2, c)
        lhs = project(mobius_add(head, rel1, c), c)
        res1 = givens_rotations(self.rel_diag(queries[:, 1]), lhs)
        res2 = mobius_add(res1, rel2, c)
        return (res2, c), self.bh(queries[:, 0])


class TH(BaseH):    #    (TH)(MuRP)

    def __init__(self, args):
        super(TH, self).__init__(args)
        self.rel = nn.Embedding(self.sizes[1], self.rank)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)

        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else: 
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):      
        """Compute embedding and biases of queries."""

        c = F.softplus(self.c[queries[:, 1]])                                     
        trans_rel_hyperbolic_1 = self.rel(queries[:, 1])
        
        # Euclidean space
        head_original = self.entity(queries[:, 0])                        

        # Hyperbolic space
        trans_rel_hyperbolic_1 = expmap0(trans_rel_hyperbolic_1, c)
        head_hyperbolic = project(expmap0(head_original, c), c)
        head_rot_hyperbolic = mobius_add(head_hyperbolic,trans_rel_hyperbolic_1, c)
        res = project(head_rot_hyperbolic, c)

        return (res, c), self.bh(queries[:, 0])  
    

class ThreeH(BaseH):    #  3H

    def __init__(self, args):
        super(ThreeH, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)             # rel_diag (num_relations, rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0  

        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else: 
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()


    def get_queries(self, queries):        
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

        c = F.softplus(self.c[queries[:, 1]])                                      # (batch_size, 1)
        rot_mat_hyperbolic = self.rel_diag(queries[:, 1])                          # (batch_size, rank)

        # Euclidean space
        head_original = self.entity(queries[:, 0])                                 # (batch_size, rank), Euclidean space 

        # Hyperbolic space
        head_hyperbolic = project(expmap0(head_original, c), c)
        head_rot_hyperbolic = _3d_rotation(rot_mat_hyperbolic ,head_hyperbolic)
        res = project(head_rot_hyperbolic, c)

        return (res, c), self.bh(queries[:, 0]) 


class ThreeH_TH(BaseH):    # 3H-TH  

    def __init__(self, args):
        super(ThreeH_TH, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)             # rel_diag (num_relations, rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0  
       
        self.rel = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank), dtype=self.data_type)

        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else: 
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()


    def get_queries(self, queries):      # QUATE + HYPERBOLIC
        """Compute embedding and biases of queries."""

        def _3d_rotation(relation, entity):    # entity:  (batch_size, rank)     relation: (batch_size, rank)
            relation_re, relation_i, relation_j, relation_k = torch.chunk(relation, 4, dim = -1)        # (batch_size, rank/4)
            entity_re, entity_i, entity_j, entity_k = torch.chunk(entity, 4, dim = -1)                  # (batch_size, rank/4)

            # normalize relation
            denominator_relation = torch.sqrt(relation_re ** 2 + relation_i ** 2 + relation_j ** 2 + relation_k ** 2)
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

        c = F.softplus(self.c[queries[:, 1]])                                       
        trans_rel_hyperbolic_1, trans_rel_hyperbolic_2 = torch.chunk(self.rel(queries[:, 1]), 2, dim = 1)                             # (batch_size, rank)
        rot_mat_hyperbolic = self.rel_diag(queries[:, 1])                            # (batch_size, rank)
       
        # Euclidean space
        head_original = self.entity(queries[:, 0])                                          # (batch_size, rank), Euclidean space 

        # Hyperbolic space
        trans_rel_hyperbolic_1 = expmap0(trans_rel_hyperbolic_1, c)
        trans_rel_hyperbolic_2 = expmap0(trans_rel_hyperbolic_2, c)
        head_hyperbolic = project(mobius_add(expmap0(head_original, c), trans_rel_hyperbolic_1, c), c)
        head_rot_hyperbolic = _3d_rotation(rot_mat_hyperbolic ,head_hyperbolic)
        head_rot_trans_hyperbolic = mobius_add(head_rot_hyperbolic, trans_rel_hyperbolic_2, c)
        res = project(head_rot_trans_hyperbolic, c)

        return (res, c), self.bh(queries[:, 0])       


class ThreeE_TE_ThreeH_TH(BaseH):    #  3E-TE-3H-TH

    def __init__(self, args):
        super(ThreeE_TE_ThreeH_TH, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)             # rel_diag (num_relations, 2 * rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0  #初始化参数 ,最后（-1，1）均匀分布
       
        self.rel = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank), dtype=self.data_type)

        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else: 
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):
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


        c = F.softplus(self.c[queries[:, 1]])                                     
        trans_rel_euclidean, trans_rel_hyperbolic_1 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)    # (batch_size, rank)
        rot_mat_euclidean, rot_mat_hyperbolic = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)   # (batch_size, rank)
        
        # Euclidean space
        head_original = self.entity(queries[:, 0])                                          # (batch_size, rank), Euclidean space 
        head_rot_euclidean = _3d_rotation(rot_mat_euclidean, head_original)             # (batch_size, rank)  Rotation
        head_rot_trans_euclidean = head_rot_euclidean + trans_rel_euclidean                 # (batch_size, rank) Rotation and Translation

        # Hyperbolic space
        trans_rel_hyperbolic_1 = expmap0(trans_rel_hyperbolic_1, c)
        head_hyperbolic = project(expmap0(head_rot_trans_euclidean, c), c)
        head_rot_hyperbolic = _3d_rotation(rot_mat_hyperbolic ,head_hyperbolic)
        head_rot_trans_hyperbolic = mobius_add(head_rot_hyperbolic, trans_rel_hyperbolic_1, c)
        res = project(head_rot_trans_hyperbolic, c)

        return (res, c), self.bh(queries[:, 0])  
   
    
class TwoE_TE_TwoH_TH(BaseH):      # 2E-TE-2H-TH

    def __init__(self, args):
        super(TwoE_TE_TwoH_TH, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)             # rel_diag (num_relations, 2*rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0 
       
        self.rel = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank), dtype=self.data_type)

        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else: 
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):
        c = F.softplus(self.c[queries[:, 1]])                                     
        
        trans_rel_euclidean, trans_rel_hyperbolic_1 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)    # (batch_size, rank)
        rot_mat_euclidean, rot_mat_hyperbolic = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)   # (batch_size, rank)
        
        # Euclidean space
        head_original = self.entity(queries[:, 0])                                          # (batch_size, rank), Euclidean space 
        head_rot_euclidean = givens_rotations(rot_mat_euclidean, head_original)             # (batch_size, rank)  Rotation
        head_rot_trans_euclidean = head_rot_euclidean + trans_rel_euclidean                 # (batch_size, rank) Rotation and Translation

        # Hyperbolic space
        trans_rel_hyperbolic_1 = expmap0(trans_rel_hyperbolic_1, c)
        head_hyperbolic = project(expmap0(head_rot_trans_euclidean, c), c)
        head_rot_hyperbolic = givens_rotations(rot_mat_hyperbolic ,head_hyperbolic)
        head_rot_trans_hyperbolic = mobius_add(head_rot_hyperbolic, trans_rel_hyperbolic_1, c)
        res = project(head_rot_trans_hyperbolic, c)

        return (res, c), self.bh(queries[:, 0])       