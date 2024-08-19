from torch import Tensor

from .base_model import BaseKGE
import numpy as np
import torch
import torch.nn.functional as F
import unittest


class Polynomial(BaseKGE):

    def __init__(self, args):
        super().__init__(args)

        self.name = 'Polynomial'
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
        self.k = int(np.sqrt(self.embedding_dim // 2))
        self.num_sample = self.embedding_dim
        self.gamma = torch.linspace(0, 1, steps=self.num_sample)
        from scipy.special import roots_legendre
        roots, weights = roots_legendre(self.num_sample)

    @staticmethod
    def polynomial(embedding, x_samples) -> Tensor:
        # powers_of_x = torch.stack([x_samples ** i for i in range(embedding.size(1))], dim=-1)
        # emb_expanded = embedding.unsqueeze(1)
        # poly_vector = emb_expanded * powers_of_x
        # poly_vector = poly_vector.sum(dim=-1)
        #
        # # Normalize the output vector for each item in the batch
        # poly_vector = F.normalize(poly_vector, p=2, dim=-1)

        # Determine the polynomial degree based on the embedding dimension
        degree = embedding.size(1) - 1  # Highest degree equals embedding_dim - 1

        # Generate powers of x_samples up to the specified degree
        powers_of_x = torch.stack([x_samples ** i for i in range(degree + 1)], dim=-1)

        # Expand the embedding tensor for element-wise multiplication with powers_of_x
        emb_expanded = embedding.unsqueeze(1)

        # Compute the polynomial vector by element-wise multiplication and sum across the powers
        poly_vector = (emb_expanded * powers_of_x).sum(dim=-1)

        # Normalize the output vector for each item in the batch
        poly_vector = F.normalize(poly_vector, p=2, dim=-1)

        return poly_vector

    def forward_triples(self, idx_triple: torch.Tensor) -> torch.Tensor:
        print("idx_triple", idx_triple)
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(idx_triple)
        self.gamma = self.gamma.to(head_ent_emb.device)
        h_x = self.polynomial(head_ent_emb, self.gamma)
        t_x = self.polynomial(tail_ent_emb, self.gamma)
        r_x = self.polynomial(rel_ent_emb, self.gamma)
        r_h_x = self.polynomial(rel_ent_emb, h_x)

        # if self.args["scoring_func"] == "compositional":
        score = torch.trapz(r_h_x * t_x, self.gamma, dim=1)

        # if self.args["scoring_func"] == "vtp":
        #     termA = - torch.trapz(t_x, self.gamma, dim=1)
        #     termB = torch.trapz(h_x * r_x, self.gamma, dim=1)
        #     termC = torch.trapz(r_x, self.gamma, dim=1)
        #     termD = torch.trapz(t_x * h_x, self.gamma, dim=1)
        #     score = termA * termB + termC * termD
        #
        # if self.args["scoring_func"] == "trilinear":
        #     score = torch.trapezoid(h_x * t_x * r_x, self.gamma, dim=1)

        return score



