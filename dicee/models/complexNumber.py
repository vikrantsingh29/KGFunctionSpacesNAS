from .base_model import BaseKGE
import numpy as np
import torch
import torch.nn.functional as F


class ComplexNumber(BaseKGE):

    def __init__(self, args):
        super().__init__(args)

        self.name = 'ComplexNumber'
        self.scoring_function = "compositional"
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
        self.k = int(np.sqrt(self.embedding_dim // 2))
        self.num_sample = self.embedding_dim
        self.gamma = torch.linspace(0, 1, steps=self.num_sample)  # N(0,1)
        from scipy.special import roots_legendre
        roots, weights = roots_legendre(self.num_sample)

    def complex(self, embedding, x_samples) -> torch.Tensor:
        real_part = embedding * torch.cos(x_samples)
        imag_part = embedding * torch.sin(x_samples)
        complex_representation = torch.complex(real_part,imag_part)
        return complex_representation

    def complex_representation(self,embedding, x_samples):
        real_part = embedding * torch.cos(x_samples)
        imag_part = embedding * torch.sin(x_samples)
        complex_representation = real_part + imag_part
        return complex_representation


    def forward_triples(self, idx_triple: torch.Tensor) -> torch.Tensor:

        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(idx_triple)
        self.gamma = self.gamma.to(head_ent_emb.device)
        h_x = self.complex(head_ent_emb, self.gamma)
        t_x = self.complex(tail_ent_emb, self.gamma)
        r_x = self.complex(rel_ent_emb, self.gamma)
        r_h_x = self.complex(rel_ent_emb, h_x.real)
        complex_product = r_h_x * t_x
        real_product = complex_product.real
        score = torch.trapz(real_product, self.gamma, dim=1)

        # score = 0
        # if self.scoring_function == "compositional":
        #     score = torch.trapz(r_h_x * t_x, self.gamma, dim=1)
        # elif self.scoring_function == "trilinear":
        #     score = self.score_calculator(h_x, r_x, t_x, self.scoring_function)

        # score = torch.trapz(product ,self.gamma, dim=1).mean(dim=1)

        return score

    def score_calculator(self, fhx, frx, ftx, scoring_function):
        score = 0
        if scoring_function == 'trilinear':
            score = torch.trapz(fhx * frx * ftx, self.gamma, dim=1)
        elif scoring_function == 'vtp':
            score = 0

        return score


