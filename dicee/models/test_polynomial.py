import torch
import unittest
from .polynomial import Polynomial

class TestPolynomial(unittest.TestCase):

    def setUp(self):
        args = {'embedding_dim': 4, 'num_entities': 3, 'num_relations': 2}
        self.model = Polynomial(args)
        # Manually setting embedding weights for testing
        self.model.entity_embeddings.weight.data = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                                 [0.0, 1.0, 0.0, 0.0],
                                                                 [1.0, -1.0, 0.0, 0.0]])
        self.model.relation_embeddings.weight.data = torch.tensor([[0.0, 1.0, 0.0, 0.0],
                                                                   [0.0, 0.0, 1.0, 0.0]])
        self.gamma = torch.tensor([0, 0.5, 1], dtype=torch.float32)

    def test_h_x(self):
        # Test h_x computation
        head_idx = torch.tensor([0])  # Index of the head entity
        head_ent_emb = self.model.entity_embeddings(head_idx)
        h_x = self.model.polynomial(head_ent_emb, self.gamma)

        # Expected h_x for embedding [1.0, 0.0, 0.0, 0.0] and gamma [0, 0.5, 1] is [1.0, 1.0, 1.0]
        expected_h_x = torch.tensor([[1.0, 1.0, 1.0]])
        print(h_x, expected_h_x , "h_x", "expected_h_x")
        self.assertTrue(torch.allclose(h_x, expected_h_x))

    def test_r_x(self):
        # Test r_x computation
        rel_idx = torch.tensor([0])  # Index of the relation
        rel_ent_emb = self.model.relation_embeddings(rel_idx)
        r_x = self.model.polynomial(rel_ent_emb, self.gamma)

        # Expected r_x for embedding [0.0, 1.0, 0.0, 0.0] and gamma [0, 0.5, 1] is [0.0, 0.5, 1.0]
        expected_r_x = torch.tensor([[0.0, 0.5, 1.0]])
        print(r_x, expected_r_x, "r_x", "expected_r_x")
        self.assertTrue(torch.allclose(r_x, expected_r_x))

    def test_t_x(self):
        # Test t_x computation
        tail_idx = torch.tensor([1])  # Index of the tail entity
        tail_ent_emb = self.model.entity_embeddings(tail_idx)
        t_x = self.model.polynomial(tail_ent_emb, self.gamma)

        # Expected t_x for embedding [0.0, 1.0, 0.0, 0.0] and gamma [0, 0.5, 1] is [0.0, 1.0, 1.0]
        expected_t_x = torch.tensor([[0.0, 1.0, 1.0]])
        print(t_x, expected_t_x, "t_x", "expected_t_x")
        self.assertTrue(torch.allclose(t_x, expected_t_x))

    def test_r_h_x(self):
        # Test r_h_x computation (r applied to h_x)
        head_idx = torch.tensor([0])  # Index of the head entity
        head_ent_emb = self.model.entity_embeddings(head_idx)
        h_x = self.model.polynomial(head_ent_emb, self.gamma)

        rel_idx = torch.tensor([0])  # Index of the relation
        rel_ent_emb = self.model.relation_embeddings(rel_idx)
        r_h_x = self.model.polynomial(rel_ent_emb, h_x)

        # Expected r_h_x is r_x multiplied by h_x
        expected_r_h_x = torch.tensor([[0.0, 0.5, 1.0]])  # Same
        print(r_h_x, expected_r_h_x, "r_h_x", "expected_r_h_x")
        self.assertTrue(torch.allclose(r_h_x, expected_r_h_x))