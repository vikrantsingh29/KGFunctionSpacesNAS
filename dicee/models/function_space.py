from nni.mutable import ensure_frozen

from .base_model import BaseKGE
import torch
import numpy as np
from scipy.special import roots_legendre


class FMult(BaseKGE):
    """ Learning Knowledge Neural Graphs"""
    """ Learning Neural Networks for Knowledge Graphs"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'FMult'
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
        self.k = int(np.sqrt(self.embedding_dim // 2))
        self.num_sample = 50
        # self.gamma = torch.rand(self.k, self.num_sample) [0,1) uniform=> worse results
        self.gamma = torch.randn(self.k, self.num_sample)  # N(0,1)
        from scipy.special import roots_legendre
        roots, weights = roots_legendre(self.num_sample)
        self.roots = torch.from_numpy(roots).repeat(self.k, 1).float()  # shape self.k by self.n
        self.weights = torch.from_numpy(weights).reshape(1, -1).float()  # shape 1 by self.n

    def compute_func(self, weights: torch.FloatTensor, x) -> torch.FloatTensor:
        n = len(weights)
        # Weights for two linear layers.
        w1, w2 = torch.hsplit(weights, 2)
        # (1) Construct two-layered neural network
        w1 = w1.view(n, self.k, self.k)
        w2 = w2.view(n, self.k, self.k)
        # (2) Forward Pass
        out1 = torch.tanh(w1 @ x)  # torch.sigmoid => worse results
        out2 = w2 @ out1
        return out2  # no non-linearity => better results

    def chain_func(self, weights, x: torch.FloatTensor):
        n = len(weights)
        # Weights for two linear layers.
        w1, w2 = torch.hsplit(weights, 2)
        # (1) Construct two-layered neural network
        w1 = w1.view(n, self.k, self.k)
        w2 = w2.view(n, self.k, self.k)
        # (2) Perform the forward pass
        out1 = torch.tanh(torch.bmm(w1, x))
        out2 = torch.bmm(w2, out1)
        return out2

    def forward_triples(self, idx_triple: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings: batch, \mathbb R^d
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(idx_triple)
        # (2) Compute NNs on \Gamma
        # Logits via FDistMult...
        # h_x = self.compute_func(head_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        # r_x = self.compute_func(rel_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        # t_x = self.compute_func(tail_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        # out = h_x * r_x * t_x  # batch, \mathbb{R}^k, |gamma|
        # (2) Compute NNs on \Gamma
        self.gamma = self.gamma.to(head_ent_emb.device)

        h_x = self.compute_func(head_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        t_x = self.compute_func(tail_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        r_h_x = self.chain_func(weights=rel_ent_emb, x=h_x)  # batch, \mathbb{R}^k, |\Gamma|
        # (3) Compute |\Gamma| predictions
        out = torch.sum(r_h_x * t_x, dim=1)  # batch, |gamma| #
        # (4) Average (3) over \Gamma
        out = torch.mean(out, dim=1)  # batch
        return out

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings for head entities and relations
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)

        # (2) Compute NNs on \Gamma for head and relation embeddings
        self.gamma = self.gamma.to(head_ent_emb.device)
        h_x = self.compute_func(head_ent_emb, x=self.gamma)  # batch, \mathbb{R}^k, |\Gamma|
        r_h_x = self.compute_func(weights=rel_ent_emb, x=h_x)  # batch, \mathbb{R}^k, |\Gamma|

        # (3) Use all entity embeddings as potential tails
        all_tails = self.entity_embeddings.weight
        all_tails_x = self.compute_func(all_tails, x=self.gamma)  # num_entities, \mathbb{R}^k, |\Gamma|

        # (4) Compute scores against all tails
        scores = torch.trapz(r_h_x.unsqueeze(1) * all_tails_x.unsqueeze(0), self.gamma, dim=3).mean(
            dim=2)  # batch, num_entities

        return scores


import torch.nn as nn
import torch.nn.functional as F

# class DynamicNeuralNetworkNAS(nn.Module):
#     def __init__(self, adjusted_input_dim, num_samples, num_layers, activation_function, dropout_rate):
#         super().__init__()
#         self.num_layers = num_layers
#         self.layer_matrix_size = int((adjusted_input_dim // num_layers) ** 0.5)
#         self.num_samples = num_samples
#         self.activation = getattr(F, activation_function)
#         self.dropout = nn.Dropout(dropout_rate)
#
#         # Layers dynamically added based on num_layers
#         self.layers = nn.ModuleList(
#             [nn.Linear(self.layer_matrix_size, self.layer_matrix_size) for _ in range(num_layers)])
#
#     def forward(self, weights: torch.FloatTensor, x):
#         # Dynamic adjustment for batch size
#         batch_size = weights.shape[0]
#
#         # Split the input embeddings into segments for each layer
#         split_size = self.layer_matrix_size * self.layer_matrix_size
#         embedding_segments = torch.hsplit(weights, self.num_layers)
#         layer_outputs = []
#
#         for i, layer in enumerate(self.layers):
#             if i < len(embedding_segments):
#                 weights = embedding_segments[i].view(batch_size, self.layer_matrix_size, self.layer_matrix_size)
#
#                 if (i % 2 == 0):
#                     x = self.activation(weights @ x)
#                 else:
#                     x = weights @ x
#
#                 x = self.dropout(x)
#
#
#         # layer_outputs.append(x)
#         # # Example: Multiplying outputs of consecutive layers
#         # for i in range(len(layer_outputs) - 1):
#         #     x = layer_outputs[i] * layer_outputs[i + 1]
#
#         return x

# class DynamicNeuralNetworkNAS(nn.Module):
#     def __init__(self, adjusted_input_dim, num_samples, num_layers, activation_function, dropout_rate):
#         super().__init__()
#         self.num_layers = num_layers
#         self.layer_matrix_size = int((adjusted_input_dim // num_layers) ** 0.5)
#         self.num_samples = num_samples
#         self.activation = getattr(F, activation_function)
#         self.dropout = nn.Dropout(dropout_rate)
#
#         # Layers dynamically added based on num_layers
#         self.layers = nn.ModuleList(
#             [nn.Linear(self.layer_matrix_size, self.layer_matrix_size) for _ in range(num_layers)])
#
#     def forward(self, weights: torch.FloatTensor, x):
#         # Dynamic adjustment for batch size
#         batch_size = weights.shape[0]
#
#         # Split the input embeddings into segments for each layer
#         embedding_segments = torch.hsplit(weights, self.num_layers)
#
#         # Initialize the score with the result of the first layer computation
#         W = embedding_segments[0].view(batch_size, self.layer_matrix_size, self.layer_matrix_size)
#         score = self.compute_func(W, x ,0)  # Assuming compute_func is defined and handles the computation for one layer
#
#         # Iterate over the remaining layers and update the score
#         for i, segment in enumerate(embedding_segments[1:], start=1):
#             W = segment.view(batch_size, self.layer_matrix_size, self.layer_matrix_size)
#             score = score * self.compute_func(W, x ,i)  # Multiplying score with the result of compute_func for each layer
#
#         return score
#
#     def compute_func(self, W, x , i):
#         if i%2==0 and i!=0:
#             out = self.activation(W @ x)
#         else:
#             out = W @ x
#
#         return out
# class DynamicNeuralNetworkNAS(nn.Module):
#     def __init__(self, adjusted_input_dim, num_samples, num_layers, activation_function, dropout_rate):
#         super().__init__()
#         self.num_layers = num_layers
#         self.layer_matrix_size = int((adjusted_input_dim // num_layers) ** 0.5)
#         self.num_samples = num_samples
#         self.activation = getattr(F, activation_function)  # Ensure this is a valid nn module like nn.Sigmoid
#         self.dropout = nn.Dropout(dropout_rate)
#
#     def forward(self, weights: torch.FloatTensor, x):
#         # Dynamic adjustment for batch size
#         batch_size = weights.shape[0]
#
#         # Split the input weights into segments for each layer
#         embedding_segments = torch.hsplit(weights, self.num_layers)
#
#         # Initialize score with input
#         score = x
#
#         for i, segment in enumerate(embedding_segments):
#             # Reshape the segment for the layer operation
#             layer_weights = segment.view(batch_size, self.layer_matrix_size, self.layer_matrix_size)
#
#             # Apply layer-wise transformation using external weights
#             transformed = torch.matmul(layer_weights, score)
#
#             # Apply activation function and dropout
#             if i % 2 == 0:
#                 transformed = self.activation(transformed)
#             transformed = self.dropout(transformed)
#
#             # Residual connection
#             if score.size() == transformed.size():
#                 score = score + transformed
#             else:
#                 score = transformed
#
#             # score = self.dropout(score)
#
#         return score


import torch
import torch.nn as nn
import torch.nn.functional as F
import nni



class DynamicNeuralNetworkNAS(nn.Module):
    def __init__(self, adjusted_input_dim, num_samples, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        params = nni.get_current_parameter()
        self.layer_matrix_size = int((adjusted_input_dim // num_layers) ** 0.5)
        self.num_samples = num_samples
        self.dropout = nn.Dropout(dropout_rate)

        # Using different activations for even and odd layers
        self.even_activation = F.sigmoid
        # self.odd_activation = F.leaky_relu

        # Layer normalization
        self.layer_norm = nn.LayerNorm([self.layer_matrix_size, self.num_samples])

    def forward(self, weights: torch.FloatTensor, x):
        batch_size = weights.shape[0]

        # Split the input weights into segments for each layer
        embedding_segments = torch.hsplit(weights, self.num_layers)


        for i, segment in enumerate(embedding_segments):
            layer_weights = segment.view(batch_size, self.layer_matrix_size, self.layer_matrix_size)

            # Apply transformation using external weights
            transformed = torch.matmul(layer_weights, x)

            # Apply layer normalization
            transformed = self.layer_norm(transformed)

            # Apply activation function based on layer index
            if i % 2 == 0:
                transformed = self.even_activation(transformed)
            # else:
            #     transformed = self.odd_activation(transformed)

            # transformed = self.dropout(transformed)

            #  residual connection
            if i > 0 and x.size() == transformed.size():
                x = x + transformed / (i + 1)
            else:
                x = transformed

        return transformed

# import nni
# from nni import nas
# from nni.nas.nn import pytorch as pyt
#
# class DynamicLayer(nn.Module):
#     def __init__(self, input_features):
#         super(DynamicLayer, self).__init__()
#         # Define LayerChoice with dense and convolutional options
#         self.layer_choice = pyt.LayerChoice([
#             nn.Linear(input_features, 50),  # Dense layer
#             nn.Sequential(  # Convolutional layer wrapped in Sequential
#                 nn.Conv1d(1, 10, kernel_size=5, padding=2),
#                 nn.Flatten(),
#                 nn.Linear(10 * input_features, 50)  # Reshape to match output
#             )
#         ])
#
#         # **Crucially, place `ensure_frozen` within __init__ to fix context:**
#         self.layer_choice = ensure_frozen(self.layer_choice)
#
#     def forward(self, x):
#         return self.layer_choice(x)
#
# class DynamicNeuralNetworkNAS(nn.Module):
#     def __init__(self, input_dim, num_layers=2):
#         super(DynamicNeuralNetworkNAS, self).__init__()
#         self.layers = nn.ModuleList([
#             DynamicLayer(input_dim if i == 0 else 50) for i in range(num_layers)
#         ])
#
#     def forward(self, weights, x):
#         batch_size = weights.shape[0]
#         x = x.view(batch_size, 1, -1)  # Reshape for convolutional compatibility
#
#         for layer in self.layers:
#             x = layer(x)
#             x = F.relu(x)  # Example activation
#
#         return x.view(batch_size, -1)  # Reshape output


# Example usage
# input_dim = 50  # Adjust based on your setup
# output_dim = 50  # Same as input_dim for your case
# num_layers = 2  # Number of layers in the network
#
# # Initialize the network
# dynamic_network = DynamicNeuralNetworkNAS(input_dim, output_dim, num_layers)
#
# # Example input
# x = torch.randn(32, input_dim)  # Batch size of 32
# weights = torch.randn(32, 50)  # Batch of weights, one for each item in the batch
#
# # Forward pass
# output = dynamic_network(weights, x)

# class KnowledgeGraphCNNExtended(nn.Module):
#     def __init__(self, input_channels, output_channels1, output_channels2, kernel_size):
#         super(KnowledgeGraphCNNExtended, self).__init__()
#         # First 1D convolutional layer
#         self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=output_channels1, kernel_size=kernel_size,
#                                stride=1, padding='same')
#         # Second 1D convolutional layer, taking the output of the first layer as input
#         self.conv2 = nn.Conv1d(in_channels=output_channels1, out_channels=output_channels2, kernel_size=kernel_size,
#                                stride=1, padding='same')
#         # Fully connected layer; adjust dimensions as needed
#         self.fc = nn.Linear(50 * output_channels2,
#                             50)  # Assuming the final convolution output is reshaped back to 50-dimensional space
#
#     def forward(self, x, weights):
#         # Check if x is already batched (e.g., shape (192, 50)) or a single vector (shape (50,))
#         if x.dim() == 2 and x.shape[0] == weights.shape[0]:  # x is already batched
#             x = x.unsqueeze(1)  # Add a channel dimension, resulting in shape (192, 1, 50)
#         elif x.dim() == 1:  # x is a single vector
#             x = x.unsqueeze(0).unsqueeze(0)  # Add channel and batch dimensions, resulting in shape (1, 1, 50)
#             x = x.expand(weights.shape[0], 1, -1)  # Expand to match the batch size, resulting in shape (192, 1, 50)
#         else:
#             raise ValueError("Unexpected shape for x")
#
#         weights = weights.unsqueeze(1)
#
#         x_weighted = x * weights
#         # Pass the weighted input through the first convolutional layer
#         x_conv1 = F.relu(self.conv1(x_weighted))
#
#         # Pass the result through the second convolutional layer
#         x_conv2 = F.relu(self.conv2(x_conv1))
#
#         # Flatten the output from the convolutional layers to feed into the fully connected layer
#         x_flat = torch.flatten(x_conv2, 1)
#         x_out = self.fc(x_flat)
#
#         return x_out


# class KnowledgeGraphCNNExtended(nn.Module):
#     def __init__(self, input_channels, output_channels1, output_channels2, kernel_size, num_points, num_layers):
#         super(KnowledgeGraphCNNExtended, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=input_channels,
#                                out_channels=output_channels1,
#                                kernel_size=kernel_size,
#                                stride=1, padding='same')
#         self.conv2 = nn.Conv1d(in_channels=output_channels1,
#                                out_channels=output_channels2,
#                                kernel_size=kernel_size,
#                                stride=1, padding='same')
#         # Adjust the fully connected layer dynamically based on num_points
#         self.fc = nn.Linear(num_points * output_channels2, num_points)
#         self.num_layers = num_layers
#
#     def forward(self, x, weights):
#         # Ensure x is of shape (batchSize, 1, num_points)
#         if x.dim() == 1:  # x is a single vector
#             x = x.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_points)
#             x = x.repeat(weights.shape[0], 1, 1)  # Shape: (batchSize, 1, num_points)
#         elif x.dim() == 3 and x.shape[1] == 1:  # x is already in the correct shape
#             pass  # No action needed
#         else:
#             raise ValueError(f"Unexpected shape for x: {x.shape}")
#
#         weights = weights.unsqueeze(1)  # Shape: (batchSize, 1, embedding_dim)
#
#         # Split the input weights into segments for each layer
#         segment_size = weights.shape[2] // self.num_layers  # Assuming weights is of shape (batchSize, 1, embedding_dim)
#         weight_segments = torch.split(weights, segment_size, dim=2)
#
#         # Process input through the first convolutional layer with the first segment of weights
#         x_conv1 = F.relu(self.conv1(x @ weight_segments[0]))  # Element-wise multiplication of x and the first segment
#
#         # Process the output of the first layer through the second convolutional layer with the second segment of weights
#         x_conv2 = F.relu(self.conv2(x_conv1 * weight_segments[1]))  # Element-wise multiplication
#
#         # Flatten and pass through the fully connected layer
#         x_flat = torch.flatten(x_conv2, 1)
#         x_out = self.fc(x_flat)
#
#         return x_out
#
import math


# class KnowledgeGraphCNNExtended(nn.Module):
#     def __init__(self, input_channels, output_channels1, output_channels2, kernel_size, k, num_points, num_layers):
#         super(KnowledgeGraphCNNExtended, self).__init__()
#         self.k = k
#         self.num_layers = num_layers
#         self.num_points = num_points
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.layer_norm = nn.LayerNorm([self.k, num_points]).to(self.device)
#         self.conv1 = nn.Conv1d(in_channels=k,
#                                out_channels=output_channels1,
#                                kernel_size=kernel_size,
#                                stride=1, padding='same')
#         self.pool1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=output_channels1,
#                                out_channels=output_channels1,
#                                kernel_size=kernel_size,
#                                stride=1, padding='same')
#         self.pool2 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
#         # FC layer to transform the flattened output back to (batch_size, k, num_points)
#         fc_input_features = output_channels1 * num_points
#         self.fc = nn.Linear(fc_input_features, k * num_points)  # Adjust the size based on pooling
#
#
#     def forward(self, weights, x):
#         batch_size = weights.shape[0]
#         x = x.to(self.device)
#
#         if x.dim() == 2 and x.shape[0] == self.k:
#             x = x.unsqueeze(0).repeat(batch_size, 1, 1)
#         elif x.dim() == 3 and x.shape[0] == batch_size and x.shape[1] == self.k:
#             pass  # x is already in correct shape
#         else:
#             raise ValueError(f"Unexpected shape for x: {x.shape}")
#
#         embedding_segments = torch.chunk(weights, self.num_layers, dim=1)
#
#         for i, segment in enumerate(embedding_segments):
#             layer_weights = segment.view(batch_size, self.k, self.k)
#             transformed_x = torch.bmm(layer_weights, x)
#             transformed_x = self.layer_norm(transformed_x)
#
#             if i == 0:
#                 x_conv = self.conv1(transformed_x)
#                 x_conv = self.pool1(x_conv)  # Apply pooling after the first convolution
#             else:
#                 x_conv = self.conv2(x_conv)
#                 x_conv = self.pool2(x_conv)  # Apply pooling after the second convolution
#
#             x = x_conv
#         # Flatten the output and pass through the FC layer to get back to (batch_size, k, num_points)
#         x_flat = x_conv.view(batch_size, -1)
#         x_out = self.fc(x_flat)
#         x_out = x_out.view(batch_size, self.k, self.num_points)  # Reshape to desired output shape
#
#         return x_out
#
#
# import nni


class NAS(BaseKGE):
    """ Learning Knowledge Neural Graphs"""
    """ Learning Neural Networks for Knowledge Graphs"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'NAS'
        params = nni.get_current_parameter()
        self.n_layers = 2
        # dropout_rate = params["dropout_rate"]
        # activation =  params["activation_function"]
        tuned_embedding_dim = False
        while int(np.sqrt(self.embedding_dim / self.n_layers)) != np.sqrt(
                self.embedding_dim / self.n_layers):
            self.embedding_dim += 1
            tuned_embedding_dim = True

        if tuned_embedding_dim:
            print(f"\n\n*****Embedding dimension reset to {self.embedding_dim} to fit model architecture!*****\n")
        self.k = int(np.sqrt(self.embedding_dim // self.n_layers))
        self.n = 50
        self.a, self.b = -1.0, 1.0
        self.discrete_points = torch.linspace(self.a, self.b, steps=self.n).repeat(self.k, 1)
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
        # self.neuralNetwork = DynamicNeuralNetworkNAS(50)
        # Model instantiation with updated parameters for two convolutional layers
        output_channels1 = self.k  # Number of filters in the first convolutional layer
        output_channels2 = self.k  # Number of filters in the second convolutional layer
        input_channels = 1  # Treating each 50-dimensional vector as a 1D 'image' with a single channel
        kernel_size = 3

        # Create the extended model instance
        # Create the extended model instance
        # self.extended_model = KnowledgeGraphCNNExtended(input_channels,
        #                                                 output_channels1,
        #                                                 output_channels2,
        #                                                 kernel_size,
        #                                                 self.k,
        #                                                 self.n,
        #                                                 self.n_layers)

        self.extended_model = DynamicNeuralNetworkNAS(self.embedding_dim, self.n,self.n_layers,0)



    def forward_triples(self, idx_triple: torch.Tensor) -> torch.Tensor:
        # idx_triple = idx_triple.to(self.entity_embeddings.weight.device)
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(idx_triple)

        self.discrete_points = self.discrete_points.to(head_ent_emb.device)
        h_x = self.extended_model(weights=head_ent_emb, x=self.discrete_points)
        t_x = self.extended_model(weights=tail_ent_emb, x=self.discrete_points)
        r_x = self.extended_model(weights=rel_ent_emb, x=self.discrete_points)
        r_h_x = self.extended_model(weights=rel_ent_emb, x=h_x)
        rtx = self.extended_model(weights=rel_ent_emb, x=t_x)

        score = 0

        if self.args["scoring_func"] == "compositional":
            # score = torch.trapezoid(r_h_x * t_x, self.discrete_points, dim=1)
            score = torch.trapezoid(r_h_x * t_x, self.discrete_points, dim=2).mean(dim=1)

        if self.args["scoring_func"] == "vtp":
            termA = - torch.trapz(t_x, self.discrete_points, dim=2).mean(dim=1)
            termB = torch.trapz(h_x * r_x, self.discrete_points, dim=2).mean(dim=1)
            termC = torch.trapz(r_x, self.discrete_points, dim=2).mean(dim=1)
            termD = torch.trapz(t_x * h_x, self.discrete_points, dim=2).mean(dim=1)
            score = termA * termB + termC * termD

        if self.args["scoring_func"] == "trilinear":
            score = torch.trapezoid(h_x * t_x * r_x, self.discrete_points, dim=2).mean(dim=1)

        return score

    def forward_k_vs_all(self, x: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings for head entities and relations
        head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)

        # (2) Move discrete points to the correct device
        self.discrete_points = self.discrete_points.to(head_ent_emb.device)

        # (3) Compute NNs for head and relation embeddings on discrete points
        h_x = self.neuralNetwork(weights=head_ent_emb, x=self.discrete_points)
        r_x = self.neuralNetwork(weights=rel_ent_emb, x=self.discrete_points)
        r_h_x = self.neuralNetwork(weights=rel_ent_emb, x=h_x)

        # (4) Use all entity embeddings as potential tails
        all_tails = self.entity_embeddings.weight
        all_tails_x = self.neuralNetwork(weights=all_tails, x=self.discrete_points)

        scores = torch.trapz(r_h_x.unsqueeze(1) * all_tails_x.unsqueeze(0), self.discrete_points, dim=3).mean(dim=2)


        return scores


class GFMult(BaseKGE):
    """ Learning Knowledge Neural Graphs"""
    """ Learning Neural Networks for Knowledge Graphs"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'GFMult'
        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
        self.k = int(np.sqrt(self.embedding_dim // 2))
        self.num_sample = 250
        roots, weights = roots_legendre(self.num_sample)
        self.roots = torch.from_numpy(roots).repeat(self.k, 1).float()  # shape self.k by self.n
        self.weights = torch.from_numpy(weights).reshape(1, -1).float()  # shape 1 by self.n

    def compute_func(self, weights: torch.FloatTensor, x) -> torch.FloatTensor:
        n = len(weights)
        # Weights for two linear layers.
        w1, w2 = torch.hsplit(weights, 2)
        # (1) Construct two-layered neural network
        w1 = w1.view(n, self.k, self.k)
        w2 = w2.view(n, self.k, self.k)
        # (2) Forward Pass
        out1 = torch.tanh(w1 @ x)  # torch.sigmoid => worse results
        out2 = w2 @ out1
        return out2  # no non-linearity => better results

    def chain_func(self, weights, x: torch.FloatTensor):
        n = len(weights)
        # Weights for two linear layers.
        w1, w2 = torch.hsplit(weights, 2)
        # (1) Construct two-layered neural network
        w1 = w1.view(n, self.k, self.k)
        w2 = w2.view(n, self.k, self.k)
        # (2) Perform the forward pass
        out1 = torch.tanh(torch.bmm(w1, x))
        out2 = torch.bmm(w2, out1)
        return out2

    def forward_triples(self, idx_triple: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings: batch, \mathbb R^d
        head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(idx_triple)
        # (2) Compute NNs on \Gamma
        self.roots = self.roots.to(head_ent_emb.device)
        self.weights = self.weights.to(head_ent_emb.device)

        h_x = self.compute_func(head_ent_emb, x=self.roots)  # batch, \mathbb{R}^k, |\Gamma|
        t_x = self.compute_func(tail_ent_emb, x=self.roots)  # batch, \mathbb{R}^k, |\Gamma|
        r_h_x = self.chain_func(weights=rel_ent_emb, x=h_x)  # batch, \mathbb{R}^k, |\Gamma|
        # (3) Compute |\Gamma| predictions.
        out = torch.sum(r_h_x * t_x, dim=1) * self.weights  # batch, |gamma| #
        # (4) Average (3) over \Gamma
        out = torch.mean(out, dim=1)  # batch
        return out


class FMult2(BaseKGE):
    """ Learning Knowledge Neural Graphs"""
    """ Learning Neural Networks for Knowledge Graphs"""

    def __init__(self, args):
        super().__init__(args)
        self.name = 'FMult2'
        self.n_layers = 2
        tuned_embedding_dim = False
        while int(np.sqrt((self.embedding_dim - 1) / self.n_layers)) != np.sqrt(
                (self.embedding_dim - 1) / self.n_layers):
            self.embedding_dim += 1
            tuned_embedding_dim = True

        if tuned_embedding_dim:
            print(f"\n\n*****Embedding dimension reset to {self.embedding_dim} to fit model architecture!*****\n")
        self.k = int(np.sqrt((self.embedding_dim - 1) // self.n_layers))
        self.n = 50
        self.a, self.b = -1.0, 1.0
        # self.score_func = "vtp" # "vector triple product"
        # self.score_func = "trilinear"
        self.score_func = "compositional"
        # self.score_func = "full-compositional"
        # self.discrete_points = torch.linspace(self.a, self.b, steps=self.n)
        self.discrete_points = torch.linspace(self.a, self.b, steps=self.n).repeat(self.k, 1)

        self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim)
        self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim)
        self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)

    def build_func(self, Vec):
        n = len(Vec)
        # (1) Construct self.n_layers layered neural network
        W = list(torch.hsplit(Vec[:, :-1], self.n_layers))
        # (2) Reshape weights of the layers
        for i, w in enumerate(W):
            W[i] = w.reshape(n, self.k, self.k)
        return W, Vec[:, -1]

    def build_chain_funcs(self, list_Vec):
        list_W = []
        list_b = []
        for Vec in list_Vec:
            W_, b = self.build_func(Vec)
            list_W.append(W_)
            list_b.append(b)

        W = list_W[-1][1:]
        for i in range(len(list_W) - 1):
            for j, w in enumerate(list_W[i]):
                if i == 0 and j == 0:
                    W_temp = w
                else:
                    W_temp = w @ W_temp
            W_temp = W_temp + list_b[i].reshape(-1, 1, 1)
        W_temp = list_W[-1][0] @ W_temp / ((len(list_Vec) - 1) * w.shape[1])
        W.insert(0, W_temp)
        return W, list_b[-1]

    def compute_func(self, W, b, x) -> torch.FloatTensor:
        out = W[0] @ x
        for i, w in enumerate(W[1:]):
            if i % 2 == 0:  # no non-linearity => better results
                out = out + torch.sigmoid(w @ out)
            else:
                out = out + w @ out
        return out + b.reshape(-1, 1, 1)

    def function(self, list_W, list_b):
        def f(x):
            if len(list_W) == 1:
                return self.compute_func(list_W[0], list_b[0], x)
            score = self.compute_func(list_W[0], list_b[0], x)
            for W, b in zip(list_W[1:], list_b[1:]):
                score = score * self.compute_func(W, b, x)
            return score

        return f

    def trapezoid(self, list_W, list_b):
        return torch.trapezoid(self.function(list_W, list_b)(self.discrete_points), x=self.discrete_points, dim=-1).sum(
            dim=-1)

    def forward_triples(self, idx_triple: torch.Tensor) -> torch.Tensor:
        # (1) Retrieve embeddings: batch, \mathbb R^d
        head_ent_emb, rel_emb, tail_ent_emb = self.get_triple_representation(idx_triple)
        if self.discrete_points.device != head_ent_emb.device:
            self.discrete_points = self.discrete_points.to(head_ent_emb.device)
        if self.score_func == "vtp":
            h_W, h_b = self.build_func(head_ent_emb)
            r_W, r_b = self.build_func(rel_emb)
            t_W, t_b = self.build_func(tail_ent_emb)
            out = -self.trapezoid([t_W], [t_b]) * self.trapezoid([h_W, r_W], [h_b, r_b]) + self.trapezoid([r_W], [
                r_b]) * self.trapezoid([t_W, h_W], [t_b, h_b])
        elif self.score_func == "compositional":
            t_W, t_b = self.build_func(tail_ent_emb)
            chain_W, chain_b = self.build_chain_funcs([head_ent_emb, rel_emb])
            out = self.trapezoid([chain_W, t_W], [chain_b, t_b])
        elif self.score_func == "full-compositional":
            chain_W, chain_b = self.build_chain_funcs([head_ent_emb, rel_emb, tail_ent_emb])
            out = self.trapezoid([chain_W], [chain_b])
        elif self.score_func == "trilinear":
            h_W, h_b = self.build_func(head_ent_emb)
            r_W, r_b = self.build_func(rel_emb)
            t_W, t_b = self.build_func(tail_ent_emb)
            out = self.trapezoid([h_W, r_W, t_W], [h_b, r_b, t_b])
        return out


class DynamicNeuralNetworkNAS(nn.Module):
    def __init__(self, adjusted_input_dim, num_samples, num_layers, dropout_rate, k):
        super().__init__()
        self.num_layers = num_layers
        self.num_samples = num_samples
        self.dropout = nn.Dropout(0.05)
        self.k = k

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device)

        self.layer_norm = nn.LayerNorm([self.k, self.num_samples]).to(device)
        self.batch_norm = nn.BatchNorm1d(self.k).to(device)

    @staticmethod
    def identity_function(x):
        return x

    # Function to get the activation function from a string name
    def get_activation_function(self,name):
        if name == 0:
            return F.relu
        elif name == 1:
            return torch.tanh
        elif name == 2:
            return torch.sigmoid
        elif name == 3:
            return self.identity_function
        else:
            raise ValueError(f"Unknown activation function: {name}")

    def forward(self, weights: torch.FloatTensor, x):
        weights = weights.to(self.device)
        x = x.to(self.device)
        batch_size = weights.shape[0]

        if x.dim() == 2 and x.shape[0] == self.k:
            x = x.unsqueeze(0).repeat(batch_size, 1, 1)
        elif x.dim() == 3 and x.shape[0] == batch_size and x.shape[1] == self.k:
            pass
        else:
            raise ValueError(f"Unexpected shape for x: {x.shape}")

        embedding_segments = torch.chunk(weights, self.num_layers, dim=1)
        activation_functions = {}
        params = nni.get_current_parameter()
        for i, segment in enumerate(embedding_segments):
            layer_weights = segment.view(batch_size, self.k, self.k)
            # func_name = params[f"activation_function{i}"]
            # activation_functions[i] = self.get_activation_function(func_name)

            # Apply transformation using external weights
            transformed = torch.bmm(layer_weights, x)
            transformed = self.layer_norm(transformed)
            # transformed = activation_functions[i](transformed)

            # Apply activation function based on layer index
            if i % 2 == 0:
                transformed = self.activation(transformed)

            #  residual connection
            residual_connection = 1  # params["residual_connection"]
            if residual_connection == 1 and i > 0:
                x = x + transformed
            else:
                x = transformed

        return x


# class NAS(BaseKGE):
#     def __init__(self, args):
#         super().__init__(args)
#         self.name = 'NAS'
#         params = nni.get_current_parameter()
#         self.n_layers = self.args["num_layer"]
#         self.quadrature = self.args["quadrature"]
#         dropout_rate = 0  # params["dropout_rate"]
#         tuned_embedding_dim = False
#         while int(np.sqrt((self.embedding_dim) / self.n_layers)) != np.sqrt(
#                 (self.embedding_dim) / self.n_layers):
#             self.embedding_dim += 1
#             tuned_embedding_dim = True
#
#         if tuned_embedding_dim:
#             print(f"\n\n*****Embedding dimension reset to {self.embedding_dim} to fit model architecture!*****\n")
#         self.k = int(np.sqrt((self.embedding_dim) // self.n_layers))
#         self.n = self.args["num_sample"]
#         self.a, self.b = -1.0, 1.0
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#         if self.quadrature == 1:
#             discrete_points, weights = roots_legendre(self.n)
#             self.discrete_points = torch.from_numpy(discrete_points).repeat(self.k, 1).float()
#             self.weights = torch.from_numpy(weights).reshape(1, -1).float()
#         else:
#             self.discrete_points = torch.linspace(self.a, self.b, steps=self.n).repeat(self.k, 1)
#
#         self.entity_embeddings = torch.nn.Embedding(self.num_entities, self.embedding_dim).to(device)
#         self.relation_embeddings = torch.nn.Embedding(self.num_relations, self.embedding_dim).to(device)
#         self.param_init(self.entity_embeddings.weight.data), self.param_init(self.relation_embeddings.weight.data)
#         self.neuralNetwork = DynamicNeuralNetworkNAS(self.embedding_dim, self.n, self.n_layers, dropout_rate,
#                                                      self.k).to(device)
#
#     def forward_triples(self, idx_triple: torch.Tensor) -> torch.Tensor:
#         # idx_triple = idx_triple.to(self.entity_embeddings.weight.device)
#         head_ent_emb, rel_ent_emb, tail_ent_emb = self.get_triple_representation(idx_triple)
#
#         self.discrete_points = self.discrete_points.to(head_ent_emb.device)
#         self.weights = self.weights.to(head_ent_emb.device)
#         h_x = self.neuralNetwork(weights=head_ent_emb, x=self.discrete_points)
#         t_x = self.neuralNetwork(weights=tail_ent_emb, x=self.discrete_points)
#         r_x = self.neuralNetwork(weights=rel_ent_emb, x=self.discrete_points)
#         r_h_x = self.neuralNetwork(weights=rel_ent_emb, x=h_x)
#
#         if self.args["scoring_func"] == "compositional":
#             if self.quadrature == 0:
#                 score = torch.trapezoid(r_h_x * t_x, self.discrete_points, dim=2).mean(dim=1)
#             else:
#                 # Assuming r_h_x * t_x computes the function values at the quadrature points
#                 function_values = r_h_x * t_x  # Shape might be [batch_size, k, n] or similar
#                 # Calculate the integral using Gaussian quadrature
#                 integral = torch.sum(function_values, dim=1) * self.weights
#                 score = torch.mean(integral, dim=1)
#
#         if self.args["scoring_func"] == "vtp":
#             termA = - torch.trapz(t_x, self.discrete_points, dim=2).mean(dim=1)
#             termB = torch.trapz(h_x * r_x, self.discrete_points, dim=2).mean(dim=1)
#             termC = torch.trapz(r_x, self.discrete_points, dim=2).mean(dim=1)
#             termD = torch.trapz(t_x * h_x, self.discrete_points, dim=2).mean(dim=1)
#             score = termA * termB + termC * termD
#
#         if self.args["scoring_func"] == "trilinear":
#             score = torch.trapezoid(h_x * t_x * r_x, self.discrete_points, dim=2).mean(dim=1)
#
#         return score
#
#     def forward_k_vs_all(self, x: torch.Tensor) -> torch.Tensor:
#         # (1) Retrieve embeddings for head entities and relations
#         head_ent_emb, rel_ent_emb = self.get_head_relation_representation(x)
#
#         # (2) Move discrete points to the correct device
#         self.discrete_points = self.discrete_points.to(head_ent_emb.device)
#
#         # (3) Compute NNs for head and relation embeddings on discrete points
#         h_x = self.neuralNetwork(weights=head_ent_emb, x=self.discrete_points)
#         r_x = self.neuralNetwork(weights=rel_ent_emb, x=self.discrete_points)
#         r_h_x = self.neuralNetwork(weights=rel_ent_emb, x=h_x)
#
#         # (4) Use all entity embeddings as potential tails
#         all_tails = self.entity_embeddings.weight
#         all_tails_x = self.neuralNetwork(weights=all_tails, x=self.discrete_points)
#
#         scores = torch.trapz(r_h_x.unsqueeze(1) * all_tails_x.unsqueeze(0), self.discrete_points, dim=3).mean(dim=2)
#
#         return scores
