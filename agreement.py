# # import json
# # import numpy as np
# #
# # # File path to the JSON data
# # file_path = "C:\\Users\\vikrant.singh\\Downloads\\akshay_annotated.json"
# #
# # # Load JSON data from the file
# # with open(file_path, 'r') as file:
# #     json_data = json.load(file)
# #
# # # Extracting scores and converting them to integers, handling empty strings
# # scores = []
# # for key,item in json_data.items():  # Iterate over dictionary items
# #     if 'verbalized' in item and 'annotated' in item['verbalized'] and 'score' in item['verbalized']['annotated']:
# #         score = item['verbalized']['annotated']['score']
# #         # if isinstance(score, str) and score.isdigit():  # Check if the score is a digit and is a string
# #         #     scores.append(score)
# #         # else:
# #         if score:
# #             scores.append(int(score))
# #
# # # Ensuring there are scores to calculate statistics
# # if scores:
# #     mean_score = np.mean(scores)
# #     median_score = np.median(scores)
# #     mode_score = max(set(scores), key=scores.count)
# #     std_deviation = np.std(scores)
# # else:
# #     mean_score = median_score = mode_score = std_deviation = None
# #
# # print("Mean Score:", mean_score, "Median Score:", median_score, "Mode Score:", mode_score, "Standard Deviation:", std_deviation)
#
#
# from sympy import symbols, integrate, diff, Heaviside, Max
# import torch.nn.functional as F
# import torch
# # Define symbols for the 3D vectors
# h1, h2, h3, r1, r2, r3, t1, t2, t3, t1_prime, t2_prime, t3_prime, x, gamma = symbols('h1 h2 h3 r1 r2 r3 t1 t2 t3 t1_prime t2_prime t3_prime x gamma')
#
# # Define simplified linear functions for f_h(x), f_r(x), and f_t(x) in 3D
# f_h_x = 1 * x + 2 * x*x + 3*x*x*x
# f_r_x = 2 * x + 3 * x*x + 2*x*x*x
# f_t_x = 5 * x + 7 * x*x + 5*x*x*x
# f_t_prime_x = 2 * x + 3 * x*x + 2*x*x*x
#
# # Define the VTP scoring function for 3D vectors
# def vtp_scoring_function_3d(ft_x, fh_x, fr_x):
#     integral_ft = integrate(ft_x, (x, 0, 1))
#     integral_fh_fr = integrate(fh_x * fr_x, (x, 0, 1))
#     integral_fr = integrate(fr_x, (x, 0, 1))
#     integral_ft_fh = integrate(ft_x * fh_x, (x, 0, 1))
#
#     score_vtp = -integral_ft * integral_fh_fr + integral_fr * integral_ft_fh
#     score_trilinear = integrate(ft_x * fh_x * fr_x ,(x, 0, 1) )
#     return score_vtp
#
# # Calculate the VTP scores for positive and negative triples in 3D
# vtp_positive_3d = vtp_scoring_function_3d(f_t_x, f_h_x, f_r_x).__float__()
# vtp_negative_3d = vtp_scoring_function_3d(f_t_prime_x, f_h_x, f_r_x).__float__()
#
# # Define the margin-based loss function for 3D vectors
#
# # Assign sample values for 3D vectors
# # sample_values_3d = {
# #     h1: 1, h2: 2, h3: 3,  # h vector
# #     r1: 21, r2: 3, r3: 2,  # r vector
# #     t1: 3, t2: 5, t3: 4,  # t vector
# #     t1_prime: 1.2, t2_prime: 1.7, t3_prime: 2.2,  # t' vector
# #     gamma: 1  # Margin value
# # }
# print(vtp_positive_3d, vtp_negative_3d)
# margin_loss_3d = Max(0, 1 - vtp_positive_3d + vtp_negative_3d).__float__()
#
# print(margin_loss_3d)
# # Calculate the gradients of the margin loss with respect to h, r, and t in 3D
# grad_loss_h1_3d = diff(margin_loss_3d, h1).simplify()
# grad_loss_h2_3d = diff(margin_loss_3d, h2).simplify()
# grad_loss_h3_3d = diff(margin_loss_3d, h3).simplify()
# grad_loss_r1_3d = diff(margin_loss_3d, r1).simplify()
# grad_loss_r2_3d = diff(margin_loss_3d, r2).simplify()
# grad_loss_r3_3d = diff(margin_loss_3d, r3).simplify()
# grad_loss_t1_3d = diff(margin_loss_3d, t1).simplify()
# grad_loss_t2_3d = diff(margin_loss_3d, t2).simplify()
# grad_loss_t3_3d = diff(margin_loss_3d, t3).simplify()
# # loss = F.margin_ranking_loss(vtp_positive_3d, vtp_negative_3d, torch.tensor([1.0]))
#
# print(grad_loss_h1_3d, grad_loss_h2_3d,
#       grad_loss_h3_3d, grad_loss_r1_3d,
#       grad_loss_r2_3d, grad_loss_r3_3d,
#       grad_loss_t1_3d, grad_loss_t2_3d,
#       grad_loss_t3_3d
# )


import torch

# Define the dimension and create a linspace tensor
n_dimension = 4
x = torch.linspace(0, 1, n_dimension, requires_grad=True)

# Define random embeddings for h, r, t, and t'
h = torch.randn(n_dimension, requires_grad=True)
r = torch.randn(n_dimension, requires_grad=True)
t = torch.randn(n_dimension, requires_grad=True)
t1 = torch.randn(n_dimension, requires_grad=True)
t2 = torch.randn(n_dimension, requires_grad=True)
t3 = torch.randn(n_dimension, requires_grad=True)

t_prime = torch.randn(n_dimension, requires_grad=True)

# Non-linear functions
def f_h(x, h):
    return h * x ** torch.arange(1, n_dimension + 1)

def f_r(x, r):
    return r * x ** torch.arange(1, n_dimension + 1)

def f_t(x, t):
    return t * x ** torch.arange(1, n_dimension + 1)

# VTP Scoring function with integration
def vtp_scoring(h, r, t):
    A = torch.trapz(f_t(x, t), x)
    B = torch.trapz(f_h(x, h) * f_r(x, r), x)
    C = torch.trapz(f_r(x, r), x)
    D = torch.trapz(f_t(x, t) * f_h(x, h), x)
    # trilinear = torch.trapz(f_t(x, t) * f_h(x, h) * f_r(x, r), x)
    return -A * B + C * D

# Compute the scores for positive and negative samples
score_positive = vtp_scoring(h, r, t2)
score_negative = vtp_scoring(h, r, t_prime)

# Margin-based loss function
gamma = 1.0
loss = torch.max(torch.tensor(0.0), gamma + score_negative - score_positive)
print(loss)

# Compute gradients
loss.backward()

# Gradients with respect to h, r, and t
grad_h = h.grad
grad_r = r.grad
grad_t = t2.grad

print("Gradient w.r.t h:", grad_h)
print("Gradient w.r.t r:", grad_r)
print("Gradient w.r.t t:", grad_t)
