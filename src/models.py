#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7


import torch
from torch import nn
import torch.nn.functional as F


class DynamicsModel(nn.Module):
    """Dynamics Model of 1D Mass Point, LQR."""
    def __init__(self, A, B, K, device='cuda'):
        super(DynamicsModel, self).__init__()
        self.device = device

        # Define parameters
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32)) # (2 x 2)
        self.B = nn.Parameter(torch.tensor(B, dtype=torch.float32)) # (2 x 1)
        self.K = nn.Parameter(torch.tensor(K, dtype=torch.float32)) # (1 x 2)

        self.to(device)  # Move entire model to the device

    def forward(self, initial_states, T=30):
        """
        Parameters:
            initial_state: (batch_size x 2), initial state of mass of mass point
            T: time steps
        Return:
        x_pred: (batch_size x T x 2), state(position, velocity) of mass point
        u_pred: (batch_size x T x 1), control of mass point
        """
        batch_size = initial_states.shape[0]
        # Initialize the predictions list with the initial states at the first time step
        x_pred_list = [initial_states] # T x batch_size x 2
        u_pred_list = [] # T x batch_size x 1

        # Loop over time steps
        for t in range(T):
            x_t = torch.stack([x_pred_list[-1][b] for b in range(batch_size)], dim=0) # batch_size x 2
            u_t = -torch.matmul(self.K, x_t.unsqueeze(-1)).squeeze(-1) # self.K: 1 x 2, x_t.unsqueeze(-1): batch_size x 2 x 1, u_t: batch_size x 1
            u_pred_list.append(u_t)

            if t < T - 1:  # No need to compute the next state after the last time step
                next_x = torch.matmul(self.A, x_t.unsqueeze(-1)).squeeze(-1) + \
                         torch.matmul(self.B, u_t.unsqueeze(-1)).squeeze(-1)
                x_pred_list.append(next_x)

        # Concatenate the list of tensors along the time dimension
        x_pred = torch.stack(x_pred_list, dim=1) # batch_size x T x 2
        u_pred = torch.stack(u_pred_list, dim=1) # batch_size x T x 1

        return x_pred, u_pred