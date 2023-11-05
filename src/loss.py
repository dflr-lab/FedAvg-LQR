#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7


import torch


def LQRLoss(x_pred, x, u_pred, u): 
    """Compute the mean loss(tracking error and contorl error) of LQR.
    Parameters:
        x_pred: (batch_size x T x 2)
        x: (batch_size x T x 2)
        u_pred: (batch_size x T x 1)
        u: (batch_size x T x 1)
    Return:
        mean loss
    """
    mse_x = torch.mean((x_pred - x) ** 2)
    mse_u = torch.mean((u_pred - u) ** 2)
    return mse_x + mse_u


def LQRLossX(x_pred, x):
    """Compute the mean loss of tracking error for LQR.
    Parameters:
        x_pred: (batch_size x T x 2)
        x: (batch_size x T x 2)
    """
    mse_x = torch.mean((x_pred - x) ** 2)
    return mse_x


def LQRLossU(u_pred, u):
    """Compute the mean loss of contorl error for LQR.
    Parameters:
        u_pred: (batch_size x T x 1)
        u: (batch_size x T x 1)
    """
    mse_u = torch.mean((u_pred - u) ** 2)
    return mse_u