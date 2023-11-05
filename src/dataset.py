#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.linalg import solve_discrete_are


class SyntheticDataset(Dataset):
    """Synthetic Dataset of mass point."""
    def __init__(self, robot_id, idxs=None, data_path='../data'):
        self.x_data = np.load(f'{data_path}/robot_{robot_id}_x_data.npy')
        self.u_data = np.load(f'{data_path}/robot_{robot_id}_u_data.npy')
        self.idxs = [i for i in range(len(self.x_data))] if idxs is None else idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        x = self.x_data[self.idxs[item]]
        u = self.u_data[self.idxs[item]]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(u, dtype=torch.float32)


class DataGenerator:
    """A data generator."""
    def __init__(self) -> None:
        # Define the parameters of the dynamics system
        self.A = np.array(
            [[1, 1], 
            [0, 1]]
        )
        self.B = np.array([[0], [1]])

        # Define the LQR weight matrices
        self.Q = np.eye(2)  # Identity matrix
        self.R_values = [1, 10, 100]  # Different R values correspond to different robots and different R values represent personalization

        self.num_robots = 3
        self.num_initial_states = 50
        self.T = 30
        self.state_noise_variance = 0.01
        self.control_noise_variance = 0.01  # Noise variance

    def lqr_gain(self, A, B, Q, R):
        """compute the Linear Quadratic Regulator (LQR) gain matrix K.
        Parameters:
            A: (2 x 2)
            B: (2 x 1)
            Q: (2 x 2), identity matrix
            R: (1 x 1)
        Return:
            K: (1 x 2)
        """
        # Computing the Discrete-Time Algebraic Riccati Equation
        P = solve_discrete_are(A, B, Q, R)
        # Compute the Linear Quadratic Regulator (LQR) gain
        K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
        return K

    def simulate_system(self, K, initial_state):
        """Given K and initial state, the function returns the values of
           x and u at T discrete time instances.
        Parameters:
            K: (1 x 2)
            initial_state: [postion, velocity],
        Return:
            x: (T x 2)
            u: (T x 1)
        """
        x = np.zeros((self.T, 2))
        u = np.zeros((self.T, 1))
        x[0] = initial_state

        for t in range(self.T):
            u[t] = -K @ x[t]  # Compute t-th control by u = -Kx
            control_noise = np.random.normal(0, np.sqrt(self.control_noise_variance)) 
            state_noise = np.random.normal(0, np.sqrt(self.state_noise_variance), (2,)) 

            if t < self.T - 1:
                x[t + 1] = self.A @ x[t] + self.B @ (u[t] + control_noise) + state_noise  # Compute (t + 1)-th state

            # If the system reaches a steady state (with position and velocity approaching zero), the simulation is terminated prematurely.
            # if np.allclose(x[t + 1], [0, 0], atol=1e-2):
            #     break

        return x, u

    def generate_dataset(self,):
        """Generate datasets for all robots.
        Postion value range: [-300,-75].
        Velocity value range: [5, 20].
        Return:
            datasets: (num_robots x num_initial_states x (x, u)), x:(T x 2), u:(T x 1)
            K_matrices: (num_robots x 1 x 2)
        """
        datasets = [] # num_robots x num_initial_states x (x, u), x = T x 2, u = T x 1
        K_matrices = []  # (num_robots x 1 x 2)

        for k in range(self.num_robots):
            R = self.R_values[k]  # Set R for k-th robot
            K = self.lqr_gain(self.A, self.B, self.Q, R)  # Compute K matrix
            K_matrices.append(K)
            robot_data = []

            for _ in range(self.num_initial_states):
                initial_v = random.uniform(5, 20) #  Velocity value range: [5, 20]
                initial_state = np.array([-15*initial_v, initial_v]) # Postion value range: [-300,-75]
                x, u = self.simulate_system(K, initial_state)
                robot_data.append((x, u))

            datasets.append(robot_data)

        return datasets, K_matrices
    
def load_and_print_data():
    data_directory = '../data'

    if not os.path.exists(data_directory):
        print(f"Directory does not exist: {data_directory}")
        return

    matrix_A = np.load(os.path.join(data_directory, 'matrix_A.npy'))
    matrix_B = np.load(os.path.join(data_directory, 'matrix_B.npy'))
    print("Matrix A:")
    print(matrix_A)
    print("\nMatrix B:")
    print(matrix_B)

    K_matrices = np.load(os.path.join(data_directory, 'K_matrices.npy'))
    print("\nK matrices:")
    print(K_matrices)

    num_robots = 3
    for i in range(1, num_robots + 1):
        x_data = np.load(os.path.join(data_directory, f'robot_{i}_x_data.npy'))
        u_data = np.load(os.path.join(data_directory, f'robot_{i}_u_data.npy'))

        print(f"\nRobot {i} x data:")
        print(x_data)
        print(x_data.shape)
        print(f"\nRobot {i} u data:")
        print(u_data)
        print(u_data.shape)
