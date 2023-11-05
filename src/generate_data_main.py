#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7


import os
import numpy as np
from dataset import DataGenerator, load_and_print_data


def main():
    datagenerator = DataGenerator()
    datasets, K_matrices = datagenerator.generate_dataset()

    if not os.path.exists('../data'):
        os.makedirs('../data')

    # Save A matrix and B matrix
    np.save('../data/matrix_A.npy', datagenerator.A)
    np.save('../data/matrix_B.npy', datagenerator.B)

    # save data of each robot
    for i, dataset in enumerate(datasets): # for each robot
        # separate the x and u data, and convert them into NumPy arrays
        x_data = np.array([data[0] for data in dataset]) # num_initial_states x T x 2
        u_data = np.array([data[1] for data in dataset]) # num_initial_states x T x 1

        # save x and u
        np.save(f'../data/robot_{i+1}_x_data.npy', x_data)
        np.save(f'../data/robot_{i+1}_u_data.npy', u_data)

    # save K matrices of all the robots
    K_matrices_array = np.array(K_matrices)
    np.save('../data/K_matrices.npy', K_matrices_array)


if __name__ == "__main__":
    main()
    load_and_print_data()