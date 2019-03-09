'''
Pre-training DNNs with Noisy Data
the DNNs contains one Gaussian-Bernoulli RBM and two Bernoulli-Bernoulli RBMs
'''

import numpy as np
import tensorflow as tf
from tfrbm import BBRBM, GBRBM
import os
import scipy.io as scio

def prepare_data(file_path):
    all_data = np.zeros([1,257])
    #all_data = np.array(all_data)
    file_names = os.listdir(file_path)
    for name in file_names:
        full_path = file_path + name
        data = scio.loadmat(full_path)
        data = np.array(data['htkdata']).transpose()
        all_data = np.vstack((all_data, data))

    return all_data[1:all_data.shape[0]]

#prepare the noisy data set
noisy_data_path = "DataSet/train/noisy_speech/"
input_data = prepare_data(file_path = noisy_data_path)

#begin pretraining Gaussian-Bernoulli RBM
gb_n_visible = input_data.shape[1]
gb_n_hid = 2048
gb_learning_rate=0.01
gb_momentum=0.95
gb_err_function='mse'
sigma=1

gbrbm = GBRBM(n_visible = gb_n_visible, n_hidden = gb_n_hid, learning_rate = gb_learning_rate, momentum = gb_momentum, err_function = gb_err_function, use_tqdm=False, sample_visible=True, sigma = sigma)

gb_n_epoches = 40
gb_batch_size=128

errs = gbrbm.fit(data_x = input_data, n_epoches = gb_n_epoches, batch_size = gb_batch_size, shuffle=True, verbose=True)

gb_filename = 'pretrain_models/gbrbm.ckpt'
gb_name = 'rbm'
gbrbm.save_weights(filename = gb_filename, name = gb_name)

#begin pretraining the first  Bernoulli-Bernoulli RBM
bb_input_data_1 = gbrbm.transform(input_data)
bb_input_data_1 = np.array(bb_input_data_1)
#print(bb_input_data_1.shape)

bb_n_visible_1 = bb_input_data_1.shape[1]
bb_n_hid_1 = 2048
bb_learning_rate_1 = 0.01
bb_momentum_1 = 0.95
bb_err_function_1 = 'mse'

bbrbm_1 = BBRBM(n_visible = bb_n_visible_1, n_hidden = bb_n_hid_1, learning_rate = bb_learning_rate_1, momentum = bb_momentum_1, err_function = bb_err_function_1, use_tqdm=False)

bb_n_epoches_1 = 10
bb_batch_size_1 = 128

errs_1 = bbrbm_1.fit(data_x = bb_input_data_1, n_epoches = bb_n_epoches_1, batch_size = bb_batch_size_1, shuffle=True, verbose=True)

bb_filename_1 = 'pretrain_models/bbrbm_1.ckpt'
bb_name_1 = 'rbm'
bbrbm_1.save_weights(filename = bb_filename_1, name = bb_name_1)

#begin pretraining the second  Bernoulli-Bernoulli RBM
bb_input_data_2 = bbrbm_1.transform(bb_input_data_1)
bb_input_data_2 = np.array(bb_input_data_2)

bb_n_visible_2 = bb_input_data_2.shape[1]
bb_n_hid_2 = 2048
bb_learning_rate_2 = 0.01
bb_momentum_2 = 0.95
bb_err_function_2 = 'mse'

bbrbm_2 = BBRBM(n_visible = bb_n_visible_2, n_hidden = bb_n_hid_2, learning_rate = bb_learning_rate_2, momentum = bb_momentum_2, err_function = bb_err_function_2, use_tqdm=False)

bb_n_epoches_2 = 5
bb_batch_size_2 = 128

errs_2 = bbrbm_1.fit(data_x = bb_input_data_2, n_epoches = bb_n_epoches_2, batch_size = bb_batch_size_2, shuffle=True, verbose=True)

bb_filename_2 = 'pretrain_models/bbrbm_2.ckpt'
bb_name_2 = 'rbm'
bbrbm_2.save_weights(filename = bb_filename_2, name = bb_name_2)
