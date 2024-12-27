import os
import sys
import numpy as np
import random
import h5py
import pickle
import argparse
from tqdm import tqdm
import torch
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ["IF_CPP_MIN_LOG_LEVEL"] = '2'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rfcutils  # Assuming this module exists and is compatible

get_db = lambda p: 10 * np.log10(p)
get_pow = lambda s: np.mean(np.abs(s)**2, axis=-1)
get_sinr = lambda s, i: get_pow(s) / get_pow(i)
get_sinr_db = lambda s, i: get_db(get_sinr(s, i))

sig_len = 40960
default_n_per_batch = 100
all_sinr = np.arange(-30, 0.1, 3)

seed_number = 0

def get_soi_generation_fn(soi_sig_type):
    if soi_sig_type == 'QPSK':
        generate_soi = lambda n, s_len: rfcutils.generate_qpsk_signal(n, s_len // 16)
        demod_soi = rfcutils.qpsk_matched_filter_demod
    elif soi_sig_type == 'QAM16':
        generate_soi = lambda n, s_len: rfcutils.generate_qam16_signal(n, s_len // 16)
        demod_soi = rfcutils.qam16_matched_filter_demod
    elif soi_sig_type == 'QPSK2':
        generate_soi = lambda n, s_len: rfcutils.generate_qpsk2_signal(n, s_len // 4)
        demod_soi = rfcutils.qpsk2_matched_filter_demod
    elif soi_sig_type == 'OFDMQPSK':
        generate_soi = lambda n, s_len: rfcutils.generate_ofdm_signal(n, s_len // 80)
        _, _, _, RES_GRID = rfcutils.generate_ofdm_signal(1, sig_len // 80)
        demod_soi = lambda s: rfcutils.ofdm_demod(s, RES_GRID)
    else:
        raise Exception("SOI Type not recognized")
    return generate_soi, demod_soi


def generate_demod_testmixture(soi_type, interference_sig_type, n_per_batch=default_n_per_batch):
    generate_soi, demod_soi = get_soi_generation_fn(soi_type)

    with h5py.File(os.path.join('/scratch/general/vast/u1110463/dataset', 'interferenceset_frame', f'{interference_sig_type}_raw_data.h5'), 'r') as data_h5file:
        sig_data = np.array(data_h5file.get('dataset'))
        sig_type_info = data_h5file.get('sig_type')[()]
        if isinstance(sig_type_info, bytes):
            sig_type_info = sig_type_info.decode("utf-8")

    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)

    all_sig_mixture, all_sig1, all_bits1, meta_data = [], [], [], []
    for idx, sinr in tqdm(enumerate(all_sinr)):
        sig1, _, bits1, _ = generate_soi(n_per_batch, sig_len)
        sig2 = sig_data[np.random.randint(sig_data.shape[0], size=(n_per_batch)), :]

        sig_target = sig1[:, :sig_len]

        rand_start_idx2 = np.random.randint(sig2.shape[1] - sig_len, size=sig2.shape[0])
        inds2 = torch.tensor(rand_start_idx2).unsqueeze(-1) + torch.arange(sig_len)
        inds2 = inds2.to(torch.int64)
        sig_interference = torch.from_numpy(sig2).gather(1, inds2)

        # Interference Coefficient
        rand_gain = torch.sqrt(10**(-sinr / 10)).float()
        rand_phase = torch.rand(sig_interference.shape[0], 1)
        coeff = rand_gain * torch.exp(1j * 2 * np.pi * rand_phase)

        sig_mixture = torch.from_numpy(sig_target) + sig_interference * coeff

        all_sig_mixture.append(sig_mixture.numpy())
        all_sig1.append(sig_target)
        all_bits1.append(bits1)

        actual_sinr = get_sinr_db(sig_target, (sig_interference * coeff).numpy())
        meta_data.append(np.vstack((
            [rand_gain.numpy().real for _ in range(n_per_batch)],
            [sinr for _ in range(n_per_batch)],
            actual_sinr,
            [soi_type for _ in range(n_per_batch)],
            [interference_sig_type for _ in range(n_per_batch)]
        )))

    all_sig_mixture = np.concatenate(all_sig_mixture, axis=0)
    all_sig1 = np.concatenate(all_sig1, axis=0)
    all_bits1 = np.concatenate(all_bits1, axis=0)
    meta_data = np.concatenate(meta_data, axis=1).T

    with open(os.path.join('/scratch/general/vast/u1110463/dataset', f'Training_Dataset_{soi_type}_{interference_sig_type}.pkl'), 'wb') as f:
        pickle.dump((all_sig_mixture, all_sig1, all_bits1, meta_data), f, protocol=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Synthetic Dataset')
    parser.add_argument('-b', '--n_per_batch', default=100, type=int, help='Number of samples per batch')
    parser.add_argument('--random_seed', default=0, type=int, help='Random seed for reproducibility')
    parser.add_argument('--soi_sig_type', required=True, help='Type of signal of interest (SOI)')
    parser.add_argument('--interference_sig_type', required=True, help='Type of interference signal')

    args = parser.parse_args()

    soi_type = args.soi_sig_type
    interference_sig_type = args.interference_sig_type

    generate_demod_testmixture(soi_type, interference_sig_type, args.n_per_batch)