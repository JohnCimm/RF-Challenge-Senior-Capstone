import os
import sys
import h5py
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import rfcutils  # Assuming this module exists and is compatible

get_db = lambda p: 10 * np.log10(p)
get_pow = lambda s: np.mean(np.abs(s)**2)
get_sinr = lambda s, i: get_pow(s) / get_pow(i)
get_sinr_db = lambda s, i: get_db(get_sinr(s, i))

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def get_soi_generation_fn(soi_sig_type):
    if soi_sig_type == 'QPSK':
        generate_soi = lambda n, s_len: rfcutils.generate_qpsk_signal(n, s_len // 16)
    elif soi_sig_type == 'QAM16':
        generate_soi = lambda n, s_len: rfcutils.generate_qam16_signal(n, s_len // 16)
    elif soi_sig_type == 'QPSK2':
        generate_soi = lambda n, s_len: rfcutils.generate_qpsk2_signal(n, s_len // 4)
    elif soi_sig_type == 'OFDMQPSK':
        generate_soi = lambda n, s_len: rfcutils.generate_ofdm_signal(n, s_len // 80)
    elif soi_sig_type == 'CommSignal2':
        with h5py.File(os.path.join('dataset', 'interferenceset_frame', f'{soi_sig_type}_raw_data.h5'), 'r') as data_h5file:
            commsignal2_data = np.array(data_h5file.get('dataset'))
        def generate_commsignal2_signal(n, s_len):
            sig1 = commsignal2_data[np.random.randint(commsignal2_data.shape[0], size=(n)), :]
            rand_start_idx1 = np.random.randint(sig1.shape[1] - s_len, size=sig1.shape[0])
            inds1 = torch.tensor(rand_start_idx1).unsqueeze(-1) + torch.arange(s_len)
            sig_target = torch.gather(torch.from_numpy(sig1), 1, inds1)
            return sig_target, None, None, None  # Returning dummy values for consistency with rfcutils functions
        generate_soi = generate_commsignal2_signal
    else:
        raise Exception("SOI Type not recognized")
    return generate_soi

def generate_dataset(sig_data, soi_type, interference_sig_type, sig_len, n_examples, n_per_batch, foldername, seed, verbosity):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generate_soi = get_soi_generation_fn(soi_type)

    n_batches = int(np.ceil(n_examples / n_per_batch))
    for idx in tqdm(range(n_batches), disable=not bool(verbosity)):
        sig1, _, _, _ = generate_soi(n_per_batch, sig_len)
        sig2 = sig_data[np.random.randint(sig_data.shape[0], size=(n_per_batch)), :]

        sig_target = sig1[:, :sig_len]

        rand_start_idx2 = np.random.randint(sig2.shape[1] - sig_len, size=sig2.shape[0])
        inds2 = torch.tensor(rand_start_idx2).unsqueeze(-1) + torch.arange(sig_len)
        sig_interference = torch.gather(torch.from_numpy(sig2), 1, inds2)

        # Interference Coefficient
        rand_sinr_db = -36 * torch.rand(sig_interference.shape[0], 1) + 3
        rand_gain = 10 ** (-0.5 * rand_sinr_db / 10)
        rand_phase = torch.rand(sig_interference.shape[0], 1)
        coeff = rand_gain * torch.exp(1j * 2 * np.pi * rand_phase)

        sig_mixture = sig_target + sig_interference * coeff

        sig_mixture_comp = torch.stack((sig_mixture.real, sig_mixture.imag), dim=-1).numpy()
        sig_target_comp = torch.stack((sig_target.real, sig_target.imag), dim=-1).numpy()

        mixture_filename = f'{dataset_type}_{soi_type}_{interference_sig_type}_mixture_{idx:04}.h5'
        if not os.path.exists(os.path.join(foldername)):
            os.makedirs(os.path.join(foldername))
        with h5py.File(os.path.join(foldername, mixture_filename), 'w') as h5file0:
            h5file0.create_dataset('mixture', data=sig_mixture_comp)
            h5file0.create_dataset('target', data=sig_target_comp)
            h5file0.create_dataset('sig_type', data=f'{soi_type}_{interference_sig_type}_mixture')

        del sig1, sig2, sig_mixture_comp, sig_target_comp
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Synthetic Dataset')
    parser.add_argument('-l', '--sig_len', default=40960, type=int)
    parser.add_argument('-n', '--n_examples', default=240000, type=int, help='')
    parser.add_argument('-b', '--n_per_batch', default=4000, type=int, help='')
    parser.add_argument('-d', '--dataset', default='train', help='')
    parser.add_argument('--random_seed', default=0, type=int, help='')
    parser.add_argument('-v', '--verbosity', default=1, help='')
    parser.add_argument('--soi_sig_type', help='')
    parser.add_argument('--interference_sig_type', help='')
    args = parser.parse_args()

    soi_type = args.soi_sig_type
    interference_sig_type = args.interference_sig_type
    with h5py.File(os.path.join('/scratch/general/vast/dataset', 'interferenceset_frame', f'{interference_sig_type}_raw_data.h5'), 'r') as data_h5file:
        sig_data = np.array(data_h5file.get('dataset'))
        sig_type_info = data_h5file.get('sig_type')[()]
        if isinstance(sig_type_info, bytes):
            sig_type_info = sig_type_info.decode("utf-8")

    # Generate synthetic dataset based on input arguments
    dataset_type = args.dataset
    foldername = os.path.join('/scratch/general/vast/u1110463/dataset', f'Dataset_{soi_type}_{interference_sig_type}_Mixture')

    generate_dataset(sig_data, soi_type, interference_sig_type, args.sig_len, args.n_examples, args.n_per_batch, foldername, args.random_seed, args.verbosity)
