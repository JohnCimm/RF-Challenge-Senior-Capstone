#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --account=kingspeak-gpu
#SBATCH --partition=kingspeak-gpu
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:titanx:1
#SBATCH --mem=16GB
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N

$ module --ignore_cache load "miniconda3/24.9.2"
source /uufs/chpc.utah.edu/common/home/u1110463/software/miniconda3/john/etc/profile.d/conda.sh

source activate base


conda activate rfsionna 
 
 
%tfds build icassp2024rfchallenge/dataset_utils/tfds_scripts/Dataset_QPSK_CommSignal2_Mixture.py --data_dir /scratch/general/vast/u1110463/tfds/ > 11_7_test.log
%python /uufs/chpc.utah.edu/common/home/u1110463/dataset_utils/example_generate_rfc_mixtures.py --soi_sig_type QPSK --interference_sig_type CommSignal2 --n_per_batch 1000 > 10_25_test.log
python -u /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/train_unet_model.py 0 > optimization_alldata.log

#python -u /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/train_torchwavenet.py  1 > wavenet_test2.log
#python -u /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/train_torchwavenet.py  2 > wavenet_test2.log






### GENERATE INITIAL DATASETS

 #python /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_generate_rfc_mixtures.py --soi_sig_type QPSK --interference_sig_type EMISignal1 --n_per_batch 1000 > torch_dataset1.log
 #python /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_generate_rfc_mixtures.py --soi_sig_type QPSK --interference_sig_type CommSignal3 --n_per_batch 1000 > torch_dataset3.log
 #python /uufs/chpc.utah.edu/common/home/u1110463/dataset_utils/example_generate_rfc_mixtures.py --soi_sig_type QPSK --interference_sig_type CommSignal5G1 --n_per_batch 1000 > torch_dataset4.log


 #python /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_generate_rfc_mixtures.py --soi_sig_type OFDMQPSK --interference_sig_type EMISignal1 --n_per_batch 1000 > torch_dataset5.log
 #python /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_generate_rfc_mixtures.py --soi_sig_type OFDMQPSK --interference_sig_type CommSignal2 --n_per_batch 1000 > torch_dataset6.log
 #python /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_generate_rfc_mixtures.py --soi_sig_type OFDMQPSK --interference_sig_type CommSignal3 --n_per_batch 1000 > torch_dataset7.log
 #python /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_generate_rfc_mixtures.py --soi_sig_type OFDMQPSK --interference_sig_type CommSignal5G1 --n_per_batch 1000 > torch_dataset8.log

### GENERATE PYTORCH DATASETS FROM INTIAL DATASET 
#python  /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_preprocess_npy_dataset.py QPSK_CommSignal2  > npn_CommSignal.log
# python  /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_preprocess_npy_dataset.py QPSK_CommSignal3 > npn_CommSignal.log
# python  /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_preprocess_npy_dataset.py QPSK_CommSignal5G1 > npn_CommSignal.log
# python  /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_preprocess_npy_dataset.py QPSK_EMISignal1 > npn_CommSignal.log

# python  /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_preprocess_npy_dataset.py OFDMQPSK_CommSignal2 > npn_CommSignal.log
# python  /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_preprocess_npy_dataset.py OFDMQPSK_CommSignal3 > npn_CommSignal.log
# python  /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_preprocess_npy_dataset.py OFDMQPSK_CommSignal5G1 > npn_CommSignal.log
# python  /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_preprocess_npy_dataset.py OFDMQPSK_EMISignal1 > npn_CommSignal.log

###  GENERATE TENSORFLOW UNET DATASET
# tfds build icassp2024rfchallenge/dataset_utils/tfds_scripts/Dataset_QPSK_CommSignal2_Mixture.py --data_dir /scratch/general/vast/u1110463/tfds/ > tensorflow_datasets.log
# tfds build icassp2024rfchallenge/dataset_utils/tfds_scripts/Dataset_QPSK_CommSignal3_Mixture.py --data_dir /scratch/general/vast/u1110463/tfds/ > tensorflow_datasets.log
# tfds build icassp2024rfchallenge/dataset_utils/tfds_scripts/Dataset_QPSK_CommSignal5G1_Mixture.py --data_dir /scratch/general/vast/u1110463/tfds/ > tensorflow_datasets.log
# tfds build icassp2024rfchallenge/dataset_utils/tfds_scripts/Dataset_QPSK_EMISignal1_Mixture.py --data_dir /scratch/general/vast/u1110463/tfds/ > tensorflow_datasets.log

# tfds build icassp2024rfchallenge/dataset_utils/tfds_scripts/Dataset_OFDMQPSK_CommSignal2_Mixture.py --data_dir /scratch/general/vast/u1110463/tfds/ > tensorflow_datasets.log
# tfds build icassp2024rfchallenge/dataset_utils/tfds_scripts/Dataset_OFDMQPSK_CommSignal3_Mixture.py --data_dir /scratch/general/vast/u1110463/tfds/ > tensorflow_datasets.log
# tfds build icassp2024rfchallenge/dataset_utils/tfds_scripts/Dataset_OFDMQPSK_CommSignal5G1_Mixture.py --data_dir /scratch/general/vast/u1110463/tfds/ > tensorflow_datasets.log
# tfds build icassp2024rfchallenge/dataset_utils/tfds_scripts/Dataset_OFDMQPSK_EMISignal1_Mixture.py --data_dir /scratch/general/vast/u1110463/tfds/ > tensorflow_datasets.log
## Example RF Challenge Datasets

# Create training set examples similar to TestSet from the Grand Challenge specifications
#python /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_generate_competition_trainmixture.py --soi_sig_type QPSK --interference_sig_type EMISignal1 --n_per_batch 1000 < rf_challenge_data.log
#python /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_generate_competition_trainmixture.py --soi_sig_type QPSK --interference_sig_type CommSignal2 --n_per_batch 1000 < rf_challenge_data.log
#python /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_generate_competition_trainmixture.py --soi_sig_type QPSK --interference_sig_type CommSignal3 --n_per_batch 1000 < rf_challenge_data.log
#python /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_generate_competition_trainmixture.py --soi_sig_type QPSK --interference_sig_type CommSignal5G1 --n_per_batch 1000 < rf_challenge_data.log

#python /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_generate_competition_trainmixture.py --soi_sig_type OFDMQPSK --interference_sig_type EMISignal1 --n_per_batch 1000 < rf_challenge_data.log
#python /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_generate_competition_trainmixture.py --soi_sig_type OFDMQPSK --interference_sig_type CommSignal2 --n_per_batch 1000 < rf_challenge_data.log
#python /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_generate_competition_trainmixture.py --soi_sig_type OFDMQPSK --interference_sig_type CommSignal3 --n_per_batch 1000 < rf_challenge_data.log
#python /uufs/chpc.utah.edu/common/home/u1110463/icassp2024rfchallenge/dataset_utils/example_generate_competition_trainmixture.py --soi_sig_type OFDMQPSK --interference_sig_type CommSignal5G1 --n_per_batch 1000 < rf_challenge_data.log

