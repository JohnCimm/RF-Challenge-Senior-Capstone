#!bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --account=kingspeak-gpu
#SBATCH --partition=kingspeak-gpu
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:titanx:1
#SBATCH --mem=16GB
#SBATCH -e slurm-%j.err-%N
#SBATCH -o slurm-%j.out-%N
%module unload cuda
%module load cuda/11.8
%module load cudnn/8.7.0.84-11.8-gpu
module load miniconda3/23.11.0
source /uufs/chpc.utah.edu/common/home/u1110463/miniconda3/etc/profile.d/conda.sh
source activate base


conda activate /uufs/chpc.utah.edu/common/home/u1110463/software/miniconda3/john/envs/rfsionna 
 
#tfds build icassp2024rfchallenge/dataset_utils/tfds_scripts/Dataset_QPSK_CommSignal2_Mixture.py --data_dir /scratch/general/vast/u1110463/tfds/ > 11_7_test.log
#python /uufs/chpc.utah.edu/common/home/u1110463/dataset_utils/example_generate_rfc_mixtures.py --soi_sig_type QPSK --interference_sig_type CommSignal2 --n_per_batch 1000 > 10_25_test.log
python -u /uufs/chpc.utah.edu/common/home/u1110463/rrf_test.py 0 > rrc_scipy_test1.log