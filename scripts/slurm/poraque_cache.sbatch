#!/bin/bash
# file: poraque_cache.sbatch
#
# Build the dataset cache on a CPU node, and stop.
#
# The cache is NumPy work -- reading every CHGCAR/TAUCAR, computing V_ext,
# spectrally downsampling, writing -- and on a cold parallel filesystem it is
# minutes to hours during which a GPU allocation would sit idle. So the two
# halves of a training run are two jobs: this one on a CPU partition, then
# scripts/slurm/poraque_ddp.sbatch on the GPU partition with the SAME config,
# which finds the cache and rebuilds nothing.
#
#   cache=$(sbatch --parsable scripts/slurm/poraque_cache.sbatch configs/train.yaml)
#   sbatch --dependency=afterok:${cache} scripts/slurm/poraque_ddp.sbatch configs/train.yaml
#
# One task, on purpose. The cache builder is single-process; several tasks
# would each build the whole cache (the writes are atomic, so nothing tears,
# but every task does the full work). Memory should cover one native-grid
# material decoded in float64 plus the downsampled copy -- a 108^3 density is
# ~10 MB, so the request below is generous rather than tight.
#
#SBATCH --job-name=poraque-cache
#SBATCH --partition=sequana_cpu     # a CPU partition; adjust to the cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # one process. Not four.
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-cache-%j.out
#SBATCH --error=slurm-cache-%j.err

set -euo pipefail

CONFIG="${1:-configs/train.yaml}"

# Same environment discipline as the GPU job: the run banner prints where
# torch was imported from, and this makes the answer be the environment.
export PYTHONNOUSERSITE=1
module load python/3.11 2>/dev/null || true
source "${CONDA_PREFIX_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate poraque

# The spectral downsampling is FFT work; let NumPy's BLAS/FFT use the cores
# this task was given.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# --cache-only forces the CPU and forms no process group whatever the config
# says, so the config can carry `training.device: cuda` for the GPU job that
# follows. It exits 0 after a summary naming the cache, the material count per
# task, the grid shapes, and the size on disk and decoded in RAM -- read the
# last one before sizing the GPU job's memory request.
poraque-train --config "${CONFIG}" --cache-only
