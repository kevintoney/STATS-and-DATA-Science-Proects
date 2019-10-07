#!/bin/sh
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=124000
#SBATCH --job-name=mastersProj
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --error=mastersProj.stderr
#SBATCH --output=mastersProj.stdout

module purge
module load anaconda
conda activate mypy36
python dogsVGG16HCC.py output.out