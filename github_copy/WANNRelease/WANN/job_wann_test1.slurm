#!/bin/bash
#SBATCH --job-name=wann_mnist_test1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user="s2225387@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --mem-per-cpu=200M
#SBATCH --time=00:03:00
#SBATCH --partition=cpu-short
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1

module load Python/3.7.4-GCCcore-8.3.0
module load OpenMPI/4.0.1-GCC-8.3.0-2.32
source $HOME/python_venvs/wann_venv/bin/activate


echo "[$SHELL] #### Starting Python job"
echo "[$SHELL] ## This is $SLURM_JOB_USER and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

time python3 wann_train.py -p p/mnist256_128.json -o "128gens" -n 23
