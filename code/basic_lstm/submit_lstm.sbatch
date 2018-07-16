#!/bin/bash

#SBATCH --job-name=gen-lstm
#SBATCH -p owners,gpu
#SBATCH --gres gpu:1
#SBATCH --time 24:00:00
#SBATCH -C GPU_BRD:TESLA

module load py-keras/2.0.8_py27
module load py-tensorflow/1.8.0_py27

python ./wonderland_train.py
