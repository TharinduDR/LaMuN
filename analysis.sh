#!/bin/bash
#SBATCH --partition=a2000-48h
#SBATCH --mem=10G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk
#SBATCH --output=logs/analysis_%j.log

export HF_HOME=/mnt/nfs/homes/ranasint/hf_home
huggingface-cli login --token

python -m llm_zero.lamun_analysis



