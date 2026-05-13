#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --mem=35G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk
#SBATCH --output=logs/qwen_%j.log

export HF_HOME=/mnt/nfs/homes/ranasint/hf_home
huggingface-cli login --token

python -m llm_zero.qwen_3



