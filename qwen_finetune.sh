#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --mem=39G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk
#SBATCH --output=logs/qwen_finetune_%j.log

export HF_HOME=/mnt/nfs/homes/ranasint/hf_home
huggingface-cli login --token

python -m llm_finetune.qwen_2 --language eng --num-samples 1000 --epochs 3



