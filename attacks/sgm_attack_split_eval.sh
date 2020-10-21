#!/bin/bash

# Model A
echo "SGM-attack ----------------------------------------"
python sgm_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch res18 --split 0
python sgm_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch res18 --split 1
python sgm_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch res18 --split 2
python sgm_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch res18 --split 3
python sgm_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch res18 --split 4
python sgm_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch res18 --split 5

