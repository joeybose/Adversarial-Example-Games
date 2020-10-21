#!/bin/bash

# Model A
echo "Model A ----------------------------------------"
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelA' --adv_models='modelA_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=0
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelA' --adv_models='modelA_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=0
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelA' --adv_models='modelA_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=0
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelA' --adv_models='modelA_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=0
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelA' --adv_models='modelA_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=0

# Model B
echo "Model B ----------------------------------------"
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelB' --adv_models='modelB_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=1
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelB' --adv_models='modelB_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=1
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelB' --adv_models='modelB_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=1
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelB' --adv_models='modelB_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=1
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelB' --adv_models='modelB_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=1

# Model C
echo "Model C ----------------------------------------"
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelC' --adv_models='modelC_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=2
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelC' --adv_models='modelC_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=2
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelC' --adv_models='modelC_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=2
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelC' --adv_models='modelC_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=2
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelC' --adv_models='modelC_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=2

# Model D
echo "Model D ----------------------------------------"
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelD' --adv_models='modelD_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=3
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelD' --adv_models='modelD_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=3
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelD' --adv_models='modelD_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=3
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelD' --adv_models='modelD_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=3
python momentum_iterative_attack.py --n_iter=100 --dataset mnist --model='modelD' --adv_models='modelD_ens' --epsilon=0.3 --transfer --source_arch='ens_adv' --type=3
