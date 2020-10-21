# NoBoxAttack

This contains sample commands to run all of our Baseline models, taken from the
original git repos but repurposed for the NoBox codebase.

## Requirements
```
wandb (latest version) \
pytorch==1.4 \
torchvision \
cudatoolkit==10.1\
PIL \
numpy \
json \
wandb \
tqdm \
matplotlib \
image \
ipdb \
[advertorch](https://github.com/BorealisAI/advertorch) \
robustml
```

# AEG-split eval on cifar (Joey):
```
ipython --pdb -- no_box_attack.py --dataset cifar --namestr="Split 2 Cifar eps 0.3" --perturb_loss Linf --epsilon=0.3 --attack_ball Linf --batch_size 128 --test_batch_size 64 --attack_epochs 150 --extragradient --lr 1e-3 --lr_model 1e-3 --max_iter 20 --pgd_on_critic --attack_loss cross_entropy --model Resnet --dir_test_models ../ --command train --source_arch wide_resnet --eval_freq 1 --transfer --train_set test --lambda_on_clean 2000 --split 2
```

# AEG-Transfer Eval on cifar (Hugo):
```
CUDA_VISIBLE_DEVICES=0 ipython --pdb -- no_box_attack.py --dataset cifar --namestr="2000 Res->VGG16 Cifar" --perturb_loss Linf --epsilon=0.03125 --attack_ball Linf --batch_size 512 --test_batch_size 64 --attack_epochs 150 --extragradient --lr 1e-3 --lr_model 1e-3 --max_iter 20 --pgd_on_critic --attack_loss cross_entropy --model Resnet --dir_test_models /checkpoint/hberard/NoBoxAttack/transfer_eval --command train --source_arch res18  --eval_freq 1 --transfer --train_set test --lambda_on_clean 2000 --target_arch VGG16 wide_resnet dense121 googlenet res18 --num_eval_samples 256
``` 

# AEG-Ens Robust Eval on cifar (Gauthier):
```
python no_box_attack.py --dataset cifar --namestr="test" --perturb_loss Linf --epsilon=0.03125 --attack_ball Linf --batch_size 128 --test_batch_size 128 --attack_epochs 50 --extragradient --lr 1e-3 --lr_model 1e-3 --max_iter 10 --attack_loss cross_entropy --model Resnet --dir_test_models ../ --command train --model_name res18 --adv_models res18_ens --source_arch res18 --ensemble_adv_trained --type 3 --eval_freq 1 --transfer --train_set test --lambda_on_clean 10 --pgd_on_critic```
```

# AEG-Ens Robust Eval on MNIST (Gauthier):
```
python no_box_attack.py --dataset mnist --namestr="D-Ens Mnist eps 0.3" --perturb_loss Linf --epsilon=0.3 --attack_ball Linf --batch_size 512 --test_batch_size 64 --attack_epochs 150 --extragradient --lr 1e-3 --lr_model 1e-3 --max_iter 20 --pgd_on_critic --attack_loss cross_entropy --model CondGen --dir_test_models ../ --command train --model_name modelD --adv_models modelD_ens modelB modelC modelD_ens1 modelD modelA --source_arch ens_adv --type 3 --eval_freq 1 --transfer --train_set test --lambda_on_clean 10```
```


# AEG on madry cifar challenge:
```
python no_box_attack.py --dataset cifar --namestr="test" --perturb_loss Linf --epsilon=0.03125 --attack_ball Linf --batch_size 128 --test_batch_size 128 --attack_epochs 300 --extragradient --lr 1e-3 --lr_model 1e-3 --max_iter 20 --attack_loss cross_entropy --model Resnet --dir_test_models /checkpoint/hberard/NoBoxAttack --command train --eval_freq 1 --transfer --train_set test --lambda_on_clean 2000 --pgd_on_critic --source_arch adv
```

# Baseline

## NAttack:

# Baseline split eval (Joey):
```
python momentum_iterative_attack.py --n_iter=1 --dataset cifar --epsilon=0.03125 --transfer --source_arch wide_resnet --split 0
```

# NAttack:
```
python nattack.py --n_iter=20 --query_step=1 --dataset mnist --epsilon=0.3 --sweep --wandb
```

## AutoAttack Individual
```
python autoattack.py --n_iter=20 --query_step=1 --dataset mnist --epsilon=0.3 --sweep --wandb
```

## AutoAttack
```
python autoattack.py --n_iter=20 --query_step=1 --dataset mnist --epsilon=0.3 --wandb
```

## AutoZoom-Bilin

```
python run_autozoom.py -a autozoom_bilin -d mnist --m 10000 --switch_iterations 1000 --init_const 10
```

## MI Attack
```
python momentum_iterative_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer
```

## DIM Attack
```
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer
```

## SGM Attack
```
python sgm_attack.py --n_iter=20 --dataset cifar --transfer
```


# Plotting
Navigate to the visualization subdir `cd ../visualization` then run the
following command but remember to replace all the appropriate run_keys with the
key on Weights and Biases. You will also need a `settings.json` file with you
wandb api key.

```
python plot_wandb.py --no_box='run_key' --nattack='run_key' \
--auto_attack_ind='run_key' --dataset='mnist' --username=wandb_username
```
# Ensemble Adversarial Training Defence Evaluation
Add the following flags, `--model`, `--adv_models`, `--type` and `--source_arch='ens_adv'
For MNIST:
`python momentum_iterative_attack.py --n_iter=100 --dataset mnist
--model='modelB' --adv_models='modelB_ens' --epsilon=0.3 --transfer
--source_arch='ens_adv' --type=1
`

For CIFAR:
python momentum_iterative_attack.py --n_iter=100 --dataset cifar
--adv_models='wide_resnet_ens' --epsilon=0.03125 --transfer
--source_arch='res18' --ensemble_adv_trained

## License
[MIT](https://choosealicense.com/licenses/mit/)


