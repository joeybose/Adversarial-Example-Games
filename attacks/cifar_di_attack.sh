#!/bin/bash

echo "Res 18 ----------------------------------------"
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch res18 --target_arch VGG16
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch res18 --target_arch res18
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch res18 --target_arch wide_resnet
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch res18 --target_arch dense121
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch res18 --target_arch googlenet


echo "DN-121 ----------------------------------------"
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch dense121 --target_arch VGG16
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch dense121 --target_arch res18
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch dense121 --target_arch wide_resnet
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch dense121 --target_arch dense121
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch dense121 --target_arch googlenet

echo "WR ----------------------------------------"
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch wide_resnet --target_arch VGG16
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch wide_resnet --target_arch res18
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch wide_resnet --target_arch wide_resnet
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch wide_resnet --target_arch dense121
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch wide_resnet --target_arch googlenet
