#!/bin/bash

echo "source_arch VGG16 ----------------------------------------"
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch VGG16 --target_arch VGG16
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch VGG16 --target_arch res18
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch VGG16 --target_arch wide_resnet
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch VGG16 --target_arch dense121
python diverse_input_attack.py --n_iter=100 --dataset cifar --epsilon=0.03125 --transfer --source_arch VGG16 --target_arch googlenet

