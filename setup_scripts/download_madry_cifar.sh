#!/bin/bash

root_dir=$1
cd $root_dir

echo "===Cloning Madry===="
git clone https://github.com/MadryLab/cifar10_challenge
cd cifar10_challenge
python fetch_model.py secret
python fetch_model.py natural
python fetch_model.py adv_trained
cd ..
mkdir -p madry_challenge_models
mv cifar10_challenge/models madry_challenge_models/cifar
mv madry_challenge_models/cifar/model_0 madry_challenge_models/cifar/secret
mv cifar10_challenge/ classifiers/madry_challenge/
echo "===DONE===="
