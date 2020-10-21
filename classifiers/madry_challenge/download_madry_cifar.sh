#!/bin/bash

git clone https://github.com/MadryLab/cifar10_challenge
cd cifar10_challenge
python fetch_model.py secret
python fetch_model.py natural
python fetch_model.py adv_trained

cd models
mv model_0 secret
mv naturally_trained natural

cd ../..
mkdir -p madry_challenge_models
mv cifar10_challenge/models madry_challenge_models/cifar
