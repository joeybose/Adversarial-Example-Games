#!/bin/bash

git clone https://github.com/MadryLab/mnist_challenge
cd mnist_challenge
python fetch_model.py secret
python fetch_model.py natural
python fetch_model.py adv_trained
cd ..
mkdir -p madry_challenge_models
mv mnist_challenge/models madry_challenge_models/mnist
