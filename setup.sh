#!/bin/bash
root_dir=$(pwd)
#echo "creates the conda env"
#conda create -n NoBox
#conda activate NoBox
#echo "Install the requirements"
#conda install python==3.7
#conda install pip
#pip install tensorflow-gpu=1.14

echo "Install advertorch\n"
cd $root_dir
git clone https://github.com/BorealisAI/advertorch
cd advertorch
python setup.py install
cd ..
rm -rf advertorch

echo "Get Madry CIFAR\n"
cd $root_dir
bash setup_scripts/download_madry_cifar.sh $root_dir

echo "Get Madry Mnist\n"
cd $root_dir
bash setup_scripts/download_madry_mnist.sh $root_dir
