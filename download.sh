#!/bin/sh

mkdir -p ./data
cd data

echo "Download OOD datasets..."
wget https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz
wget https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz
wget https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz
wget https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz
wget https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz

echo "Unzip..."
tar -xvzf Imagenet.tar.gz
tar -xvzf Imagenet_resize.tar.gz
tar -xvzf LSUN.tar.gz
tar -xvzf LSUN_resize.tar.gz
tar -xvzf iSUN.tar.gz

cd ..

mkdir -p ./models
cd models

echo "Download DenseNet models..."
wget https://www.dropbox.com/s/wr4kjintq1tmorr/densenet10.pth.tar.gz
wget https://www.dropbox.com/s/vxuv11jjg8bw2v9/densenet100.pth.tar.gz

echo "Unzip..."
tar -xvzf densenet10.pth.tar.gz
tar -xvzf densenet100.pth.tar.gz

cd ..