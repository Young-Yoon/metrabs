#!/bin/bash
# install docker
curl -fsSL https://get.docker.com/ | sudo sh

# install nvidia-container-toolkit to access nvidia-gpu in the docker container
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey |  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list |  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update

sudo apt-get install -y nvidia-container-runtime
sudo tee /etc/docker/daemon.json > /dev/null <<EOT
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         } 
    },
    "default-runtime": "nvidia" 
}
EOT

# restart docker
sudo systemctl restart docker
sudo chmod 666 /var/run/docker.sock

# docker build
docker build -t coreai/reality-sync:dreamfusion-latest --build-arg UID=$UID --build-arg USER_NAME=$USER --build-arg CUDA=11.6 -f Dockerfile .

