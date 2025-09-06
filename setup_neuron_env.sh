#!/usr/bin/env bash
set -euo pipefail

# Non-interactive apt
export DEBIAN_FRONTEND=noninteractive

# Load distro release info to get VERSION_CODENAME
. /etc/os-release

echo "Configuring AWS Neuron APT repository..."
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | sudo gpg --dearmor -o /usr/share/keyrings/neuron-keyring.gpg
sudo sed -i 's|deb https://apt.repos.neuron.amazonaws.com|deb [signed-by=/usr/share/keyrings/neuron-keyring.gpg] https://apt.repos.neuron.amazonaws.com|' /etc/apt/sources.list.d/neuron.list

echo "Updating apt package index..."
sudo apt-get update -y

echo "Installing prerequisites (headers, git, curl, wget, g++)..."
sudo apt-get install -y \
  linux-headers-"$(uname -r)" \
  git \
  curl \
  wget \
  g++

echo "Installing AWS Neuron drivers, runtime, and tools..."
sudo apt-get install -y aws-neuronx-dkms=2.* || echo "aws-neuronx-dkms already installed"
sudo apt-get install -y aws-neuronx-collectives || echo "aws-neuronx-collectives already installed"
sudo apt-get install -y aws-neuronx-tools=2.* || echo "aws-neuronx-tools already installed"

# Add Neuron tools to PATH for the current session
export PATH=/opt/aws/neuron/bin:$PATH

echo "Installing Python 3.9..."
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update -y
sudo apt-get install -y python3.9 python3.9-venv

echo "Creating Python 3.9 virtual environment..."
python3.9 -m venv aws_neuron_venv_pytorch_inf1

echo "Activating virtual environment..."
source aws_neuron_venv_pytorch_inf1/bin/activate
python -m pip install -U pip

echo "Installing Jupyter notebook kernel..."
pip install ipykernel 
python3.9 -m ipykernel install --user --name aws_neuron_venv_pytorch_inf1 --display-name "Python (torch-neuron)"
pip install jupyter notebook
pip install environment_kernels

echo "Setting pip repository pointing to the Neuron repository..."
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

echo "Installing PyTorch Neuron..."
python -m pip install torch-neuron neuron-cc[tensorflow] "protobuf" torchvision

echo "All done. Activate the environment with: source aws_neuron_venv_pytorch_inf1/bin/activate"
