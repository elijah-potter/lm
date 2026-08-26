#! /bin/bash

apt update -y
apt install curl build-essential vim htop -y
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
