#! /bin/bash

cd lm-cli
RUSTFLAGS="-C target-cpu=native" cargo run --release -- train model.json train.json
