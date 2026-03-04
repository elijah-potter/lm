tar -xzf ./train.tar.gz --no-same-owner
tar -xzf ./test.tar.gz --no-same-owner

RUSTFLAGS="-C target-cpu=native" cargo run --release -- train train test 0.1 0.8 24 256 8 1024 48 model.mpk
