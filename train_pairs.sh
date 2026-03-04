tar -xzf ./train_pairs.tar.gz
tar -xzf ./test_pairs.tar.gz
RUSTFLAGS="-C target-cpu=native" cargo run --release -- train train_pairs test_pairs 0.1 0.8 24 256 8 1024 48 pairs.mpk
