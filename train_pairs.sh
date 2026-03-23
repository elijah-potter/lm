RUSTFLAGS="-C target-cpu=native" cargo run --release -- train train_pairs.tar.gz test_pairs.tar.gz 0.1 0.8 24 256 8 1024 48 pairs.mpk
