RUSTFLAGS="-C target-cpu=native" cargo run --release -- train train.tar.gz test.tar.gz 0.1 0.8 24 256 8 1024 48 model.mpk
