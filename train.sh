RUSTFLAGS="-C target-cpu=native" cargo run --release -- train train.tar.gz test.tar.gz 0.1 0.8 16 512 8 2048 48 model.mpk
