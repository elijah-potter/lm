RUSTFLAGS="-C target-cpu=native" cargo run --release -- train train.tar.gz test.tar.gz 0.1 0.04 19 512 8 2176 48 model.mpk
