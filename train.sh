RUSTFLAGS="-C target-cpu=native" cargo run --release -p lm-cli -- train dolma test.tar.gz 0.1 0.0003 24 512 8 2176 48 model.mpk 
