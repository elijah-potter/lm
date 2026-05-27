RUSTFLAGS="-C target-cpu=native" cargo run --release -- train \
     ./dolma \
     test.tar.gz \
     0.1 \
     0.8 \
     26 \
     1280 \
     20 \
     5120 \
     256 \
     model-512m.mpk
