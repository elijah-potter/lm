#! /bin/bash

set -eo pipefail

cargo clean

tar --zstd -cf snapshot.tar.zst *
