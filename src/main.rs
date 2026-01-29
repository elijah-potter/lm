#![recursion_limit = "256"]
mod batcher;
mod dataset;
mod model;
mod tokenizer;
mod training;

use std::path::PathBuf;

use burn::module::Module;
use burn::record::CompactRecorder;
use clap::Parser;

use self::model::ModelConfig;

#[derive(Parser, Debug)]
enum Command {
    Train {
        train_data: PathBuf,
        test_data: PathBuf,
        dropout: f64,
        /// The total number of transformer blocks.
        transformer_blocks: usize,
        embed_dims: usize,
        /// The number of attention heads to use in the transformer.
        attn_heads: usize,
        /// The size of the perceptrons between the attention blocks.
        percept_size: usize,
        save_to: PathBuf,
    },
}

fn main() {
    let command = Command::parse();

    match command {
        Command::Train {
            train_data,
            test_data,
            dropout,
            transformer_blocks,
            embed_dims,
            attn_heads,
            percept_size,
            save_to,
        } => {
            let model = training::train(
                ModelConfig::new(transformer_blocks, embed_dims, attn_heads, percept_size)
                    .with_dropout(dropout),
                train_data,
                test_data,
            );

            let recorder = CompactRecorder::new();
            model.save_file(save_to, &recorder).unwrap();
        }
    }
}
