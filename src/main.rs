#![recursion_limit = "256"]
mod batcher;
mod dataset;
mod generation;
mod model;
mod tokenizer;
mod training;

use std::path::PathBuf;

use burn::backend::Wgpu;
use burn::module::Module;
use burn::record::{CompactRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use clap::Parser;

use self::generation::generate;
use self::model::{Model, ModelConfig, ModelRecord};

#[derive(Parser, Debug)]
enum Command {
    Generate {
        /// The total number of transformer blocks.
        transformer_blocks: usize,
        embed_dims: usize,
        /// The number of attention heads to use in the transformer.
        attn_heads: usize,
        /// The size of the perceptrons between the attention blocks.
        percept_size: usize,
        load_from: PathBuf,
        context: String,
    },
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

type TargetBackend = Wgpu<f32, i32>;

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
            let model = training::train::<TargetBackend>(
                ModelConfig::new(transformer_blocks, embed_dims, attn_heads, percept_size)
                    .with_dropout(dropout),
                train_data,
                test_data,
            );

            let recorder = CompactRecorder::new();
            model.save_file(save_to, &recorder).unwrap();
        }

        Command::Generate {
            load_from,
            context,
            transformer_blocks,
            embed_dims,
            attn_heads,
            percept_size,
        } => {
            let device = Default::default();

            let record: ModelRecord<TargetBackend> =
                NamedMpkFileRecorder::<FullPrecisionSettings>::new()
                    .load(load_from.into(), &device)
                    .expect("Should be able to load the model weights from the provided file");

            // Initialize a new model with the loaded record/weights
            let mut model =
                ModelConfig::new(transformer_blocks, embed_dims, attn_heads, percept_size)
                    .init(&device);
            model = model.load_record(record);

            generate(&model, context.as_str())
        }
    }
}
