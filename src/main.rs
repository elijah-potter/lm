#![recursion_limit = "256"]
mod batcher;
mod dataset;
mod generation;
mod model;
mod tokenizer;
mod training;

use std::path::PathBuf;
use std::time::Instant;

use burn::backend::{NdArray, Wgpu};
use burn::module::Module;
use burn::record::{CompactRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use clap::Parser;

use self::generation::generate_tokens;
use self::model::{ModelConfig, ModelRecord};

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
        temperature: f64,
        load_from: PathBuf,
        context: String,
    },
    Train {
        train_data: PathBuf,
        test_data: PathBuf,
        dropout: f64,
        /// Base learning-rate factor for the Noam scheduler.
        #[clap(default_value_t = 0.1)]
        lr_factor: f64,
        /// The total number of transformer blocks.
        transformer_blocks: usize,
        embed_dims: usize,
        /// The number of attention heads to use in the transformer.
        attn_heads: usize,
        /// The size of the perceptrons between the attention blocks.
        percept_size: usize,
        epochs: usize,
        #[arg(short, long)]
        start_model: Option<PathBuf>,
        #[arg(long)]
        start_optimizer: Option<PathBuf>,
        save_to: PathBuf,
    },
}

type TrainingBackend = Wgpu<f32, i32>;
type InferenceBackend = NdArray<f32, i32>;

fn main() {
    let command = Command::parse();

    match command {
        Command::Train {
            train_data,
            test_data,
            dropout,
            lr_factor,
            transformer_blocks,
            embed_dims,
            attn_heads,
            percept_size,
            epochs,
            start_model,
            start_optimizer,
            save_to,
        } => {
            let record = if let Some(path) = start_model {
                let device = Default::default();
                NamedMpkFileRecorder::<FullPrecisionSettings>::new()
                    .load(path.into(), &device)
                    .expect("Should be able to load the model weights from the provided file")
            } else {
                None
            };

            let model = training::train::<TrainingBackend>(
                ModelConfig::new(transformer_blocks, embed_dims, attn_heads, percept_size)
                    .with_dropout(dropout),
                train_data,
                test_data,
                epochs,
                lr_factor,
                record,
                start_optimizer,
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
            temperature,
            percept_size,
        } => {
            let device = Default::default();

            let record: ModelRecord<InferenceBackend> =
                NamedMpkFileRecorder::<FullPrecisionSettings>::new()
                    .load(load_from.into(), &device)
                    .expect("Should be able to load the model weights from the provided file");

            // Initialize a new model with the loaded record/weights
            let mut model =
                ModelConfig::new(transformer_blocks, embed_dims, attn_heads, percept_size)
                    .init(&device);
            model = model.load_record(record);

            let ctx_chars: Vec<_> = context.chars().collect();

            generate_tokens(&model, &ctx_chars, temperature);
        }
    }
}
