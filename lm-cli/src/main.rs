use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use burn::tensor::{Device, DeviceKind, FloatDType};
use clap::Parser;

use lm_core::{Model, ModelConfig, SamplingConfig, TrainingConfig};

#[derive(Debug, Parser)]
enum Args {
    Train {
        model_config: PathBuf,
        training_config: PathBuf,
    },
    Infer {
        model_path: PathBuf,
        model_config: PathBuf,
        sampling_config: PathBuf,
        text: String,
    },
}

fn main() {
    let args = Args::parse();

    match args {
        Args::Train {
            model_config,
            training_config,
        } => {
            let mut device = Device::cuda(0);
            device.configure(FloatDType::BF16).unwrap();

            let model_config_file = File::open(model_config).unwrap();
            let model_config: ModelConfig = serde_json::from_reader(model_config_file).unwrap();

            let train_config_file = File::open(training_config).unwrap();
            let train_config: TrainingConfig = serde_json::from_reader(train_config_file).unwrap();
            let mut model = Model::new(model_config, &device, true);
            model.train(train_config);
        }
        Args::Infer {
            model_path,
            model_config,
            sampling_config,
            text,
        } => {
            let mut device = Device::flex();
            device.configure(FloatDType::BF16).unwrap();

            let model_config_file = File::open(model_config).unwrap();
            let model_config: ModelConfig = serde_json::from_reader(model_config_file).unwrap();

            let mut model = Model::new(model_config, &device, false);
            let mut model_file = File::open(model_path).unwrap();
            model.load_from_weights(&mut model_file);

            let sampling_config_file = File::open(sampling_config).unwrap();
            let sampling_config: SamplingConfig =
                serde_json::from_reader(sampling_config_file).unwrap();

            let mut stdout = std::io::stdout();

            model.completion(
                &text,
                &sampling_config,
                |bytes| {
                    stdout.write_all(bytes).unwrap();
                    stdout.flush();
                },
                128,
            );
        }
    }
}
