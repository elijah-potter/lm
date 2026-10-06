use std::fs::File;
use std::path::PathBuf;

use burn::tensor::{Device, DeviceKind};
use clap::Parser;

use lm_core::{Model, ModelConfig, TrainingConfig};

#[derive(Debug, Parser)]
enum Args {
    Train {
        model_config: PathBuf,
        training_config: PathBuf,
    },
    Infer {
        model_path: PathBuf,
        model_config: PathBuf,
        text: String,
    },
}

fn main() {
    let args = Args::parse();
    let device = Device::wgpu(DeviceKind::DiscreteGpu(0));

    match args {
        Args::Train {
            model_config,
            training_config,
        } => {
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
            text,
        } => {
            let model_config_file = File::open(model_config).unwrap();
            let model_config: ModelConfig = serde_json::from_reader(model_config_file).unwrap();

            let mut model = Model::new(model_config, &device, false);
            let mut model_file = File::open(model_path).unwrap();
            model.load_from_weights(&mut model_file);

            model.completion(&text);
        }
    }
}
