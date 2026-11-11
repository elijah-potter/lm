use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use burn::tensor::{Device, DeviceKind, FloatDType};
use clap::Parser;

use lm_core::{LoraConfig, Model, ModelConfig, SamplingConfig, TrainingConfig};

#[derive(Debug, Parser)]
enum Args {
    Train {
        model_config: PathBuf,
        training_config: PathBuf,
        base_model: Option<PathBuf>,
        #[arg(long)]
        lora_config: Option<PathBuf>,
        #[arg(long, requires = "lora_config")]
        lora_weight: Option<PathBuf>,
    },
    Infer {
        model_path: PathBuf,
        model_config: PathBuf,
        sampling_config: PathBuf,
        text: String,
        #[arg(long, requires = "lora_weight")]
        lora_config: Option<PathBuf>,
        #[arg(long, requires = "lora_config")]
        lora_weight: Option<PathBuf>,
    },
}

fn main() {
    let args = Args::parse();

    match args {
        Args::Train {
            model_config,
            training_config,
            base_model,
            lora_config,
            lora_weight,
        } => {
            let mut device = Device::cuda(0);

            let model_config_file = File::open(model_config).unwrap();
            let model_config: ModelConfig = serde_json::from_reader(model_config_file).unwrap();

            let train_config_file = File::open(training_config).unwrap();
            let train_config: TrainingConfig = serde_json::from_reader(train_config_file).unwrap();

            let mut model = Model::new(model_config, &device, true);

            if let Some(base_model) = base_model {
                let mut model_file = File::open(base_model).unwrap();
                model.load_from_weights(&mut model_file);
            }

            if let Some(config) = lora_config {
                let config: LoraConfig =
                    serde_json::from_reader(File::open(config).unwrap()).unwrap();
                match lora_weight {
                    Some(path) => {
                        model.load_lora_from_weights(&mut File::open(path).unwrap(), config)
                    }
                    None => model.load_empty_lora_weights(config),
                }
            }
            model.train(train_config);
        }
        Args::Infer {
            model_path,
            model_config,
            sampling_config,
            text,
            lora_config,
            lora_weight,
        } => {
            let mut device = Device::flex();

            let model_config_file = File::open(model_config).unwrap();
            let model_config: ModelConfig = serde_json::from_reader(model_config_file).unwrap();

            let mut model = Model::new(model_config, &device, false);
            let mut model_file = File::open(model_path).unwrap();
            model.load_from_weights(&mut model_file);

            if let (Some(config), Some(weights)) = (lora_config, lora_weight) {
                let config: LoraConfig =
                    serde_json::from_reader(File::open(config).unwrap()).unwrap();
                model.load_lora_from_weights(&mut File::open(weights).unwrap(), config);
            }

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
