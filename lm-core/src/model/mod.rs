use burn::lr_scheduler::linear::LinearLrSchedulerConfig;
use burn::module::Module;
use burn::nn::transformer::TransformerEncoderConfig;
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::optim::AdamConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::prelude::Device;
use burn::train::metric::{LearningRateMetric, LossMetric};
use burn::train::{Learner, SupervisedTraining};

use self::dataloader::create_dataloader;
use self::model_inner::ModelInner;
use self::tokenizer::Tokenizer;

mod dataloader;
mod model_config;
mod model_inner;
mod tensor_utils;
mod tokenizer;
mod training_config;

pub use model_config::ModelConfig;
pub use training_config::TrainingConfig;

pub struct Model {
    inner: ModelInner,
    config: ModelConfig,
    device: Device,
    tokenizer: Tokenizer,
}

impl Model {
    pub fn new(config: ModelConfig, device: &Device, prepare_for_training: bool) -> Self {
        let tokenizer = Tokenizer::new();

        let device = if prepare_for_training {
            device.clone().autodiff()
        } else {
            device.clone()
        };

        Self {
            inner: ModelInner {
                embedding: EmbeddingConfig::new(
                    tokenizer.vocab_size(),
                    config.embedding_dimensions,
                )
                .init(&device),
                pos_embedding: EmbeddingConfig::new(
                    config.max_seq_len,
                    config.embedding_dimensions,
                )
                .init(&device),
                transformer: TransformerEncoderConfig::new(
                    config.embedding_dimensions,
                    config.hidden_size,
                    config.attention_heads,
                    config.transformer_blocks,
                )
                .init(&device),
                max_seq_len: config.max_seq_len,
                pad_idx: tokenizer.pad_id(),
            },
            config,
            device: device.clone(),
            tokenizer,
        }
    }

    pub fn train(&mut self, config: TrainingConfig) {
        let train_loader = create_dataloader(
            self.device.clone(),
            config.batch_size,
            self.tokenizer.clone(),
            self.config.max_seq_len,
            config.train_dataset,
        );

        let validate_loader = create_dataloader(
            self.device.clone(),
            config.batch_size,
            self.tokenizer.clone(),
            self.config.max_seq_len,
            config.validate_dataset,
        );

        let optim = AdamConfig::new()
            .with_beta_1(config.optim_beta_1 as f32)
            .with_beta_2(config.optim_beta_2 as f32)
            .with_epsilon(config.optim_epsilon as f32)
            .with_weight_decay(Some(WeightDecayConfig::new(
                config.optim_weight_decay as f32,
            )))
            .init();

        let lr_scheduler = LinearLrSchedulerConfig::new(
            config.learning_rate / 1000.0,
            config.learning_rate,
            config.lr_warmup_duration,
        )
        .init()
        .unwrap();

        let training = SupervisedTraining::new("./checkpoints", train_loader, validate_loader)
            .metric_train_numeric(LossMetric::new())
            .metric_valid_numeric(LossMetric::new())
            .metric_train_numeric(LearningRateMetric::new())
            .with_default_checkpointers()
            .grads_accumulation(6)
            .num_epochs(config.epochs)
            .summary();

        let result = training.launch(Learner::new(self.inner.clone(), optim, lr_scheduler));
        result.model.valid();

        self.inner = result.model;
    }
}
