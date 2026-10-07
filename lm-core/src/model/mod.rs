use std::io::{Read, Write};

use burn::Tensor;
use burn::lr_scheduler::linear::LinearLrSchedulerConfig;
use burn::module::Module;
use burn::nn::transformer::TransformerEncoderConfig;
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::optim::AdamConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::prelude::Device;
use burn::store::{BurnpackStore, ModuleRecord, ModuleSnapshot};
use burn::tensor::activation::softmax;
use burn::tensor::ops::PadMode;
use burn::tensor::{Bytes, Float, Int};
use burn::train::metric::{LearningRateMetric, LossMetric};
use burn::train::{Learner, SupervisedTraining};

use self::dataloader::{BatchItem, create_dataloader};
use self::model_inner::ModelInner;
use self::tokenizer::Tokenizer;

mod dataloader;
mod model_config;
mod model_inner;
mod sampling_config;
mod tensor_utils;
mod tokenizer;
mod training_config;

pub use model_config::ModelConfig;
pub use sampling_config::SamplingConfig;
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
                resizer: LinearConfig::new(config.embedding_dimensions, tokenizer.vocab_size())
                    .init(&device),
                max_seq_len: config.max_seq_len,
                pad_idx: tokenizer.pad_id(),
            },
            config,
            device: device.clone(),
            tokenizer,
        }
    }

    pub fn load_from_weights(&mut self, weights: &mut impl Read) {
        let mut data = Vec::new();
        weights.read_to_end(&mut data).unwrap();

        let bytes = Bytes::from_bytes_vec(data);
        let mut store = BurnpackStore::from_bytes(Some(bytes));

        self.inner.load_from(&mut store).unwrap();
    }

    pub fn completion(&mut self, text: &str, sampling_config: &SamplingConfig) {
        let mut idxs = self.tokenizer.text_to_tensor(text, self.device.clone());

        loop {
            let [batch_size, cur_len] = idxs.dims();
            assert_eq!(batch_size, 1);

            let padded_idxs = if cur_len >= self.config.max_seq_len {
                let start = cur_len - self.config.max_seq_len;
                idxs.clone().slice_dim(1, start..cur_len)
            } else {
                let pad_right = (self.config.max_seq_len).saturating_sub(cur_len);

                idxs.clone().pad(
                    [(0, pad_right)],
                    PadMode::Constant(self.tokenizer.pad_id() as f32),
                )
            };

            let item = BatchItem {
                inputs: padded_idxs.clone(),
                // It doesn't matter at inference time.
                targets: padded_idxs,
            };

            let output = self.inner.forward(item);

            let next_tok = Self::sample(output.output, cur_len, sampling_config);

            let next_tok_tens = Tensor::<2, Int>::from_data([[next_tok]], &self.device);
            idxs = Tensor::cat(vec![idxs, next_tok_tens], 1);

            std::io::stdout().write_all(&self.tokenizer.indices_to_bytes(&[next_tok]));
            std::io::stdout().flush();
        }
    }

    /// Samples from the logits emitted by the model. Needs to know the previous len to know which
    /// sequence position to sample from.
    fn sample(
        logits: Tensor<2, Float>,
        sample_nth_token: usize,
        sampling_config: &SamplingConfig,
    ) -> u32 {
        let softmax = softmax(logits, 1);

        let mut next_token_probs = softmax
            .slice_dim(0, sample_nth_token - 1..sample_nth_token)
            .squeeze::<1>();

        let vocab_size = next_token_probs.dims()[0];
        // A sorted list of token IDs with their associated probabilities, with the most likely
        // tokens first.
        let mut token_probability_pairs: Vec<(u32, f64)> = Vec::with_capacity(vocab_size);

        for i in 0..vocab_size {
            token_probability_pairs.push((
                // Index of the next highest probability token.
                i as u32,
                // The probability of that token.
                next_token_probs.clone().slice([i..i + 1]).into_scalar(),
            ));
        }

        token_probability_pairs.sort_by(|(_, a), (_, b)| b.total_cmp(a));

        // Apply top_p.
        let mut cumulative = 0.0;
        token_probability_pairs.retain(|(_, p)| {
            let low_before = cumulative < sampling_config.top_p;
            cumulative += *p;

            low_before
        });

        // Keep them as probabilities
        let sum: f64 = token_probability_pairs.iter().map(|(_, p)| p).sum();
        for (_, p) in token_probability_pairs.iter_mut() {
            *p = *p / sum;
        }

        // Actually sample
        let l = rand::random_range(0.0..=1.0);
        let mut cumulative_prob = 0.0;

        for (tok_id, prob) in &token_probability_pairs {
            cumulative_prob += prob;

            if cumulative_prob >= l {
                return *tok_id;
            }
        }

        *token_probability_pairs
            .first()
            .map(|(tok_id, _)| tok_id)
            .unwrap()
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
        self.inner = result.model.valid();
    }
}
