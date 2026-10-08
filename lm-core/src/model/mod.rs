use std::io::{Read, Write};

use burn::Tensor;
use burn::lr_scheduler::linear::LinearLrSchedulerConfig;
use burn::module::Module;
use burn::nn::transformer::{TransformerEncoderAutoregressiveCache, TransformerEncoderConfig};
use burn::nn::{EmbeddingConfig, LinearConfig};
use burn::optim::AdamConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::prelude::Device;
use burn::store::{BurnpackStore, ModuleRecord, ModuleSnapshot};
use burn::tensor::activation::softmax;
use burn::tensor::ops::PadMode;
use burn::tensor::{Bytes, Float, Int, TensorData};
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

    /// Perform completion using the model and provided sampler settings.
    /// The provided limit is the number of tokens to be generated.
    /// The provided callback will be run after each token is generated and it will be provided the
    /// bytes from that token.
    pub fn completion(
        &mut self,
        text: &str,
        sampling_config: &SamplingConfig,
        mut on_tok_complete: impl FnMut(&[u8]),
        tok_limit: usize,
    ) -> Vec<u8> {
        print!("{text}");
        let mut idxs = self.tokenizer.text_to_tensor(text, self.device.clone());
        let mut ar_cache = self.inner.transformer.new_autoregressive_cache();

        let mut output_bytes = Vec::new();

        for _ in 0..tok_limit {
            let [batch_size, cur_len] = idxs.dims();
            assert_eq!(batch_size, 1);

            let truncated_idxs = if cur_len >= self.config.max_seq_len {
                let start = cur_len - self.config.max_seq_len;
                idxs.clone().slice_dim(1, start..cur_len)
            } else {
                idxs.clone()
            };

            let output = self.inner.forward_infer(truncated_idxs, &mut ar_cache);

            let next_tok = self.sample(idxs.clone(), output, cur_len, sampling_config);
            let next_tok_tens = Tensor::<2, Int>::from_data([[next_tok]], &self.device);
            idxs = Tensor::cat(vec![idxs, next_tok_tens], 1);

            let new_bytes = self.tokenizer.indices_to_bytes(&[next_tok]);
            on_tok_complete(&new_bytes);
            output_bytes.extend(new_bytes);
        }

        output_bytes
    }

    /// Samples from the logits emitted by the model. Needs to know the previous len to know which
    /// sequence position to sample from.
    fn sample(
        &self,
        inputs: Tensor<2, Int>,
        logits: Tensor<2, Float>,
        sample_nth_token: usize,
        sampling_config: &SamplingConfig,
    ) -> u32 {
        // Perform DRY penalization
        let penalties = self.compute_dry_penalties(inputs, sampling_config);
        let penalized_logits = logits.sub(penalties);

        let softmax = softmax(penalized_logits, 1);

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

    fn compute_dry_penalties(
        &self,
        input: Tensor<2, Int>,
        config: &SamplingConfig,
    ) -> Tensor<2, Float> {
        let input = input.into_data().try_into_vec::<i32>().unwrap();
        let last_tok_id = input.last().unwrap();

        let positions: Vec<usize> = input
            .iter()
            .enumerate()
            .filter_map(|(index, &value)| {
                if value == *last_tok_id && index != input.len() - 1 {
                    Some(index)
                } else {
                    None
                }
            })
            .collect();

        // At each position
        let mut depths = Vec::with_capacity(positions.len());

        for pos in &positions {
            let mut depth = 1;

            loop {
                let matched = if let Some(t) = input.get(pos - depth) {
                    if let Some(t_e) = input.get(input.len() - 1 - depth) {
                        t == t_e
                    } else {
                        false
                    }
                } else {
                    false
                };

                if matched {
                    depth += 1;
                } else {
                    depths.push(depth);
                    break;
                }
            }
        }

        let token_ids_to_penalize: Vec<(usize, usize)> = positions
            .iter()
            .enumerate()
            .map(|(index, i)| (input[i + 1] as usize, depths[index]))
            .collect();

        let mut penalties = Vec::with_capacity(self.tokenizer.vocab_size());

        for i in 0..self.tokenizer.vocab_size() {
            let penalty = if let Some((_, depth)) = token_ids_to_penalize
                .iter()
                .filter(|(f, _)| *f == i)
                .max_by_key(|(_, depth)| depth)
            {
                if *depth >= config.dry_allowed_length {
                    config.dry_multiplier
                        * config
                            .dry_base
                            .powi(depth.saturating_sub(config.dry_allowed_length) as i32)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            penalties.push(penalty);
        }

        Tensor::<2, Float>::from_data(
            TensorData::new(penalties, [1, self.tokenizer.vocab_size()]),
            &self.device,
        )
    }

    pub fn train(&mut self, config: TrainingConfig) {
        let train_loader = create_dataloader(
            self.device.clone(),
            config.batch_size,
            self.tokenizer.clone(),
            self.config.max_seq_len,
            config.checkpoint_period,
            config.train_dataset,
        );

        let validate_loader = create_dataloader(
            self.device.clone(),
            config.batch_size,
            self.tokenizer.clone(),
            self.config.max_seq_len,
            config.validation_duration,
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
            .num_epochs(config.checkpoints)
            .summary();

        let result = training.launch(Learner::new(self.inner.clone(), optim, lr_scheduler));
        self.inner = result.model.valid();
    }
}
