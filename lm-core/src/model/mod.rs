use std::io::{Read, Write};
use std::task::Wake;

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
use burn::tensor::{Bool, Bytes, Float, Int, TensorData};
use burn::train::metric::{LearningRateMetric, LossMetric};
use burn::train::{Learner, SupervisedTraining};

use self::dataloader::{BatchItem, create_dataloader};
use self::lora_weights::LoraCheckpointer;
use self::model_inner::ModelInner;
use self::tokenizer::Tokenizer;
use burn::train::checkpoint::FileCheckpointer;

mod dataloader;
mod lora;
mod lora_config;
mod lora_weights;
mod model_config;
mod model_inner;
mod sampling_config;
mod tensor_utils;
mod tokenizer;
mod training_config;

pub use lora_config::LoraConfig;
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
                .with_dropout(0.0)
                .with_norm_first(true)
                .init(&device),
                resizer: LinearConfig::new(config.embedding_dimensions, tokenizer.vocab_size())
                    .init(&device),
                lora_config: None,
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

        // Load only base parameters, even when LoRA is active.
        let config = self.inner.lora_config;
        let factors = config.map(|_| {
            self.inner
                .clone()
                .into_record_group(lora::factor_group(self.inner.transformer.layers.len()))
        });
        let mut base = lora::remove(self.inner.clone());
        base.load_from(&mut store).unwrap();
        if let Some(config) = config {
            base = lora::attach(base, config).load_record(factors.unwrap().allow_partial(true));
        }
        self.inner = base;
    }

    pub fn load_empty_lora_weights(&mut self, config: LoraConfig) {
        self.inner = lora::attach(self.inner.clone().fork(&self.device), config);
    }

    pub fn load_lora_from_weights(&mut self, weights: &mut impl Read, config: LoraConfig) {
        self.inner = lora_weights::load(weights, config, &self.inner, &self.device);
    }

    pub fn unload_lora(&mut self) {
        self.inner = lora::remove(self.inner.clone());
    }

    pub fn completion(
        &mut self,
        text: &str,
        sampling_config: &SamplingConfig,
        mut on_tok_complete: impl FnMut(&[u8]),
        tok_limit: usize,
    ) -> Vec<u8> {
        print!("{text}");
        let mut idxs = self.tokenizer.text_to_tensor(text, self.device.clone());
        let inference = self
            .inner
            .lora_config
            .map(|_| self.inner.clone().valid().materialize());
        let inner = inference.as_ref().unwrap_or(&self.inner);
        let mut ar_cache = inner.transformer.new_autoregressive_cache();

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

            let output = inner.forward_infer(truncated_idxs, &mut ar_cache);

            let next_tok = self.sample(idxs.clone(), output, cur_len, sampling_config);

            if next_tok as usize == self.tokenizer.pad_id() {
                break;
            }

            let next_tok_tens = Tensor::<2, Int>::from_data([[next_tok]], &self.device);
            idxs = Tensor::cat(vec![idxs, next_tok_tens], 1);

            let new_bytes = self.tokenizer.indices_to_bytes(&[next_tok]);
            on_tok_complete(&new_bytes);
            output_bytes.extend(new_bytes);
        }

        output_bytes
    }

    fn sample(
        &self,
        inputs: Tensor<2, Int>,
        logits: Tensor<2, Float>,
        sample_nth_token: usize,
        sampling_config: &SamplingConfig,
    ) -> u32 {
        let logits = logits.slice_dim(0, sample_nth_token - 1..sample_nth_token);

        // Perform DRY penalization
        let penalties = self.compute_dry_penalties(inputs, sampling_config);
        let penalized_logits = logits.sub(penalties);

        // Apply top_p.
        let softmaxed = softmax(penalized_logits.clone(), 1);
        let next_token_probs = softmaxed.squeeze::<1>();
        let token_probability_pairs = Self::compute_descending_probability_list(next_token_probs);

        let vocab_size = penalized_logits.dims()[1];
        let mut excluded = vec![false; vocab_size];
        let mut cumulative = 0.0;

        for &(tok_id, probability) in &token_probability_pairs {
            excluded[tok_id as usize] = cumulative >= sampling_config.top_p;
            cumulative += probability;
        }

        let mask =
            Tensor::<2, Bool>::from_data(TensorData::new(excluded, [1, vocab_size]), &self.device);

        let top_p_filtered = penalized_logits.mask_fill(mask, f64::NEG_INFINITY);

        // Apply temperature
        let temperature_applied = top_p_filtered.div_scalar(sampling_config.temperature);

        // Apply final softmax
        let final_probs = softmax(temperature_applied, 1).squeeze::<1>();
        let final_token_probability_pairs = Self::compute_descending_probability_list(final_probs);

        // Actually sample
        let l = rand::random_range(0.0..=1.0);
        let mut cumulative_prob = 0.0;

        for (tok_id, prob) in &final_token_probability_pairs {
            cumulative_prob += prob;

            if cumulative_prob >= l {
                return *tok_id;
            }
        }

        *final_token_probability_pairs
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

    // From an already softmaxed logit tensor, produce a sorted list of token IDs with their associated probabilities, with the most likely
    // tokens first.
    fn compute_descending_probability_list(probs: Tensor<1>) -> Vec<(u32, f64)> {
        let data: Vec<f32> = probs.to_data().try_to_vec().unwrap();
        let mut token_probability_pairs: Vec<(u32, f64)> = data
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i as u32, v as f64))
            .collect();

        token_probability_pairs.sort_by(|(_, a), (_, b)| b.total_cmp(a));
        token_probability_pairs
    }

    pub fn train(&mut self, config: TrainingConfig) {
        self.inner = self.inner.clone().fork(&self.device).train();
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
            .grads_accumulation(6)
            .num_epochs(config.checkpoints)
            .summary();

        let training = if self.inner.lora_config.is_some() {
            let directory = "./checkpoints/checkpoint";
            training.with_custom_checkpointers(
                LoraCheckpointer::new(directory, self.config.transformer_blocks),
                FileCheckpointer::new(directory, "optim"),
                FileCheckpointer::new(directory, "scheduler"),
            )
        } else {
            training.with_default_checkpointers()
        };
        let result = training.launch(Learner::new(self.inner.clone(), optim, lr_scheduler));
        self.inner = result.model.valid();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::{ModuleMapper, Param, ParamId};
    use burn::tensor::Tolerance;

    fn small_model(device: &Device) -> ModelInner {
        ModelInner {
            embedding: EmbeddingConfig::new(8, 4).init(device),
            pos_embedding: EmbeddingConfig::new(4, 4).init(device),
            transformer: TransformerEncoderConfig::new(4, 8, 2, 1)
                .with_dropout(0.0)
                .with_norm_first(true)
                .init(device),
            resizer: LinearConfig::new(4, 8).init(device),
            lora_config: None,
            max_seq_len: 4,
            pad_idx: 0,
        }
    }

    fn projections(model: &ModelInner) -> Vec<&Param<Tensor<2>>> {
        model
            .transformer
            .layers
            .iter()
            .flat_map(|layer| {
                [
                    &layer.mha.query.weight,
                    &layer.mha.key.weight,
                    &layer.mha.value.weight,
                    &layer.mha.output.weight,
                    &layer.pwff.linear_inner.weight,
                    &layer.pwff.linear_outer.weight,
                ]
            })
            .chain(std::iter::once(&model.resizer.weight))
            .collect()
    }

    /// Set only LoRA up-projection factors to nonzero values for inference tests.
    struct NonzeroFactors(Vec<ParamId>);

    impl ModuleMapper for NonzeroFactors {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
            if self.0.contains(&param.id) {
                param.map(|tensor| tensor.ones_like() * 0.1)
            } else {
                param
            }
        }
    }

    fn with_nonzero_lora(base: ModelInner) -> ModelInner {
        let model = lora::attach(
            base,
            LoraConfig {
                rank: 2,
                alpha: 2.0,
            },
        );
        let factors = projections(&model)
            .iter()
            .map(|weight| weight.adapter().unwrap().b.id)
            .collect();
        model.map(&mut NonzeroFactors(factors))
    }

    #[test]
    fn merged_lora_matches_inference_and_preserves_original_adapters() {
        let device = Device::flex();
        let model = with_nonzero_lora(small_model(&device));
        let merged = model.clone().valid().materialize();
        assert!(
            projections(&merged)
                .iter()
                .all(|weight| weight.adapter().is_none())
        );
        assert!(
            projections(&model)
                .iter()
                .all(|weight| weight.adapter().is_some())
        );
        for weight in projections(&model) {
            assert!(
                weight
                    .adapter()
                    .unwrap()
                    .delta()
                    .abs()
                    .sum()
                    .into_scalar::<f32>()
                    > 0.0
            );
        }

        let mut original_cache = model.transformer.new_autoregressive_cache();
        let mut merged_cache = merged.transformer.new_autoregressive_cache();
        for tokens in [vec![1, 2], vec![1, 2, 3]] {
            let input = Tensor::<1, Int>::from_data(tokens.as_slice(), &device).unsqueeze::<2>();
            let expected = model.forward_infer(input.clone(), &mut original_cache);
            merged
                .forward_infer(input, &mut merged_cache)
                .into_data()
                .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
        }
    }

    #[test]
    fn unloading_after_materialization_restores_base_output() {
        let device = Device::flex();
        let base = small_model(&device);
        let model = with_nonzero_lora(base.clone());
        let _merged = model.clone().valid().materialize();
        let restored = lora::remove(model);
        assert!(
            projections(&restored)
                .iter()
                .all(|weight| weight.adapter().is_none())
        );
        let input = Tensor::<2, Int>::from_data([[1, 2]], &device);
        let expected = base.forward_infer(
            input.clone(),
            &mut base.transformer.new_autoregressive_cache(),
        );
        restored
            .forward_infer(input, &mut restored.transformer.new_autoregressive_cache())
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
    }
}
