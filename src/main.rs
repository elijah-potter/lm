#![recursion_limit = "256"]
mod batcher;
mod dataset;
mod tokenizer;

use crate::batcher::BatchItem;
use burn::backend::{Autodiff, NdArray};
use burn::train::metric::CudaMetric;
use burn::train::{InferenceStep, Learner};
use burn::{
    backend::Wgpu,
    data::{dataloader::DataLoaderBuilder, dataset::transform::SamplerDataset},
    lr_scheduler::noam::NoamLrSchedulerConfig,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Relu,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    optim::{AdamConfig, decay::WeightDecayConfig},
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        ClassificationOutput, SupervisedTraining, TrainOutput, TrainStep,
        metric::{AccuracyMetric, LearningRateMetric, LossMetric, PerplexityMetric},
    },
};
use log::info;

use self::{batcher::GenBatcher, dataset::FileFolderDataset};

const VOCAB_SIZE: usize = 64;
const MAX_SEQ_LEN: usize = 64;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    dropout: Dropout,
    activation: Relu,
    char_embedding: Embedding<B>,
    pos_embedding: Embedding<B>,
    transformer: TransformerEncoder<B>,
}

impl<B: Backend> Model<B> {
    /// Embed a given string, simply skipping any incompatible tokens.
    /// Will panic if the provided string is too long.
    ///
    /// This is used to create the model's context.
    fn embed(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_batches, len] = input.dims();
        assert_eq!(len, MAX_SEQ_LEN);

        let tok_embedding: Tensor<B, 3> = self.char_embedding.forward(input);

        let pos_tensor_indices = Tensor::<B, 2, Int>::from_data(
            TensorData::new(
                (0i32..len as i32).collect::<Vec<_>>(),
                Shape::new([1, MAX_SEQ_LEN]),
            ),
            &self.device(),
        );
        let pos_embedding = self.pos_embedding.forward(pos_tensor_indices);

        (tok_embedding + pos_embedding) / 2
    }

    fn device(&self) -> B::Device {
        let devices = self.devices();
        devices[0].clone()
    }

    fn forward_train(
        &self,
        input: Tensor<B, 2, Int>,
        target: Tensor<B, 2, Int>,
    ) -> ClassificationOutput<B> {
        let embedding = self.embed(input);

        let [batch_size, seq_length, embedding_dims] = embedding.dims();
        assert_eq!(seq_length, MAX_SEQ_LEN);

        let output = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding));

        let loss_fn = CrossEntropyLossConfig::new().init(&self.device());

        let output_flat = output
            .clone()
            .reshape([batch_size * seq_length, embedding_dims]);
        let target_flat = target.reshape([batch_size * seq_length]);

        let loss = loss_fn.forward(output_flat.clone(), target_flat.clone());

        ClassificationOutput::new(loss, output_flat, target_flat)
    }
}

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = BatchItem<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: BatchItem<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_train(item.input, item.target);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> InferenceStep for Model<B> {
    type Input = BatchItem<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: BatchItem<B>) -> ClassificationOutput<B> {
        self.forward_train(item.input, item.target)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "0.5")]
    dropout: f64,
    /// The total number of transformer blocks.
    transformer_blocks: usize,
    embed_dims: usize,
    /// The number of attention heads to use in the transformer.
    attn_heads: usize,
    /// The size of the perceptrons between the attention blocks.
    percept_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: Relu::new(),
            char_embedding: EmbeddingConfig::new(VOCAB_SIZE, self.embed_dims).init(device),
            pos_embedding: EmbeddingConfig::new(MAX_SEQ_LEN, self.embed_dims).init(device),
            transformer: TransformerEncoderConfig::new(
                self.embed_dims,
                self.percept_size,
                self.attn_heads,
                self.transformer_blocks,
            )
            .init(device),
        }
    }
}

fn main() {
    type TargetBackend = Autodiff<Wgpu<f32, i32>>;

    let device = Default::default();
    let model = ModelConfig::new(6, 256, 8, 1024)
        .with_dropout(0.1)
        .init::<TargetBackend>(&device);

    let dataset_train = FileFolderDataset::load_from_folder("./train");
    let dataset_test = FileFolderDataset::load_from_folder("./test");

    let dataloader_train = DataLoaderBuilder::new(GenBatcher)
        .batch_size(32)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_train, 10_000));

    let dataloader_test = DataLoaderBuilder::new(GenBatcher)
        .batch_size(32)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_test, 1000));

    let optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1.0e-6)))
        .init();

    let accum = 6;

    let lr_scheduler = NoamLrSchedulerConfig::new(0.01 / accum as f64)
        .with_warmup_steps(6000)
        .with_model_size(512)
        .init()
        .unwrap();

    let training = SupervisedTraining::new("./checkpoints", dataloader_train, dataloader_test)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(AccuracyMetric::new().with_pad_token(0))
        .metric_valid_numeric(AccuracyMetric::new().with_pad_token(0))
        .metric_train_numeric(PerplexityMetric::new().with_pad_token(0))
        .metric_valid_numeric(PerplexityMetric::new().with_pad_token(0))
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .grads_accumulation(accum)
        .num_epochs(128)
        .summary();

    info!("Selected device: {:?}", device);

    let result = training.launch(Learner::new(model, optim, lr_scheduler));

    let recorder = CompactRecorder::new();
    result.model.save_file("model", &recorder).unwrap();
}
