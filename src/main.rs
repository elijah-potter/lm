#![recursion_limit = "256"]
mod batcher;
mod dataset;
mod model;
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

use self::model::ModelConfig;
use self::{batcher::GenBatcher, dataset::FileFolderDataset};

fn main() {
    const VOCAB_SIZE: usize = 64;
    const MAX_SEQ_LEN: usize = 64;
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

    // let recorder = CompactRecorder::new();
    // result.model.save_file("model", &recorder).unwrap();
}
