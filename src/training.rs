use std::path::Path;

use burn::backend::Autodiff;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::transform::SamplerDataset;
use burn::lr_scheduler::noam::NoamLrSchedulerConfig;
use burn::module::Module;
use burn::optim::AdamConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::prelude::Backend;
use burn::record::CompactRecorder;
use burn::train::metric::{
    AccuracyMetric, CudaMetric, LearningRateMetric, LossMetric, PerplexityMetric,
};
use burn::train::{Learner, SupervisedTraining};
use log::info;

use crate::batcher::GenBatcher;
use crate::dataset::FileFolderDataset;
use crate::model::{Model, ModelConfig, ModelRecord};

pub fn train<B: Backend>(
    m: ModelConfig,
    train_folder: impl AsRef<Path>,
    test_folder: impl AsRef<Path>,
    epochs: usize,
    lr_factor: f64,
    start_from_record: Option<ModelRecord<Autodiff<B>>>,
) -> Model<B> {
    let device = Default::default();
    let mut model = m.init::<Autodiff<B>>(&device);

    if let Some(record) = start_from_record {
        model = model.load_record(record);
    }

    let dataset_train = FileFolderDataset::load_from_folder(train_folder);
    let dataset_test = FileFolderDataset::load_from_folder(test_folder);

    let dataloader_train = DataLoaderBuilder::new(GenBatcher)
        .batch_size(32)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_train, 100_000));

    let dataloader_test = DataLoaderBuilder::new(GenBatcher)
        .batch_size(32)
        .num_workers(4)
        .build(SamplerDataset::new(dataset_test, 10000));

    let optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1.0e-6)))
        .init();

    let accum = 6;

    let lr_scheduler = NoamLrSchedulerConfig::new(lr_factor / accum as f64)
        .with_warmup_steps(6000)
        .with_model_size(m.embed_dims)
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
        .num_epochs(epochs)
        .summary();

    info!("Selected device: {:?}", device);

    let result = training.launch(Learner::new(model, optim, lr_scheduler));
    result.model
}
