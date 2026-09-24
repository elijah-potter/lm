use std::path::{Path, PathBuf};

use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::transform::{SamplerDataset, SamplerDatasetOptions};
use burn::data::dataset::{Dataset, DatasetError};
use burn::lr_scheduler::linear::LinearLrSchedulerConfig;
use burn::module::{AutodiffModule, Module};
use burn::optim::AdamConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::prelude::Device;
use burn::store::ModuleRecord;
use burn::train::metric::{
    AccuracyMetric, CudaMetric, LearningRateMetric, LossMetric, PerplexityMetric,
};
use burn::train::{Learner, SupervisedTraining};
use log::info;

use crate::batcher::GenBatcher;
use crate::dataset::FileFolderDataset;
use crate::dolma_dataset::DolmaDataset;
use crate::model::{Model, ModelConfig};
use crate::tokenizer::PAD_TOKEN;

enum TrainingDataset {
    Files(FileFolderDataset),
    Dolma(DolmaDataset),
}

impl Dataset<Vec<char>> for TrainingDataset {
    fn get(&self, index: usize) -> Result<Vec<char>, DatasetError> {
        match self {
            Self::Files(dataset) => dataset.get(index),
            Self::Dolma(dataset) => dataset.get(index),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Files(dataset) => dataset.len(),
            Self::Dolma(dataset) => dataset.len(),
        }
    }
}

fn load_dataset(path: impl AsRef<Path>) -> TrainingDataset {
    let path = path.as_ref();
    let file_name = path.file_name().and_then(|name| name.to_str());

    if file_name.is_some_and(|name| name.ends_with(".json.gz")) {
        return TrainingDataset::Dolma(DolmaDataset::load_from_json_gz(path));
    }

    if path.is_dir() && DolmaDataset::folder_contains_shards(path) {
        return TrainingDataset::Dolma(DolmaDataset::load_from_folder(path));
    }

    if file_name.is_some_and(|name| name.ends_with(".tar.gz")) {
        return TrainingDataset::Files(FileFolderDataset::load_from_tar_gz(path));
    }

    TrainingDataset::Files(FileFolderDataset::load_from_folder(path))
}

pub fn train(
    device: Device,
    m: ModelConfig,
    train_folder: impl AsRef<Path>,
    test_folder: impl AsRef<Path>,
    epochs: usize,
    lr_factor: f64,
    start_from_record: Option<ModuleRecord>,
    start_optimizer: Option<PathBuf>,
) -> Model {
    let device = device.autodiff();
    let mut model = m.init(&device);

    if let Some(record) = start_from_record {
        model = model.load_record(record);
    }

    let dataset_train = load_dataset(train_folder);
    println!("Loaded {} files for training.", dataset_train.len());
    let dataset_test = load_dataset(test_folder);
    println!("Loaded {} files for testing.", dataset_test.len());

    let dataloader_train = DataLoaderBuilder::new(GenBatcher)
        .set_device(device.clone())
        .batch_size(8)
        .num_workers(4)
        .build(SamplerDataset::new(
            dataset_train,
            SamplerDatasetOptions::default(),
        ));

    let dataloader_test = DataLoaderBuilder::new(GenBatcher)
        .set_device(device.clone().inner())
        .batch_size(8)
        .num_workers(4)
        .build(SamplerDataset::new(
            dataset_test,
            SamplerDatasetOptions::default(),
        ));

    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1.0e-6)))
        .init();

    if let Some(path) = start_optimizer {
        optim = optim
            .load(path)
            .expect("Should be able to load the optimizer state from the provided file");
    }

    let accum = 6;

    let lr_scheduler = LinearLrSchedulerConfig::new(
        lr_factor / accum as f64 / 100.0,
        lr_factor / accum as f64,
        6000,
    )
    .init()
    .unwrap();

    let training = SupervisedTraining::new("./checkpoints", dataloader_train, dataloader_test)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        //.metric_train_numeric(AccuracyMetric::new().with_pad_token(PAD_TOKEN as usize))
        .metric_valid_numeric(AccuracyMetric::new().with_pad_token(PAD_TOKEN as usize))
        //.metric_train_numeric(PerplexityMetric::new().with_pad_token(PAD_TOKEN as usize))
        .metric_valid_numeric(PerplexityMetric::new().with_pad_token(PAD_TOKEN as usize))
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_default_checkpointers()
        .grads_accumulation(accum)
        .num_epochs(epochs)
        .summary();

    info!("Selected device: {:?}", device);

    let result = training.launch(Learner::new(model, optim, lr_scheduler));
    result.model.valid()
}
