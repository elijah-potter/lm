use std::path::{Path, PathBuf};

use burn::backend::Autodiff;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::Dataset;
use burn::data::dataset::transform::{SamplerDataset, SamplerDatasetOptions};
use burn::lr_scheduler::noam::NoamLrSchedulerConfig;
use burn::module::Module;
use burn::optim::decay::WeightDecayConfig;
use burn::optim::{AdamConfig, Optimizer};
use burn::prelude::Backend;
use burn::record::{CompactRecorder, Recorder};
use burn::train::metric::{
    AccuracyMetric, CudaMetric, LearningRateMetric, LossMetric, PerplexityMetric,
};
use burn::train::{Learner, SupervisedTraining};
use log::info;

use crate::batcher::GenBatcher;
use crate::dataset::FileFolderDataset;
use crate::model::{Model, ModelConfig, ModelRecord};

fn load_dataset(path: impl AsRef<Path>) -> FileFolderDataset {
    let path = path.as_ref();

    if path
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".tar.gz"))
    {
        return FileFolderDataset::load_from_tar_gz(path);
    }

    FileFolderDataset::load_from_folder(path)
}

pub fn train<B: Backend>(
    m: ModelConfig,
    train_folder: impl AsRef<Path>,
    test_folder: impl AsRef<Path>,
    epochs: usize,
    lr_factor: f64,
    start_from_record: Option<ModelRecord<Autodiff<B>>>,
    start_optimizer: Option<PathBuf>,
) -> Model<B> {
    let device = Default::default();
    let mut model = m.init::<Autodiff<B>>(&device);

    if let Some(record) = start_from_record {
        model = model.load_record(record);
    }

    let dataset_train = load_dataset(train_folder);
    println!("Loaded {} files for training.", dataset_train.len());
    let dataset_test = load_dataset(test_folder);
    println!("Loaded {} files for testing.", dataset_test.len());

    let dataloader_train = DataLoaderBuilder::new(GenBatcher)
        .batch_size(8)
        .num_workers(4)
        .build(SamplerDataset::new(
            dataset_train,
            SamplerDatasetOptions::default(),
        ));

    let dataloader_test = DataLoaderBuilder::new(GenBatcher)
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
        let record = CompactRecorder::new()
            .load(path.into(), &device)
            .expect("Should be able to load the optimizer state from the provided file");
        optim = optim.load_record(record);
    }

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
