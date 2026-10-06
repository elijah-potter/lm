use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub train_dataset: PathBuf,
    pub validate_dataset: PathBuf,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub optim_beta_1: f64,
    pub optim_beta_2: f64,
    pub optim_epsilon: f64,
    pub optim_weight_decay: f64,
    pub lr_warmup_duration: usize,
    pub epochs: usize,
}
