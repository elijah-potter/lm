//! LoRA-only storage using Burn's standard module records.

use std::io::Read;
use std::path::Path;

use burn::{
    module::Module,
    store::{ModuleRecord, RecordError},
    tensor::Bytes,
    train::checkpoint::{Checkpointer, CheckpointerError, FileCheckpointer},
};
use burn_pack::{Reader, Writer};

use super::{LoraConfig, lora, model_inner::ModelInner};

/// Keep only LoRA factors so loading or saving cannot include base weights.
fn factors(record: ModuleRecord, layers: usize) -> Result<ModuleRecord, RecordError> {
    let paths = lora::factor_paths(layers);
    let tensors = Reader::from_bytes(record.into_bytes()?)?
        .into_tensors()?
        .into_iter()
        .filter(|tensor| paths.contains(&tensor.name))
        .collect();
    ModuleRecord::from_bytes(Writer::new(tensors).into_bytes()?)
}

/// Load matching LoRA factors with Burn's standard shape checks.
/// Partial loading leaves omitted factors at their initial values and preserves base weights.
/// Panics on I/O or Burn record errors; the supplied configuration sets rank and alpha.
pub fn load(
    weights: &mut impl Read,
    config: LoraConfig,
    base: &ModelInner,
    device: &burn::prelude::Device,
) -> ModelInner {
    let mut data = Vec::new();
    weights.read_to_end(&mut data).unwrap();
    let record = ModuleRecord::from_bytes(Bytes::from_bytes_vec(data)).unwrap();
    let record = factors(record, base.transformer.layers.len()).unwrap();
    lora::attach(base.clone().fork(device), config).load_record(record.allow_partial(true))
}

/// Filter model checkpoints to LoRA factors, retaining Burn's file naming and retention.
/// No custom metadata is stored. Optimizer and scheduler checkpoints are unchanged.
pub struct LoraCheckpointer {
    inner: FileCheckpointer,
    layers: usize,
}

impl LoraCheckpointer {
    pub fn new(directory: impl AsRef<Path>, layers: usize) -> Self {
        Self {
            inner: FileCheckpointer::new(directory, "model"),
            layers,
        }
    }
}

impl Checkpointer<ModuleRecord> for LoraCheckpointer {
    fn save(&self, epoch: usize, record: ModuleRecord) -> Result<(), CheckpointerError> {
        let record = factors(record, self.layers).map_err(CheckpointerError::Record)?;
        self.inner.save(epoch, record)
    }

    fn delete(&self, epoch: usize) -> Result<(), CheckpointerError> {
        <FileCheckpointer as Checkpointer<ModuleRecord>>::delete(&self.inner, epoch)
    }

    fn restore(&self, epoch: usize) -> Result<ModuleRecord, CheckpointerError> {
        let record = self.inner.restore(epoch)?;
        factors(record, self.layers)
            .map(|record| record.allow_partial(true))
            .map_err(CheckpointerError::Record)
    }
}
