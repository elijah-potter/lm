use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use burn::Tensor;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::data::dataset::transform::{SamplerDataset, SamplerDatasetOptions};
use burn::data::dataset::{Dataset, DatasetError};
use burn::prelude::Device;
use burn::tensor::ops::PadMode;
use burn::tensor::{Float, Int, TensorCreationOptions, TensorData};
use parquet::file::reader::SerializedFileReader;
use parquet::record::RowAccessor;

use super::tokenizer::Tokenizer;

pub fn create_dataloader(
    device: Device,
    batch_size: usize,
    tokenizer: Tokenizer,
    max_seq_len: usize,
    epoch_size: usize,
    source: impl AsRef<Path>,
) -> Arc<dyn DataLoader<BatchItem>> {
    let dataset = ParquetDataset::load_from_file(source.as_ref());

    let sampler_options = SamplerDatasetOptions::default()
        .with_replacement()
        .with_fixed_size(epoch_size);

    DataLoaderBuilder::new(TrainingBatcher {
        tokenizer,
        max_seq_len,
    })
    .set_device(device.clone())
    .batch_size(batch_size)
    .num_workers(8)
    .build(SamplerDataset::new(dataset, sampler_options))
}

/// There's probably a better name for this.
/// It can be used for inference too.
struct TrainingBatcher {
    tokenizer: Tokenizer,
    max_seq_len: usize,
}

impl Batcher<UnbatchedItem, BatchItem> for TrainingBatcher {
    fn batch(&self, items: Vec<UnbatchedItem>, device: &Device) -> BatchItem {
        let items: Vec<(Tensor<2, Int>, Tensor<2, Int>)> = items
            .into_iter()
            // Sample the context randomly from the document.
            .map(|i| {
                if i.input.is_empty() {
                    return i.input;
                }

                let mut chars: Vec<char> = i.input.chars().collect();

                let start_char = rand::random_range(0..chars.len());
                chars.split_off(start_char).into_iter().collect()
            })
            .map(|sample| self.tokenizer.text_to_tensor(&sample, device.clone()))
            // Resize the tensor to the exact length required by the model, then get the before and
            // after tensors.
            .map(|mut t| {
                let cur_len = t.dims()[1];

                if cur_len <= self.max_seq_len {
                    let pad_right = (self.max_seq_len + 1).saturating_sub(cur_len);

                    t = t.pad(
                        [(0, pad_right)],
                        PadMode::Constant(self.tokenizer.pad_id() as f32),
                    );
                }

                (
                    t.clone().slice_dim(1, 0..self.max_seq_len),
                    t.slice_dim(1, 1..self.max_seq_len + 1),
                )
            })
            .collect();

        let inputs_catted =
            Tensor::cat(items.iter().map(|(inputs, _)| inputs.clone()).collect(), 0);
        let outputs_catted = Tensor::cat(
            items.iter().map(|(_, outputs)| outputs.clone()).collect(),
            0,
        );

        BatchItem {
            inputs: inputs_catted,
            targets: outputs_catted,
        }
    }
}

#[derive(Debug, Clone)]
struct UnbatchedItem {
    pub input: String,
}

#[derive(Debug, Clone)]
pub struct BatchItem {
    pub inputs: Tensor<2, Int>,
    pub targets: Tensor<2, Int>,
}

struct ParquetDataset {
    rows: Vec<String>,
}

impl ParquetDataset {
    pub fn load_from_file(path: &Path) -> Self {
        let file = File::open(path).unwrap();
        let reader = SerializedFileReader::new(file).unwrap();

        let mut rows = Vec::new();

        for read_row in reader.into_iter() {
            let row = read_row.unwrap();

            let text = row.get_string(0).unwrap();

            rows.push(text.clone());
        }

        Self { rows }
    }
}

impl Dataset<UnbatchedItem> for ParquetDataset {
    fn get(&self, index: usize) -> Result<UnbatchedItem, DatasetError> {
        let text = self.rows[index].clone();
        Ok(UnbatchedItem { input: text })
    }

    fn len(&self) -> usize {
        self.rows.len()
    }
}
