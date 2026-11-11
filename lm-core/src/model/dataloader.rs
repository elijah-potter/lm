use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use burn::Tensor;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::data::dataset::transform::{SamplerDataset, SamplerDatasetOptions};
use burn::data::dataset::{Dataset, DatasetError};
use burn::prelude::Device;
use burn::tensor::Int;
use burn::tensor::ops::PadMode;
use parquet::file::reader::SerializedFileReader;
use parquet::record::RowAccessor;
use rand::RngExt;

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

const ESTIMATED_CHARACTERS_PER_TOKEN: f64 = 4.62;

fn character_start(length: usize, token_window: usize, rng: &mut impl rand::Rng) -> usize {
    let reserved = (token_window as f64 * ESTIMATED_CHARACTERS_PER_TOKEN).ceil() as usize;
    rng.random_range(0..=length.saturating_sub(reserved))
}

fn sample_context(text: &str, token_window: usize, rng: &mut impl rand::Rng) -> String {
    let start = character_start(text.chars().count(), token_window, rng);
    text.chars().skip(start).collect()
}

/// Determine the indices to include in the loss calculation.
/// Will include the first "pad" token, since "pad" tokens are the same as the end of text token.
fn loss_indices(lengths: impl IntoIterator<Item = usize>, max_seq_len: usize) -> Vec<i64> {
    lengths
        .into_iter()
        .enumerate()
        .flat_map(|(row, length)| {
            let start = row * max_seq_len;
            start..start + length.min(max_seq_len)
        })
        .map(|index| index as i64)
        .collect()
}

impl Batcher<UnbatchedItem, BatchItem> for TrainingBatcher {
    fn batch(&self, items: Vec<UnbatchedItem>, device: &Device) -> BatchItem {
        let items: Vec<(Tensor<2, Int>, Tensor<2, Int>, usize)> = items
            .into_iter()
            .map(|i| sample_context(&i.input, self.max_seq_len + 1, &mut rand::rng()))
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
                    cur_len,
                )
            })
            .collect();

        let loss_indices =
            loss_indices(items.iter().map(|(_, _, length)| *length), self.max_seq_len);
        let inputs_catted = Tensor::cat(
            items.iter().map(|(inputs, _, _)| inputs.clone()).collect(),
            0,
        );
        let outputs_catted = Tensor::cat(
            items
                .iter()
                .map(|(_, outputs, _)| outputs.clone())
                .collect(),
            0,
        );

        BatchItem {
            inputs: inputs_catted,
            targets: outputs_catted,
            loss_indices,
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
    // The indices to include in the loss calculation for the whole batch.
    pub loss_indices: Vec<i64>,
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
