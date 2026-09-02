use burn::Tensor;
use burn::backend::NdArray;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::Backend;
use burn::tensor::backend::BackendTypes;
use burn::tensor::{Device, Int};
use rand::Rng;

use crate::tokenizer::text_to_indices;

pub struct GenBatcher;

impl<B: Backend> Batcher<B, Vec<char>, BatchItem<B>> for GenBatcher {
    fn batch(&self, items: Vec<Vec<char>>, device: &B::Device) -> BatchItem<B> {
        let mut inputs = Vec::with_capacity(items.len());
        let mut targets = Vec::with_capacity(items.len());

        // Allows fewer transfers to the GPU. Do just one big one.
        type CpuBackend = NdArray<f32, i32>;
        let cpu_device: <CpuBackend as BackendTypes>::Device = Default::default();

        for item in items {
            if item.is_empty() {
                continue;
            }

            let item = random_subslice(&item);

            if item.is_empty() {
                continue;
            }

            let input = text_to_indices::<CpuBackend>(&item[..item.len() - 1], &cpu_device);
            let target = text_to_indices::<CpuBackend>(&item[1..], &cpu_device);

            inputs.push(input);
            targets.push(target);
        }

        let input_tens = Tensor::cat(inputs, 0);
        let target_tens = Tensor::cat(targets, 0);

        BatchItem {
            input: Tensor::<B, 2, Int>::from_data(input_tens.into_data(), device),
            target: Tensor::<B, 2, Int>::from_data(target_tens.into_data(), device),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BatchItem<B: Backend> {
    pub input: Tensor<B, 2, Int>,
    pub target: Tensor<B, 2, Int>,
}

fn random_subslice<T>(s: &[T]) -> &[T] {
    let mut rng = rand::rng();
    let n = s.len();
    let i = rng.random_range(0..n);
    let j = rng.random_range(i + 1..=n);
    &s[i..j]
}
