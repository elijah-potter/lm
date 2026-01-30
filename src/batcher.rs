use burn::Tensor;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::Backend;
use burn::tensor::Int;
use rand::Rng;

use crate::tokenizer::text_to_indices;

pub struct GenBatcher;

impl<B: Backend> Batcher<B, Vec<char>, BatchItem<B>> for GenBatcher {
    fn batch(&self, items: Vec<Vec<char>>, device: &B::Device) -> BatchItem<B> {
        let mut inputs = Vec::with_capacity(items.len());
        let mut targets = Vec::with_capacity(items.len());

        for item in items {
            if item.is_empty() {
                continue;
            }

            let item = random_subslice(&item);

            if item.is_empty() {
                continue;
            }

            // info!("{:?}", item.iter().collect::<String>());

            let input = text_to_indices(&item[..item.len() - 1], device);
            let target = text_to_indices(&item[1..], device);

            inputs.push(input);
            targets.push(target);
        }

        let input_tens = Tensor::cat(inputs, 0);
        let target_tens = Tensor::cat(targets, 0);

        BatchItem {
            input: input_tens,
            target: target_tens,
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
