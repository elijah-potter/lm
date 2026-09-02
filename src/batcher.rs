use burn::Tensor;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::Backend;
use burn::tensor::{Int, Shape, TensorData};
use rand::Rng;

use crate::tokenizer::{MAX_SEQ_LEN, PAD_TOKEN, text_to_token_ids};

pub struct GenBatcher;

impl<B: Backend> Batcher<B, Vec<char>, BatchItem<B>> for GenBatcher {
    fn batch(&self, items: Vec<Vec<char>>, device: &B::Device) -> BatchItem<B> {
        let sequence_len = MAX_SEQ_LEN + 1;
        let mut sequences = Vec::with_capacity(items.len() * sequence_len);
        let mut batch_size = 0;

        for item in items {
            if item.is_empty() {
                continue;
            }

            let item = random_subslice(&item);
            let tokens = text_to_token_ids(item, sequence_len);

            if tokens.is_empty() {
                continue;
            }

            append_padded_sequence(&mut sequences, &tokens);
            batch_size += 1;
        }

        assert!(batch_size > 0, "cannot create a batch from empty inputs");

        let sequences = Tensor::<B, 2, Int>::from_data(
            TensorData::new(sequences, Shape::new([batch_size, sequence_len])),
            device,
        );
        let input = sequences.clone().slice([0..batch_size, 0..MAX_SEQ_LEN]);
        let target = sequences.slice([0..batch_size, 1..sequence_len]);

        BatchItem { input, target }
    }
}

#[derive(Debug, Clone)]
pub struct BatchItem<B: Backend> {
    pub input: Tensor<B, 2, Int>,
    pub target: Tensor<B, 2, Int>,
}

fn append_padded_sequence(destination: &mut Vec<i32>, tokens: &[i32]) {
    let sequence_len = MAX_SEQ_LEN + 1;
    assert!(tokens.len() <= sequence_len);

    destination.extend(std::iter::repeat_n(PAD_TOKEN, sequence_len - tokens.len()));
    destination.extend_from_slice(tokens);
}

fn random_subslice<T>(s: &[T]) -> &[T] {
    let mut rng = rand::rng();
    let n = s.len();
    let i = rng.random_range(0..n);
    let j = rng.random_range(i + 1..=n);
    &s[i..j]
}
