use std::collections::VecDeque;
use std::str;

use burn::Tensor;
use burn::module::Module;
use burn::prelude::Backend;
use burn::tensor::{Int, Shape, TensorData};
use tiktoken_rs::r50k_base_singleton;

pub const VOCAB_SIZE: usize = 50255;
pub const MAX_SEQ_LEN: usize = 256;

pub fn text_to_indices<B: Backend>(text: &[char], device: &B::Device) -> Tensor<B, 2, Int> {
    let bpe = r50k_base_singleton();

    let string: String = text.into_iter().collect();

    let idxs = bpe.encode_ordinary(string.as_str());
    let mut idxs = VecDeque::from(idxs);
    idxs.truncate(MAX_SEQ_LEN);

    while idxs.len() < MAX_SEQ_LEN {
        idxs.push_front(0);
    }

    let tok_tensor_indices = Tensor::<B, 2, Int>::from_data(
        TensorData::new(idxs.into(), Shape::new([1, MAX_SEQ_LEN])),
        device,
    );

    assert_eq!(tok_tensor_indices.dims(), [1, MAX_SEQ_LEN]);

    tok_tensor_indices
}

/// Use with autoregressive cache.
pub fn text_to_indices_unpadded<B: Backend>(
    text: &[char],
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    if text.is_empty() {
        return Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![0i32], Shape::new([1, 1])),
            device,
        );
    }

    let bpe = r50k_base_singleton();

    let string: String = text.into_iter().collect();

    let mut idxs = bpe.encode_ordinary(string.as_str());
    idxs.truncate(MAX_SEQ_LEN);

    let len = idxs.len();

    let tok_tensor_indices =
        Tensor::<B, 2, Int>::from_data(TensorData::new(idxs, Shape::new([1, len])), device);

    tok_tensor_indices
}

pub fn indices_to_text<B: Backend>(tensor: Tensor<B, 2, Int>) -> Vec<char> {
    let bpe = r50k_base_singleton();

    let data = tensor.into_data();
    let idxs: Vec<i32> = data.to_vec().unwrap();
    let idxs: Vec<u32> = idxs.into_iter().map(|i| i as u32).collect();

    let str = bpe.decode(&idxs).unwrap();
    str.chars().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32, i32>;

    #[test]
    fn roundtrip_text() {
        let device = Default::default();
        let text = "Hello World!?:,.".chars().collect::<Vec<_>>();
        let indices = text_to_indices::<TestBackend>(&text, &device);
        let output = indices_to_text(indices);
        assert_eq!(output, text);
    }
}
