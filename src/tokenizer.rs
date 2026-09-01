use std::collections::VecDeque;

use burn::Tensor;
use burn::prelude::Backend;
use burn::tensor::{Int, Shape, TensorData};
use tiktoken_rs::r50k_base_singleton;

/// Number of entries required for the r50k vocabulary, including its end-of-text token.
pub const VOCAB_SIZE: usize = 50257;
/// Token ID reserved for sequence padding.
pub const PAD_TOKEN: i32 = 50256;
pub const MAX_SEQ_LEN: usize = 256;

pub fn text_to_indices<B: Backend>(text: &[char], device: &B::Device) -> Tensor<B, 2, Int> {
    let bpe = r50k_base_singleton();

    // Limit to avoid wasting time in the tokenizer.
    let string: String = text.into_iter().take(64 * MAX_SEQ_LEN).collect();

    let idxs = bpe.encode_ordinary(string.as_str());
    let mut idxs = VecDeque::from(idxs);
    idxs.truncate(MAX_SEQ_LEN);

    while idxs.len() < MAX_SEQ_LEN {
        idxs.push_front(PAD_TOKEN as u32);
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
            TensorData::new(vec![PAD_TOKEN], Shape::new([1, 1])),
            device,
        );
    }

    let bpe = r50k_base_singleton();

    // Limit to avoid wasting time in the tokenizer.
    let string: String = text.into_iter().take(64 * MAX_SEQ_LEN).collect();

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
    let idxs: Vec<u32> = data
        .to_vec::<i32>()
        .unwrap()
        .into_iter()
        .filter(|&i| i != PAD_TOKEN)
        .map(|i| i as u32)
        .collect();

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
