use burn::Tensor;
use burn::prelude::Backend;
use burn::tensor::{Int, Shape, TensorData};
use tiktoken_rs::r50k_base_singleton;

/// Number of entries required for the r50k vocabulary, including its end-of-text token.
pub const VOCAB_SIZE: usize = 50257;
/// Token ID reserved for sequence padding.
pub const PAD_TOKEN: i32 = 50256;
pub const MAX_SEQ_LEN: usize = 128;

/// Tokenize text on the CPU, returning at most `max_tokens` r50k token IDs.
///
/// The input character count is bounded before tokenization to avoid processing
/// an entire large document when only a short training sequence is needed.
pub fn text_to_token_ids(text: &[char], max_tokens: usize) -> Vec<i32> {
    if max_tokens == 0 {
        return Vec::new();
    }

    let bpe = r50k_base_singleton();
    let string: String = text.iter().take(66 * max_tokens).collect();

    bpe.encode_ordinary(&string)
        .into_iter()
        .take(max_tokens)
        .map(|token| token as i32)
        .collect()
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

    let idxs = text_to_token_ids(text, MAX_SEQ_LEN);
    let len = idxs.len();

    Tensor::<B, 2, Int>::from_data(TensorData::new(idxs, Shape::new([1, len])), device)
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
        let indices = text_to_indices_unpadded::<TestBackend>(&text, &device);
        let output = indices_to_text(indices);
        assert_eq!(output, text);
    }

    #[test]
    fn token_ids_respect_requested_limit() {
        let text = "hello world ".repeat(100).chars().collect::<Vec<_>>();
        let token_ids = text_to_token_ids(&text, 5);

        assert_eq!(token_ids.len(), 5);
    }

    #[test]
    fn token_ids_support_an_empty_limit() {
        let text = "hello".chars().collect::<Vec<_>>();

        assert!(text_to_token_ids(&text, 0).is_empty());
    }
}
