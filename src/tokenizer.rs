use std::sync::OnceLock;

use base64::{Engine as _, engine::general_purpose};
use burn::Tensor;
use burn::prelude::Backend;
use burn::tensor::{Int, Shape, TensorData};
use tiktoken_rs::{CoreBPE, Rank};

pub const VOCAB_SIZE: usize = 8192;
/// Token ID reserved for sequence padding.
pub const PAD_TOKEN: i32 = 0;
pub const MAX_SEQ_LEN: usize = 128;

static BPE: OnceLock<CoreBPE> = OnceLock::new();

/// Return the tokenizer loaded from `temple2.tiktoken`.
///
/// It is initialized at most once and shared by all callers.
pub fn bpe_singleton() -> &'static CoreBPE {
    BPE.get_or_init(|| {
        let encoder = include_str!("../temple2.tiktoken")
            .lines()
            .map(|line| {
                let (token, rank) = line
                    .split_once(' ')
                    .expect("each temple2.tiktoken line must contain a token and rank");
                let token = general_purpose::STANDARD
                    .decode(token)
                    .expect("temple2.tiktoken contains an invalid base64 token");
                let rank: Rank = rank
                    .parse()
                    .expect("temple2.tiktoken contains an invalid token rank");
                (token, rank)
            })
            .collect();
        let special_tokens = [
            (String::from("<|pad|>"), 0),
            (String::from("<|bos|>"), 1),
            (String::from("<|eos|>"), 2),
            (String::from("<|unk|>"), 3),
        ]
        .into_iter()
        .collect();

        CoreBPE::new(
            encoder,
            special_tokens,
            "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++| ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s",
        )
        .expect("failed to build the temple2 tokenizer")
    })
}

/// Tokenize text on the CPU, returning at most `max_tokens` token IDs.
///
/// The input character count is bounded before tokenization to avoid processing
/// an entire large document when only a short training sequence is needed.
pub fn text_to_token_ids(text: &[char], max_tokens: usize) -> Vec<i32> {
    if max_tokens == 0 {
        return Vec::new();
    }

    let string: String = text.iter().take(66 * max_tokens).collect();

    bpe_singleton()
        .encode_ordinary(&string)
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

pub fn indices_to_bytes<B: Backend>(tensor: Tensor<B, 2, Int>) -> Vec<u8> {
    let data = tensor.into_data();
    let idxs: Vec<u32> = data
        .to_vec::<i32>()
        .unwrap()
        .into_iter()
        .filter(|&i| i != PAD_TOKEN)
        .map(|i| i as u32)
        .collect();

    bpe_singleton().decode_bytes(&idxs).unwrap()
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
        let output = indices_to_bytes(indices);
        assert_eq!(output, text);
    }
}
