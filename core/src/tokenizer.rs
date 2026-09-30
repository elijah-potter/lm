use burn::Tensor;
use burn::prelude::Device;
use burn::tensor::{Int, Shape, TensorData};
use tiktoken_rs::{CoreBPE, r50k_base_singleton};

pub const VOCAB_SIZE: usize = 50258;
/// Token ID reserved for sequence padding.
pub const PAD_TOKEN: i32 = 0;
pub const MAX_SEQ_LEN: usize = 128;

/// Returns the built-in OpenAI r50k_base (GPT-2) tokenizer.
pub fn bpe_singleton() -> &'static CoreBPE {
    r50k_base_singleton()
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
        .map(|token| token as i32 + 1)
        .collect()
}

/// Use with autoregressive cache.
pub fn text_to_indices_unpadded(text: &[char], device: &Device) -> Tensor<2, Int> {
    if text.is_empty() {
        return Tensor::<2, Int>::from_data(
            TensorData::new(vec![PAD_TOKEN], Shape::new([1, 1])),
            device,
        );
    }

    let idxs = text_to_token_ids(text, MAX_SEQ_LEN);
    let len = idxs.len();

    Tensor::<2, Int>::from_data(TensorData::new(idxs, Shape::new([1, len])), device)
}

pub fn indices_to_bytes(tensor: Tensor<2, Int>) -> Vec<u8> {
    let data = tensor.into_data();
    let idxs: Vec<u32> = data
        .try_to_vec_as::<i32>()
        .unwrap()
        .into_iter()
        .filter(|&i| i != PAD_TOKEN)
        .map(|i| (i - 1) as u32)
        .collect();

    bpe_singleton()
        .decode_bytes(&idxs)
        .expect("token IDs must belong to the r50k_base vocabulary")
}
