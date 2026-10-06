use burn::Tensor;
use burn::prelude::Device;
use burn::tensor::Int;
use tiktoken_rs::{CoreBPE, r50k_base};

use super::tensor_utils::int_seq_to_tensor;

#[derive(Clone)]
pub struct Tokenizer {
    bpe: CoreBPE,
}

impl Tokenizer {
    pub fn new() -> Self {
        Self {
            bpe: r50k_base().unwrap(),
        }
    }

    pub fn vocab_size(&self) -> usize {
        50_257
    }

    fn text_to_indices(&self, text: &str) -> Vec<u32> {
        self.bpe.encode_ordinary(text)
    }

    /// Emits a [1, SEQ_LEN] tensor.
    pub fn text_to_tensor(&self, text: &str, device: Device) -> Tensor<2, Int> {
        let indices = self.text_to_indices(text);
        int_seq_to_tensor(indices, device)
    }

    pub fn pad_id(&self) -> usize {
        50_256
    }
}
