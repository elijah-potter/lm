use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Copy, Clone, Debug)]
pub struct ModelConfig {
    pub embedding_dimensions: usize,
    pub transformer_blocks: usize,
    pub attention_heads: usize,
    pub hidden_size: usize,
    pub max_seq_len: usize,
}
