use serde::{Deserialize, Serialize};

/// Configuration supplied when initializing or loading LoRA weights. No dropout is applied.
#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
pub struct LoraConfig {
    /// Inner dimension of each low-rank update. Must be positive.
    pub rank: usize,
    /// Update scale numerator. The effective scale is `alpha / rank`.
    /// Must be finite and positive.
    pub alpha: f64,
}
