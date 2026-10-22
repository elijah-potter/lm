use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub top_p: f64,
    pub dry_allowed_length: usize,
    pub dry_multiplier: f64,
    pub dry_base: f64,
    pub temperature: f64,
}
