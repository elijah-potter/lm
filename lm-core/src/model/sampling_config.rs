use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub top_p: f64,
}
