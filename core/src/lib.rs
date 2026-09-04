#![recursion_limit = "256"]
//! Core language-model components for training and inference.
//!
//! This crate exposes the model, tokenizer, datasets, batching, generation, and
//! training APIs independently of the `lm` command-line application.

/// Data-loader batching for autoregressive model training.
pub mod batcher;
/// Datasets backed by folders and compressed tar archives.
pub mod dataset;
/// Datasets backed by Dolma JSONL shards.
pub mod dolma_dataset;
/// Autoregressive token generation and sampling.
pub mod generation;
/// Transformer model types and configuration.
pub mod model;
/// Tokenization utilities and vocabulary constants.
pub mod tokenizer;
/// Model training orchestration.
pub mod training;
