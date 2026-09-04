use std::io::{self, Write};

use burn::Tensor;
use burn::nn::transformer::TransformerEncoderAutoregressiveCache;
use burn::prelude::Backend;
use burn::tensor::Int;
use burn::tensor::activation::softmax;

use crate::model::Model;
use crate::tokenizer::{
    MAX_SEQ_LEN, PAD_TOKEN, VOCAB_SIZE, indices_to_bytes, text_to_indices_unpadded,
};

fn sample_next_tok<B: Backend>(
    model: &Model<B>,
    input: Tensor<B, 2, Int>,
    temperature: f64,
    repetition_penalty: f64,
    top_p_probability: f64,
    cache: &mut TransformerEncoderAutoregressiveCache<B>,
) -> (Vec<u8>, Tensor<B, 2, Int>) {
    let output = model.forward(input.clone(), cache);
    let [len, vocab_size] = output.dims();
    let final_token = output.slice([len - 1..len, 0..vocab_size]);

    // Apply repetition penalty
    let used_tokens = input
        .clone()
        .mask_fill(input.clone().equal_elem(PAD_TOKEN), -1)
        .one_hot::<3>(VOCAB_SIZE)
        .float()
        .sum_dim(1)
        .clamp_max(1.0)
        .squeeze_dim(1);

    let penalty = used_tokens
        .mul_scalar(repetition_penalty - 1.0)
        .add_scalar(1.0);

    let positive = final_token.clone().greater_equal_elem(0.0);

    let penalized = final_token
        .clone()
        .mul(penalty.clone())
        .mask_where(positive, final_token.clone().div(penalty));

    // Apply top-p sampling.
    let temperature_applied = penalized.div_scalar(temperature);
    let probs = softmax(temperature_applied, 1);

    let sampled = sample_top_p(probs, top_p_probability);

    (indices_to_bytes(sampled.clone()), sampled)
}

/// Samples one token per batch row using nucleus (top-p) sampling.
fn sample_top_p<B: Backend>(logits: Tensor<B, 2>, top_p: f64) -> Tensor<B, 2, Int> {
    // Sort probabilities from largest to smallest.
    let (sorted_probs, sorted_indices) = logits.sort_descending_with_indices(1);

    // Probability mass before each token.
    let cumulative_before = sorted_probs.clone().cumsum(1).sub(sorted_probs.clone());

    // Remove tokens after the nucleus. The token crossing `top_p`
    // remains included.
    let remove = cumulative_before.greater_equal_elem(top_p);
    let filtered_sorted_probs = sorted_probs.mask_fill(remove, 0.0);

    // Sample a rank in sorted order, then map it back to the vocabulary ID.
    let sampled_rank = filtered_sorted_probs.categorical(1);

    sorted_indices.gather(1, sampled_rank)
}

/// Generates and streams tokens using the provided sampling parameters.
pub fn generate_tokens<B: Backend>(
    model: &Model<B>,
    context: &[char],
    temperature: f64,
    repetition_penalty: f64,
    top_p_probability: f64,
) {
    let mut context_bytes: Vec<u8> = context
        .into_iter()
        .flat_map(|ch| {
            let mut buf = [0u8; 4];
            let len = ch.encode_utf8(&mut buf).len();
            buf.into_iter().take(len)
        })
        .collect();

    let mut tokenized_context = text_to_indices_unpadded(&context, &model.device());

    for c in context {
        print!("{}", c);
    }

    let mut cache = model.create_cache();

    loop {
        let (new_tok_bytes, new_tok_tens) = sample_next_tok(
            model,
            tokenized_context.clone(),
            temperature,
            repetition_penalty,
            top_p_probability,
            &mut cache,
        );

        io::stdout().write_all(&new_tok_bytes).unwrap();
        io::stdout().flush().unwrap();

        context_bytes.extend(new_tok_bytes);

        let pre = if tokenized_context.dims()[1] >= MAX_SEQ_LEN {
            cache = model.create_cache();
            tokenized_context.slice([0..1, 1..MAX_SEQ_LEN])
        } else {
            tokenized_context
        };

        tokenized_context = Tensor::cat(vec![pre, new_tok_tens], 1);
    }
}
