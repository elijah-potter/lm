use std::io::{self, Write};

use burn::Tensor;
use burn::nn::transformer::TransformerEncoderAutoregressiveCache;
use burn::prelude::Backend;
use burn::tensor::{Distribution, Int};

use crate::model::Model;
use crate::tokenizer::{indices_to_text, text_to_indices_unpadded};

fn sample_next_char<B: Backend>(
    model: &Model<B>,
    input: Tensor<B, 2, Int>,
    temperature: f64,
    cache: &mut TransformerEncoderAutoregressiveCache<B>,
) -> char {
    let output = model.forward(input, cache);
    let indices = weighted_argmax_logits(output, temperature);
    let text = indices_to_text(indices);
    *text.last().unwrap()
}

pub fn weighted_argmax_logits<B: Backend>(
    logits: Tensor<B, 2>,
    temperature: f64,
) -> Tensor<B, 2, Int> {
    let t = temperature.max(1e-6);
    let u = logits.random_like(Distribution::Uniform(0.0, 1.0));
    let g = -(-u.log()).log();
    (logits + g * t).argmax(1)
}

pub fn generate_tokens<B: Backend>(model: &Model<B>, context: &[char], temperature: f64) {
    let mut context: Vec<char> = context.to_vec();

    for c in &context {
        print!("{}", c);
    }

    let mut cache = model.create_cache();

    loop {
        let input = text_to_indices_unpadded(&context, &model.device());

        let new_tok = sample_next_char(model, input, temperature, &mut cache);

        context.push(new_tok);

        print!("{}", new_tok);
        io::stdout().flush().unwrap();
    }
}
