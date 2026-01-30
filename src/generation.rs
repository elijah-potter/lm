use burn::Tensor;
use burn::prelude::Backend;
use burn::tensor::{Distribution, Int};

use crate::model::Model;
use crate::tokenizer::{indices_to_text, text_to_indices};

pub fn generate_single_pass<B: Backend>(
    model: &Model<B>,
    context: &[char],
    temperature: f64,
) -> Vec<char> {
    let input = text_to_indices(&context, &model.device());

    let output = model.forward(input);

    let indices = weighted_argmax_logits(output, temperature);
    let text = indices_to_text(indices);

    text
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

pub fn generate_n_tokens<B: Backend>(
    model: &Model<B>,
    context: &[char],
    n: usize,
    temperature: f64,
) -> Vec<char> {
    let mut output = Vec::with_capacity(n);
    let mut context: Vec<char> = context.to_vec();

    print!("\"");

    for c in &context {
        print!("{}", c);
    }

    for _ in 0..n {
        let pass = generate_single_pass(model, &context, temperature);
        let new_tok = pass.last().unwrap();

        output.push(*new_tok);
        context.push(*new_tok);

        print!("{}", new_tok);
    }

    println!("\"");

    output
}
