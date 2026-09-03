use std::io::{self, Write};

use burn::Tensor;
use burn::nn::transformer::TransformerEncoderAutoregressiveCache;
use burn::prelude::Backend;
use burn::tensor::{Distribution, Int};

use crate::model::Model;
use crate::tokenizer::{MAX_SEQ_LEN, indices_to_bytes, text_to_indices_unpadded};

fn sample_next_tok<B: Backend>(
    model: &Model<B>,
    input: Tensor<B, 2, Int>,
    temperature: f64,
    cache: &mut TransformerEncoderAutoregressiveCache<B>,
) -> (Vec<u8>, Tensor<B, 2, Int>) {
    let output = model.forward(input, cache);
    let [len, vocab_size] = output.dims();
    let final_token = output.slice([len - 1..len, 0..vocab_size]);
    let indices = weighted_argmax_logits(final_token, temperature);
    (indices_to_bytes(indices.clone()), indices)
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
        let (new_tok_bytes, new_tok_tens) =
            sample_next_tok(model, tokenized_context.clone(), temperature, &mut cache);

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
