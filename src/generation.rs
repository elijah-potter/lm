use burn::Tensor;
use burn::prelude::Backend;

use crate::model::Model;
use crate::tokenizer::{self, indices_to_text, text_to_indices};

pub fn generate<B: Backend>(model: &Model<B>, context: &str) {
    let chars: Vec<_> = context.chars().collect();

    let input = text_to_indices(&chars, &model.device());

    let output = model.forward(input);

    let indices = output.argmax(1);
    let text = indices_to_text(indices);

    let text_str: String = text.into_iter().collect();

    dbg!(text_str);
}
