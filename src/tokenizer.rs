use burn::Tensor;
use burn::prelude::Backend;
use burn::tensor::{Int, Shape, TensorData};

const VOCAB_SIZE: usize = 64;
const MAX_SEQ_LEN: usize = 64;

pub fn text_to_indices<B: Backend>(text: &[char], device: &B::Device) -> Tensor<B, 2, Int> {
    let mut idxs: Vec<_> = text
        .iter()
        .copied()
        .map(char_to_index)
        .take(MAX_SEQ_LEN)
        .collect();

    while idxs.len() < MAX_SEQ_LEN {
        idxs.push(0);
    }

    let tok_tensor_indices =
        Tensor::<B, 2, Int>::from_data(TensorData::new(idxs, Shape::new([1, MAX_SEQ_LEN])), device);

    assert_eq!(tok_tensor_indices.dims(), [1, MAX_SEQ_LEN]);

    tok_tensor_indices
}

fn char_to_index(c: char) -> i32 {
    match c {
        'a'..='z' => (c as i32) - ('a' as i32) + 1,
        'A'..='Z' => (c as i32) - ('A' as i32) + 1,
        ' ' => VOCAB_SIZE as i32 - 1,
        _ => 0,
    }
}

fn index_to_char(i: i32) -> char {
    match i {
        1..=26 => (('a' as i32) + i - 1) as u8 as char,
        val if val as usize == VOCAB_SIZE - 1 => ' ',
        _ => '\0',
    }
}
