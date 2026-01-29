use burn::Tensor;
use burn::prelude::Backend;
use burn::tensor::{Int, Shape, TensorData};

pub const VOCAB_SIZE: usize = 64;
pub const MAX_SEQ_LEN: usize = 64;

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

pub fn indices_to_text<B: Backend>(tensor: Tensor<B, 2, Int>) -> Vec<char> {
    let data = tensor.into_data();
    let idxs: Vec<i32> = data.to_vec().unwrap();

    idxs.into_iter()
        .take(MAX_SEQ_LEN)
        .filter(|&i| i != 0)
        .map(index_to_char)
        .collect()
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
