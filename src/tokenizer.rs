use std::collections::VecDeque;

use burn::Tensor;
use burn::prelude::Backend;
use burn::tensor::{Int, Shape, TensorData};

pub const VOCAB_SIZE: usize = 64;
pub const MAX_SEQ_LEN: usize = 256;

pub fn text_to_indices<B: Backend>(text: &[char], device: &B::Device) -> Tensor<B, 2, Int> {
    let mut idxs: VecDeque<_> = text
        .iter()
        .copied()
        .map(char_to_index)
        .rev()
        .take(MAX_SEQ_LEN)
        .rev()
        .collect();

    while idxs.len() < MAX_SEQ_LEN {
        idxs.push_front(0);
    }

    let tok_tensor_indices = Tensor::<B, 2, Int>::from_data(
        TensorData::new(idxs.into(), Shape::new([1, MAX_SEQ_LEN])),
        device,
    );

    assert_eq!(tok_tensor_indices.dims(), [1, MAX_SEQ_LEN]);

    tok_tensor_indices
}

pub fn indices_to_text<B: Backend>(tensor: Tensor<B, 2, Int>) -> Vec<char> {
    let data = tensor.into_data();
    let idxs: Vec<i32> = data.to_vec().unwrap();

    idxs.into_iter()
        .take(MAX_SEQ_LEN)
        .map(index_to_char)
        .collect()
}

fn char_to_index(c: char) -> i32 {
    match c {
        'a'..='z' => (c as i32) - ('a' as i32) + 1,
        'A'..='Z' => (c as i32) - ('A' as i32) + 27,
        '!' => 53,
        '.' => 54,
        ',' => 55,
        ':' => 56,
        '?' => 57,
        '\'' | 'ʼ' => 58,
        ' ' => VOCAB_SIZE as i32 - 1,
        _ => 0,
    }
}

fn index_to_char(i: i32) -> char {
    match i {
        1..=26 => (('a' as i32) + i - 1) as u8 as char,
        27..=52 => (('A' as i32) + i - 27) as u8 as char,
        53 => '!',
        54 => '.',
        55 => ',',
        56 => ':',
        57 => '?',
        58 => '\'',
        val if val as usize == VOCAB_SIZE - 1 => ' ',
        _ => '\0',
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32, i32>;

    #[test]
    fn roundtrip_text() {
        let device = Default::default();
        let text = "Hello World!?:,.".chars().collect::<Vec<_>>();
        let indices = text_to_indices::<TestBackend>(&text, &device);
        let output = indices_to_text(indices);
        assert_eq!(output, text);
    }
}
