use burn::Tensor;
use burn::prelude::Device;
use burn::tensor::{Int, Shape, TensorCreationOptions, TensorData};

/// Convert an integer sequence to a tensor containing those same integers. Used for positional embedding and
/// tokenization.
pub fn int_seq_to_tensor(ints: Vec<u32>, device: Device) -> Tensor<2, Int> {
    let len = ints.len();

    Tensor::<2, Int>::from_data(
        TensorData::new(ints, Shape::new([1, len])),
        TensorCreationOptions::new(device),
    )
}
