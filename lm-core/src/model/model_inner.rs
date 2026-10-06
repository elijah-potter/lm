use burn::Tensor;
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::loss::{CrossEntropyLoss, MseLoss, Reduction};
use burn::nn::transformer::{
    TransformerEncoder, TransformerEncoderAutoregressiveCache, TransformerEncoderInput,
};
use burn::nn::{Embedding, Linear};
use burn::tensor::{Float, Int};
use burn::train::{ClassificationOutput, InferenceStep, RegressionOutput, TrainOutput, TrainStep};

use burn::module::Module;

use super::dataloader::BatchItem;
use super::tensor_utils::int_seq_to_tensor;

#[derive(Module, Debug)]
pub struct ModelInner {
    pub embedding: Embedding,
    pub pos_embedding: Embedding,
    pub transformer: TransformerEncoder,
    #[module(skip)]
    pub max_seq_len: usize,
    #[module(skip)]
    pub pad_idx: usize,
}

impl ModelInner {
    /// Embed a sequence of tokens.
    fn embed(&self, tens: Tensor<2, Int>) -> Tensor<3, Float> {
        let norm_embed = self.embedding.forward(tens);

        let pos_embed_row = self.pos_embedding.forward(int_seq_to_tensor(
            (0_u32..self.max_seq_len as u32).collect(),
            self.devices().first().unwrap().clone(),
        ));
        let pos_embed = pos_embed_row.repeat_dim(0, norm_embed.dims()[0]);

        norm_embed.add(pos_embed)
    }

    pub fn forward(&self, item: BatchItem) -> ClassificationOutput {
        let embedding = self.embed(item.inputs);

        let device = self.devices().into_iter().next().unwrap();

        let [batch_size, _, embedding_dims] = embedding.dims();
        let ag_mask = generate_autoregressive_mask(batch_size, self.max_seq_len, &device);

        let trans_out = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding).mask_attn(ag_mask));

        // Our cross entropy loss calculation assumes this specific shape, even if it's unintuitive.
        // In essence, we're taking each batch and placing them end to end as if they were one
        // really long sequence.
        let trans_out_reshaped = trans_out.reshape([batch_size * self.max_seq_len, embedding_dims]);
        let targets_reshaped = item.targets.reshape([batch_size * self.max_seq_len]);

        let loss = CrossEntropyLoss::new(Some(self.pad_idx), &device)
            .forward(trans_out_reshaped.clone(), targets_reshaped.clone());

        ClassificationOutput::new(loss, trans_out_reshaped, targets_reshaped)
    }
}

impl TrainStep for ModelInner {
    type Input = BatchItem;

    type Output = ClassificationOutput;

    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output> {
        let out = self.forward(item);
        TrainOutput::new(self, out.loss.backward(), out)
    }
}

impl InferenceStep for ModelInner {
    type Input = BatchItem;

    type Output = ClassificationOutput;

    fn step(&self, item: Self::Input) -> Self::Output {
        self.forward(item)
    }
}
