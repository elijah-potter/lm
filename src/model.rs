use crate::batcher::BatchItem;
use crate::tokenizer::MAX_SEQ_LEN;
use crate::tokenizer::VOCAB_SIZE;
use burn::train::InferenceStep;
use burn::{
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Relu,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainOutput, TrainStep},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    dropout: Dropout,
    activation: Relu,
    char_embedding: Embedding<B>,
    pos_embedding: Embedding<B>,
    transformer: TransformerEncoder<B>,
}

impl<B: Backend> Model<B> {
    /// Embed a given string, simply skipping any incompatible tokens.
    /// Will panic if the provided string is too long.
    ///
    /// This is used to create the model's context.
    fn embed(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [_batches, len] = input.dims();
        assert_eq!(len, MAX_SEQ_LEN);

        let tok_embedding: Tensor<B, 3> = self.char_embedding.forward(input);

        let pos_tensor_indices = Tensor::<B, 2, Int>::from_data(
            TensorData::new(
                (0i32..len as i32).collect::<Vec<_>>(),
                Shape::new([1, MAX_SEQ_LEN]),
            ),
            &self.device(),
        );
        let pos_embedding = self.pos_embedding.forward(pos_tensor_indices);

        (tok_embedding + pos_embedding) / 2
    }

    pub fn device(&self) -> B::Device {
        let devices = self.devices();
        devices[0].clone()
    }

    pub fn forward_train(
        &self,
        input: Tensor<B, 2, Int>,
        target: Tensor<B, 2, Int>,
    ) -> ClassificationOutput<B> {
        let embedding = self.dropout.forward(self.embed(input));

        let [batch_size, seq_length, embedding_dims] = embedding.dims();
        assert_eq!(seq_length, MAX_SEQ_LEN);

        let output = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding));
        let output = self.dropout.forward(output);

        let loss_fn = CrossEntropyLossConfig::new().init(&self.device());

        let output_flat = output
            .clone()
            .reshape([batch_size * seq_length, embedding_dims]);
        let target_flat = target.reshape([batch_size * seq_length]);

        let loss = loss_fn.forward(output_flat.clone(), target_flat.clone());

        ClassificationOutput::new(loss, output_flat, target_flat)
    }

    fn forward_infer(
        &self,
        input: Tensor<B, 2, Int>,
        target: Tensor<B, 2, Int>,
    ) -> ClassificationOutput<B> {
        let embedding = self.embed(input);

        let [batch_size, seq_length, embedding_dims] = embedding.dims();
        assert_eq!(seq_length, MAX_SEQ_LEN);

        let output = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding));

        let loss_fn = CrossEntropyLossConfig::new().init(&self.device());

        let output_flat = output
            .clone()
            .reshape([batch_size * seq_length, embedding_dims]);
        let target_flat = target.reshape([batch_size * seq_length]);

        let loss = loss_fn.forward(output_flat.clone(), target_flat.clone());

        ClassificationOutput::new(loss, output_flat, target_flat)
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let class = self.forward_infer(input.clone(), input);
        class.output
    }
}

impl<B: AutodiffBackend> TrainStep for Model<B> {
    type Input = BatchItem<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: BatchItem<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_train(item.input, item.target);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> InferenceStep for Model<B> {
    type Input = BatchItem<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: BatchItem<B>) -> ClassificationOutput<B> {
        self.forward_infer(item.input, item.target)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "0.5")]
    dropout: f64,
    /// The total number of transformer blocks.
    transformer_blocks: usize,
    embed_dims: usize,
    /// The number of attention heads to use in the transformer.
    attn_heads: usize,
    /// The size of the perceptrons between the attention blocks.
    percept_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: Relu::new(),
            char_embedding: EmbeddingConfig::new(VOCAB_SIZE, self.embed_dims).init(device),
            pos_embedding: EmbeddingConfig::new(MAX_SEQ_LEN, self.embed_dims).init(device),
            transformer: TransformerEncoderConfig::new(
                self.embed_dims,
                self.percept_size,
                self.attn_heads,
                self.transformer_blocks,
            )
            .init(device),
        }
    }
}
