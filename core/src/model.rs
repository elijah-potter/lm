use crate::batcher::BatchItem;
use crate::tokenizer::MAX_SEQ_LEN;
use crate::tokenizer::{PAD_TOKEN, VOCAB_SIZE};
use burn::nn::LinearConfig;
use burn::nn::attention::generate_autoregressive_mask;
use burn::nn::transformer::TransformerEncoderAutoregressiveCache;
use burn::train::InferenceStep;
use burn::{
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, Relu,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
    },
    prelude::*,
    train::{ClassificationOutput, TrainOutput, TrainStep},
};

#[derive(Module, Debug)]
pub struct Model {
    dropout: Dropout,
    activation: Relu,
    char_embedding: Embedding,
    pos_embedding: Embedding,
    transformer: TransformerEncoder,
    resizer: Linear,
}

impl Model {
    /// Embed a given string, simply skipping any incompatible tokens.
    /// Will panic if the provided string is too long.
    ///
    /// This is used to create the model's context.
    fn embed(&self, input: Tensor<2, Int>) -> Tensor<3> {
        let [_batches, len] = input.dims();
        assert!(len <= MAX_SEQ_LEN);

        let tok_embedding: Tensor<3> = self.char_embedding.forward(input);

        let pos_tensor_indices = Tensor::<2, Int>::from_data(
            TensorData::new((0i32..len as i32).collect::<Vec<_>>(), Shape::new([1, len])),
            &self.device(),
        );
        let pos_embedding = self.pos_embedding.forward(pos_tensor_indices);

        (tok_embedding + pos_embedding) / 2
    }

    pub fn device(&self) -> Device {
        let devices = self.devices();
        devices[0].clone()
    }

    pub fn forward_train(
        &self,
        input: Tensor<2, Int>,
        target: Tensor<2, Int>,
    ) -> ClassificationOutput {
        let embedding = self.dropout.forward(self.embed(input));

        let [batch_size, seq_length, _embedding_dims] = embedding.dims();
        assert_eq!(seq_length, MAX_SEQ_LEN);

        let mask = generate_autoregressive_mask(batch_size, seq_length, &self.device());

        let trans_out = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding).mask_attn(mask));
        let trans_out = self.dropout.forward(trans_out);

        let output = self.resizer.forward(trans_out);
        let output = self.dropout.forward(output);

        let loss_fn = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![PAD_TOKEN as usize]))
            .init(&self.device());

        let output_flat = output
            .clone()
            .reshape([batch_size * seq_length, VOCAB_SIZE]);
        let target_flat = target.reshape([batch_size * seq_length]);

        let loss = loss_fn.forward(output_flat.clone(), target_flat.clone());

        ClassificationOutput::new(loss, output_flat, target_flat)
    }

    pub fn create_cache(&self) -> TransformerEncoderAutoregressiveCache {
        self.transformer.new_autoregressive_cache()
    }

    fn forward_infer(
        &self,
        input: Tensor<2, Int>,
        target: Tensor<2, Int>,
        cache: &mut TransformerEncoderAutoregressiveCache,
    ) -> ClassificationOutput {
        let embedding = self.embed(input);

        let [batch_size, seq_length, _embedding_dims] = embedding.dims();
        assert!(seq_length <= MAX_SEQ_LEN);

        let mask = generate_autoregressive_mask(batch_size, seq_length, &self.device());

        let trans_out = self.transformer.forward_autoregressive_inference(
            TransformerEncoderInput::new(embedding).mask_attn(mask),
            cache,
        );
        let output = self.resizer.forward(trans_out);

        let loss_fn = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![PAD_TOKEN as usize]))
            .init(&self.device());

        let output_flat = output
            .clone()
            .reshape([batch_size * seq_length, VOCAB_SIZE]);
        let target_flat = target.reshape([batch_size * seq_length]);

        let loss = loss_fn.forward(output_flat.clone(), target_flat.clone());

        ClassificationOutput::new(loss, output_flat, target_flat)
    }

    pub fn forward(
        &self,
        input: Tensor<2, Int>,
        cache: &mut TransformerEncoderAutoregressiveCache,
    ) -> Tensor<2> {
        let class = self.forward_infer(input.clone(), input, cache);
        class.output
    }
}

impl TrainStep for Model {
    type Input = BatchItem;
    type Output = ClassificationOutput;

    fn step(&self, item: BatchItem) -> TrainOutput<ClassificationOutput> {
        let item = self.forward_train(item.input, item.target);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl InferenceStep for Model {
    type Input = BatchItem;
    type Output = ClassificationOutput;

    fn step(&self, item: BatchItem) -> ClassificationOutput {
        self.forward_infer(item.input, item.target, &mut self.create_cache())
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = "0.5")]
    pub dropout: f64,
    /// The total number of transformer blocks.
    pub transformer_blocks: usize,
    pub embed_dims: usize,
    /// The number of attention heads to use in the transformer.
    pub attn_heads: usize,
    /// The size of the perceptrons between the attention blocks.
    pub percept_size: usize,
}

impl ModelConfig {
    pub fn init(&self, device: &Device) -> Model {
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
            resizer: LinearConfig::new(self.embed_dims, VOCAB_SIZE).init(device),
        }
    }
}
