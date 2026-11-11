//! LoRA lifecycle and parameter selection for Burn's existing model modules.

use burn::{
    Tensor,
    module::{Lora, Module, Param, ParamGroup},
};

use super::{LoraConfig, model_inner::ModelInner};

/// Exact base-weight paths that receive LoRA. Embeddings, norms, and biases are excluded.
pub fn projection_paths(layers: usize) -> Vec<String> {
    let mut paths = Vec::with_capacity(layers * 6 + 1);
    for i in 0..layers {
        for projection in [
            "mha.query",
            "mha.key",
            "mha.value",
            "mha.output",
            "pwff.linear_inner",
            "pwff.linear_outer",
        ] {
            paths.push(format!("transformer.layers.{i}.{projection}.weight"));
        }
    }
    paths.push("resizer.weight".into());
    paths
}

/// Exact paths of Burn's attached factors for LoRA-only storage.
pub fn factor_paths(layers: usize) -> Vec<String> {
    projection_paths(layers)
        .into_iter()
        .flat_map(|path| [format!("{path}.lora.a"), format!("{path}.lora.b")])
        .collect()
}

/// Select only attached LoRA factors for module records.
pub fn factor_group(layers: usize) -> ParamGroup {
    ParamGroup::from_paths(factor_paths(layers))
}

/// Replace LoRA with Burn's zero-update initialization and freeze every base parameter.
/// Existing LoRA updates are removed, never merged.
pub fn attach(model: ModelInner, config: LoraConfig) -> ModelInner {
    let mut model = remove(model);
    let group = ParamGroup::from_paths(projection_paths(model.transformer.layers.len()));
    model = model.apply_lora(Lora::new(config.rank, config.alpha).set_param_group(group));
    model.lora_config = Some(config);
    model
}

/// Drop an attached update while retaining the original base, identity, and layout mappings.
/// This operates directly on the weight: a `ModuleMapper` would reattach the update afterward.
fn remove_weight(param: Param<Tensor<2>>) -> Param<Tensor<2>> {
    if param.adapter().is_none() {
        return param;
    }
    let (id, base, mapper) = param.consume();
    Param::from_mapped_value(id, base, mapper)
}

/// Remove all selected LoRA updates without merging, then restore full-model gradients.
pub fn remove(mut model: ModelInner) -> ModelInner {
    if model.lora_config.is_none() {
        return model;
    }
    for layer in &mut model.transformer.layers {
        for linear in [
            &mut layer.mha.query,
            &mut layer.mha.key,
            &mut layer.mha.value,
            &mut layer.mha.output,
            &mut layer.pwff.linear_inner,
            &mut layer.pwff.linear_outer,
        ] {
            linear.weight = remove_weight(linear.weight.clone());
        }
    }
    model.resizer.weight = remove_weight(model.resizer.weight);
    model.lora_config = None;
    model.set_require_grad(true)
}
