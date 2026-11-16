//! Exact-layout forward step for the old Magenta.js DrumsRNN attention wrapper.
//!
//! Source of operation order: magenta/magenta-js
//! music/src/music_rnn/{attention,model}.ts and core/aux_inputs.ts.
//! This is a correctness-first CPU implementation (no audio or sampling).
//! Shapes below correspond to the particular 3×256 DrumsRNN checkpoint.

use crate::lstm::{BasicLstmCell, LstmState};
use crate::{Checkpoint, Tensor};
use std::io::{self, ErrorKind};

const HIDDEN: usize = 256;
const INPUT: usize = 518; // 512 drum classes + 6 signed binary-counter bits
const OUTPUT_CLASSES: usize = 512;

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}

fn tensor<'a>(
    checkpoint: &'a Checkpoint,
    name: &str,
    required_shape: &[usize],
) -> io::Result<&'a [f32]> {
    let value = checkpoint
        .tensors
        .get(name)
        .ok_or_else(|| invalid(format!("Missing tensor {name}")))?;
    if value.shape != required_shape
        || value.values.len() != required_shape.iter().product::<usize>()
    {
        return Err(invalid(format!(
            "{name}: expected shape {required_shape:?}, got {:?}",
            value.shape
        )));
    }
    Ok(&value.values)
}

/// Row-major dense layer: [input_length, output_length].
fn dense(input: &[f32], kernel: &[f32], bias: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len() * output.len(), kernel.len());
    debug_assert_eq!(output.len(), bias.len());
    output.copy_from_slice(bias);
    for (row, &value) in input.iter().enumerate() {
        let weights = &kernel[row * output.len()..(row + 1) * output.len()];
        for (target, &weight) in output.iter_mut().zip(weights) {
            *target += value * weight;
        }
    }
}

/// The auxiliary input used by Google's Magenta.js BinaryCounter:
/// low bit first, counting from (step + 1), with -1 instead of zero.
pub fn binary_counter(step: usize) -> [f32; 6] {
    let value = step.wrapping_add(1);
    let mut bits = [-1.0; 6];
    for (index, bit) in bits.iter_mut().enumerate() {
        if (value >> index) & 1 != 0 {
            *bit = 1.0;
        }
    }
    bits
}

pub struct AttentionRnn<'a> {
    layers: [BasicLstmCell<'a>; 3],
    input_kernel: &'a [f32],
    input_bias: &'a [f32],
    attention_w: &'a [f32],
    attention_v: &'a [f32],
    attention_kernel: &'a [f32],
    attention_bias: &'a [f32],
    output_kernel: &'a [f32],
    output_bias: &'a [f32],
    class_kernel: &'a [f32],
    class_bias: &'a [f32],
    attention_length: usize,
}

pub struct AttentionState {
    pub layers: [LstmState; 3],
    /// Previous 32 output vectors, oldest to newest.
    memory: Vec<f32>,
    /// Attention vector used in the *next* input projection.
    context: Vec<f32>,
    rnn_input: Vec<f32>,
    projected_input: Vec<f32>,
    query: Vec<f32>,
    transformed_query: Vec<f32>,
    scores: Vec<f32>,
    attention: Vec<f32>,
    concatenated_output: Vec<f32>,
    output: Vec<f32>,
    logits: Vec<f32>,
}

impl AttentionState {
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.memory.fill(0.0);
        self.context.fill(0.0);
        self.rnn_input.fill(0.0);
        self.projected_input.fill(0.0);
        self.query.fill(0.0);
        self.transformed_query.fill(0.0);
        self.scores.fill(0.0);
        self.attention.fill(0.0);
        self.concatenated_output.fill(0.0);
        self.output.fill(0.0);
        self.logits.fill(0.0);
    }

    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    pub fn output(&self) -> &[f32] {
        &self.output
    }
}

impl<'a> AttentionRnn<'a> {
    pub fn from_checkpoint(checkpoint: &'a Checkpoint) -> io::Result<Self> {
        if checkpoint.attention_length == 0 || checkpoint.attention_length > 4096 {
            return Err(invalid("Attention length must be in 1..=4096"));
        }
        let layers = [
            BasicLstmCell::from_checkpoint(checkpoint, 0)?,
            BasicLstmCell::from_checkpoint(checkpoint, 1)?,
            BasicLstmCell::from_checkpoint(checkpoint, 2)?,
        ];
        for (index, layer) in layers.iter().enumerate() {
            let expected_input = if index == 0 { INPUT } else { HIDDEN };
            if layer.hidden_size() != HIDDEN || layer.input_size() != expected_input {
                return Err(invalid(format!("Unexpected input/hidden size in LSTM layer {index}")));
            }
        }
        let prefix = "rnn/attention_cell_wrapper/";
        Ok(Self {
            layers,
            input_kernel: tensor(
                checkpoint,
                &format!("{prefix}kernel"),
                &[INPUT + HIDDEN, INPUT],
            )?,
            input_bias: tensor(checkpoint, &format!("{prefix}bias"), &[INPUT])?,
            attention_w: tensor(
                checkpoint,
                &format!("{prefix}attention/attn_w"),
                &[1, 1, HIDDEN, HIDDEN],
            )?,
            attention_v: tensor(checkpoint, &format!("{prefix}attention/attn_v"), &[HIDDEN])?,
            attention_kernel: tensor(
                checkpoint,
                &format!("{prefix}attention/kernel"),
                &[HIDDEN * 6, HIDDEN],
            )?,
            attention_bias: tensor(
                checkpoint,
                &format!("{prefix}attention/bias"),
                &[HIDDEN],
            )?,
            output_kernel: tensor(
                checkpoint,
                &format!("{prefix}attention_output_projection/kernel"),
                &[HIDDEN * 2, HIDDEN],
            )?,
            output_bias: tensor(
                checkpoint,
                &format!("{prefix}attention_output_projection/bias"),
                &[HIDDEN],
            )?,
            class_kernel: tensor(
                checkpoint,
                "fully_connected/weights",
                &[HIDDEN, OUTPUT_CLASSES],
            )?,
            class_bias: tensor(checkpoint, "fully_connected/biases", &[OUTPUT_CLASSES])?,
            attention_length: checkpoint.attention_length,
        })
    }

    pub fn zero_state(&self) -> AttentionState {
        AttentionState {
            layers: [
                self.layers[0].zero_state(),
                self.layers[1].zero_state(),
                self.layers[2].zero_state(),
            ],
            memory: vec![0.0; self.attention_length * HIDDEN],
            context: vec![0.0; HIDDEN],
            rnn_input: vec![0.0; INPUT + HIDDEN],
            projected_input: vec![0.0; INPUT],
            query: vec![0.0; HIDDEN * 6],
            transformed_query: vec![0.0; HIDDEN],
            scores: vec![0.0; self.attention_length],
            attention: vec![0.0; HIDDEN],
            concatenated_output: vec![0.0; HIDDEN * 2],
            output: vec![0.0; HIDDEN],
            logits: vec![0.0; OUTPUT_CLASSES],
        }
    }

    /// Run one step for a drum class from 0 to 511.
    ///
    /// The input class is the previous encoded drum event. For autonomous
    /// generation, step=0 with class=0 is the silent bootstrap step.
    /// This computes *next-event logits* but does not sample or decode MIDI.
    pub fn step(
        &self,
        class: usize,
        step: usize,
        state: &mut AttentionState,
    ) -> io::Result<&'static str> {
        if class >= OUTPUT_CLASSES {
            return Err(invalid(format!("Class index out of range: {class}")));
        }
        if state.memory.len() != self.attention_length * HIDDEN {
            return Err(invalid("Incorrect attention memory length"));
        }

        state.rnn_input.fill(0.0);
        state.rnn_input[class] = 1.0;
        state.rnn_input[OUTPUT_CLASSES..INPUT].copy_from_slice(&binary_counter(step));
        state.rnn_input[INPUT..].copy_from_slice(&state.context);

        dense(
            &state.rnn_input, self.input_kernel, self.input_bias,
            &mut state.projected_input,
        );
        self.layers[0].step(&state.projected_input, &mut state.layers[0])?;
        for index in 1..3 {
            let (previous, next) = state.layers.split_at_mut(index);
            self.layers[index].step(&previous[index - 1].hidden, &mut next[0])?;
        }

        // TF AttentionWrapper queries from all c/h pairs in this order:
        // c0, h0, c1, h1, c2, h2; not just the final hidden vector.
        let mut offset = 0;
        for layer in &state.layers {
            state.query[offset..offset + HIDDEN].copy_from_slice(&layer.cell);
            offset += HIDDEN;
            state.query[offset..offset + HIDDEN].copy_from_slice(&layer.hidden);
            offset += HIDDEN;
        }
        dense(
            &state.query, self.attention_kernel, self.attention_bias,
            &mut state.transformed_query,
        );

        // 1x1 conv over 32 past output vectors, followed by tanh and v dot.
        // TF shape attn_w is [1,1,256,256]; last two dims are input/output.
        for slot in 0..self.attention_length {
            let memory = &state.memory[slot * HIDDEN..(slot + 1) * HIDDEN];
            let mut score = 0.0f32;
            for col in 0..HIDDEN {
                let mut projected = state.transformed_query[col];
                for row in 0..HIDDEN {
                    projected += memory[row] * self.attention_w[row * HIDDEN + col];
                }
                score += self.attention_v[col] * projected.tanh();
            }
            state.scores[slot] = score;
        }

        // Stable softmax over the memory slots, including initial zero slots.
        let peak = state.scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut denominator = 0.0f32;
        for score in &mut state.scores {
            *score = (*score - peak).exp();
            denominator += *score;
        }
        if denominator <= 0.0 || !denominator.is_finite() {
            return Err(invalid("Non-finite attention softmax"));
        }

        state.attention.fill(0.0);
        for slot in 0..self.attention_length {
            let weight = state.scores[slot] / denominator;
            for index in 0..HIDDEN {
                state.attention[index] +=
                    weight * state.memory[slot * HIDDEN + index];
            }
        }

        state.concatenated_output[..HIDDEN].copy_from_slice(&state.layers[2].hidden);
        state.concatenated_output[HIDDEN..].copy_from_slice(&state.attention);
        dense(
            &state.concatenated_output, self.output_kernel, self.output_bias,
            &mut state.output,
        );

        // New output is appended to the memory; the oldest slot is discarded.
        state.memory.copy_within(HIDDEN.., 0);
        let tail = (self.attention_length - 1) * HIDDEN;
        state.memory[tail..].copy_from_slice(&state.output);
        state.context.copy_from_slice(&state.attention);

        dense(
            &state.output, self.class_kernel, self.class_bias, &mut state.logits,
        );
        if state.logits.iter().any(|value| !value.is_finite()) {
            return Err(invalid("Non-finite class logits"));
        }
        Ok("ok")
    }
}

/// Normalize logits for inspection and eventual categorical sampling.
pub fn softmax(logits: &[f32], temperature: f32) -> io::Result<Vec<f32>> {
    if logits.is_empty() || !temperature.is_finite() || temperature <= 0.0
        || logits.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("Invalid logits or temperature"));
    }
    let scaled_peak = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max)
        / temperature;
    let scaled: Vec<f32> = logits
        .iter()
        .map(|v| (*v / temperature - scaled_peak).exp())
        .collect();
    let sum: f32 = scaled.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(invalid("Invalid softmax sum"));
    }
    Ok(scaled.into_iter().map(|v| v / sum).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn empty_tensor(shape: &[usize]) -> Tensor {
        Tensor {
            shape: shape.to_vec(),
            values: vec![0.0; shape.iter().product()],
        }
    }

    fn tiny_zero_checkpoint() -> Checkpoint {
        let prefix = "rnn/attention_cell_wrapper/";
        let mut tensors = BTreeMap::new();
        let specs: Vec<(String, Vec<usize>)> = vec![
            (format!("{prefix}kernel"), vec![774, 518]),
            (format!("{prefix}bias"), vec![518]),
            (format!("{prefix}attention/attn_w"), vec![1, 1, 256, 256]),
            (format!("{prefix}attention/attn_v"), vec![256]),
            (format!("{prefix}attention/kernel"), vec![1536, 256]),
            (format!("{prefix}attention/bias"), vec![256]),
            (format!("{prefix}attention_output_projection/kernel"), vec![512, 256]),
            (format!("{prefix}attention_output_projection/bias"), vec![256]),
            ("fully_connected/weights".into(), vec![256, 512]),
            ("fully_connected/biases".into(), vec![512]),
        ];
        for (name, shape) in specs {
            tensors.insert(name, empty_tensor(&shape));
        }
        for (i, input) in [(0usize, 774usize), (1, 512), (2, 512)] {
            let path = format!("{prefix}multi_rnn_cell/cell_{i}/basic_lstm_cell");
            tensors.insert(format!("{path}/kernel"), empty_tensor(&[input, 1024]));
            tensors.insert(format!("{path}/bias"), empty_tensor(&[1024]));
        }
        Checkpoint {
            attention_length: 2,
            tensors,
        }
    }

    #[test]
    fn counter_is_signed_and_lsb_first() {
        assert_eq!(binary_counter(0), [1.0, -1.0, -1.0, -1.0, -1.0, -1.0]);
        assert_eq!(binary_counter(1), [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0]);
        assert_eq!(binary_counter(2), [1.0, 1.0, -1.0, -1.0, -1.0, -1.0]);
        assert_eq!(binary_counter(63), [-1.0; 6]);
    }

    #[test]
    fn initial_zero_weights_return_zero_logits() {
        let checkpoint = tiny_zero_checkpoint();
        let model = AttentionRnn::from_checkpoint(&checkpoint).unwrap();
        let mut state = model.zero_state();
        model.step(0, 0, &mut state).unwrap();
        assert_eq!(state.logits().len(), 512);
        assert!(state.logits().iter().all(|value| *value == 0.0));
        model.step(37, 1, &mut state).unwrap();
        assert!(state.logits().iter().all(|value| *value == 0.0));
    }

    #[test]
    fn model_validates_tensor_shapes() {
        let mut checkpoint = tiny_zero_checkpoint();
        checkpoint.tensors.get_mut("fully_connected/weights").unwrap().shape[0] = 255;
        assert!(AttentionRnn::from_checkpoint(&checkpoint).is_err());
    }

    #[test]
    fn softmax_is_normalized_and_temperature_sensitive() {
        let probabilities = softmax(&[1.0, 2.0, 3.0], 1.0).unwrap();
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(probabilities[2] > probabilities[1]);
        let warmer = softmax(&[1.0, 2.0, 3.0], 2.0).unwrap();
        assert!(warmer[2] < probabilities[2]);
        assert!(softmax(&[1.0, 2.0], 0.0).is_err());
    }
}
