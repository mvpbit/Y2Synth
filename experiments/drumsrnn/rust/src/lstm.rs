//! One step of the TensorFlow 1.x BasicLSTMCell used by Magenta DrumsRNN.
//! Reference: tensorflow/python/keras/layers/legacy_rnn/rnn_cell_impl.py.
//!
//! TensorFlow stores the four gates in [input, candidate, forget, output]
//! order (i,j,f,o). Crucially, the forget gate adds a runtime bias of +1.0.
//! The kernel is row-major [input_size + hidden_size, 4 * hidden_size].
//! This is a scalar, correctness-first implementation, not an optimized DSP kernel.

use crate::{Checkpoint, Tensor};
use std::io::{self, ErrorKind};

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}

/// Borrowed tensors from an already loaded Magenta checkpoint.
pub struct BasicLstmCell<'a> {
    kernel: &'a [f32],
    bias: &'a [f32],
    input_size: usize,
    hidden_size: usize,
}

#[derive(Debug, Clone)]
pub struct LstmState {
    pub cell: Vec<f32>,
    pub hidden: Vec<f32>,
    gates: Vec<f32>,
}

impl LstmState {
    pub fn reset(&mut self) {
        self.cell.fill(0.0);
        self.hidden.fill(0.0);
        self.gates.fill(0.0);
    }
}

impl<'a> BasicLstmCell<'a> {
    pub fn from_tensors(kernel: &'a Tensor, bias: &'a Tensor) -> io::Result<Self> {
        if kernel.shape.len() != 2 || bias.shape.len() != 1 {
            return Err(invalid("BasicLSTMCell kernel must be rank 2 and bias rank 1"));
        }
        let gate_width = bias.shape[0];
        if gate_width == 0 || gate_width % 4 != 0 {
            return Err(invalid("BasicLSTMCell bias must have 4*N elements"));
        }
        let hidden_size = gate_width / 4;
        if kernel.shape[1] != gate_width || kernel.shape[0] < hidden_size {
            return Err(invalid("BasicLSTMCell kernel and bias shapes mismatch"));
        }
        let input_size = kernel.shape[0] - hidden_size;
        if input_size == 0
            || kernel.values.len() != kernel.shape[0].checked_mul(gate_width)
                .ok_or_else(|| invalid("BasicLSTMCell kernel shape overflow"))?
            || bias.values.len() != gate_width
        {
            return Err(invalid("BasicLSTMCell tensor lengths or input size invalid"));
        }
        Ok(Self {
            kernel: &kernel.values,
            bias: &bias.values,
            input_size,
            hidden_size,
        })
    }

    /// Load layer 0, 1, or 2 from the checkpoint using the original TF names.
    pub fn from_checkpoint(checkpoint: &'a Checkpoint, layer: usize) -> io::Result<Self> {
        if layer > 2 {
            return Err(invalid("DrumsRNN has only three BasicLSTMCell layers"));
        }
        let prefix = format!(
            "rnn/attention_cell_wrapper/multi_rnn_cell/cell_{layer}/basic_lstm_cell"
        );
        let kernel = checkpoint
            .tensors
            .get(&format!("{prefix}/kernel"))
            .ok_or_else(|| invalid(format!("Missing {prefix}/kernel")))?;
        let bias = checkpoint
            .tensors
            .get(&format!("{prefix}/bias"))
            .ok_or_else(|| invalid(format!("Missing {prefix}/bias")))?;
        Self::from_tensors(kernel, bias)
    }

    pub fn input_size(&self) -> usize {
        self.input_size
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn zero_state(&self) -> LstmState {
        LstmState {
            cell: vec![0.0; self.hidden_size],
            hidden: vec![0.0; self.hidden_size],
            gates: vec![0.0; 4 * self.hidden_size],
        }
    }

    /// Compute one LSTM step, updating the hidden and cell states in place.
    ///
    /// Preallocated buffers are reused; this method does not allocate memory.
    /// The matrix multiplication is simple scalar code and not yet optimized.
    pub fn step(&self, input: &[f32], state: &mut LstmState) -> io::Result<()> {
        if input.len() != self.input_size
            || state.hidden.len() != self.hidden_size
            || state.cell.len() != self.hidden_size
            || state.gates.len() != 4 * self.hidden_size
        {
            return Err(invalid("BasicLSTMCell input/state dimension mismatch"));
        }
        if input.iter().chain(&state.hidden).chain(&state.cell).any(|v| !v.is_finite()) {
            return Err(invalid("Non-finite BasicLSTMCell input/state"));
        }

        let width = 4 * self.hidden_size;
        state.gates.copy_from_slice(self.bias);
        // Row 0..input_size = input x, followed by prior hidden state h.
        for (row, &x) in input.iter().chain(state.hidden.iter()).enumerate() {
            let kernel_row = &self.kernel[row * width..(row + 1) * width];
            for (gate, &weight) in state.gates.iter_mut().zip(kernel_row) {
                *gate += x * weight;
            }
        }

        let n = self.hidden_size;
        for index in 0..n {
            let input_gate = sigmoid(state.gates[index]);
            let candidate = state.gates[n + index].tanh();
            let forget_gate = sigmoid(state.gates[2 * n + index] + 1.0);
            let output_gate = sigmoid(state.gates[3 * n + index]);
            let new_cell = state.cell[index] * forget_gate + input_gate * candidate;
            let new_hidden = new_cell.tanh() * output_gate;
            if !new_cell.is_finite() || !new_hidden.is_finite() {
                return Err(invalid("Non-finite BasicLSTMCell output"));
            }
            state.cell[index] = new_cell;
            state.hidden[index] = new_hidden;
        }
        Ok(())
    }
}

fn sigmoid(value: f32) -> f32 {
    // Stable for both large positive and negative values.
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-6,
            "expected {expected}, got {actual}"
        );
    }

    fn cell(kernel: Vec<f32>, bias: Vec<f32>, rows: usize, cols: usize) -> (Tensor, Tensor) {
        (
            Tensor {
                shape: vec![rows, cols],
                values: kernel,
            },
            Tensor {
                shape: vec![cols],
                values: bias,
            },
        )
    }

    #[test]
    fn zero_weights_and_zero_biases_hold_zero_state() {
        let (k, b) = cell(vec![0.0; 8], vec![0.0; 4], 2, 4);
        let lstm = BasicLstmCell::from_tensors(&k, &b).unwrap();
        let mut state = lstm.zero_state();
        lstm.step(&[2.0], &mut state).unwrap();
        assert_eq!(state.cell, [0.0]);
        assert_eq!(state.hidden, [0.0]);
    }

    #[test]
    fn forget_bias_is_added_at_runtime() {
        let (k, b) = cell(vec![0.0; 8], vec![0.0; 4], 2, 4);
        let lstm = BasicLstmCell::from_tensors(&k, &b).unwrap();
        let mut state = lstm.zero_state();
        state.cell[0] = 1.0;
        lstm.step(&[0.0], &mut state).unwrap();
        let expected_cell = sigmoid(1.0);
        approx(state.cell[0], expected_cell);
        approx(state.hidden[0], 0.5 * expected_cell.tanh());
    }

    #[test]
    fn gate_order_and_row_major_kernel_match_tf_basic_lstm() {
        // One input and one hidden unit, columns are i,j,f,o.
        // x=2, h_prev=3 -> gates [0,2,0,0]; candidate=tanh(2).
        let (k, b) = cell(
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0; 4],
            2,
            4,
        );
        let lstm = BasicLstmCell::from_tensors(&k, &b).unwrap();
        let mut state = lstm.zero_state();
        state.hidden[0] = 3.0;
        lstm.step(&[2.0], &mut state).unwrap();
        let expected_cell = 0.5 * 2.0f32.tanh();
        approx(state.cell[0], expected_cell);
        approx(state.hidden[0], 0.5 * expected_cell.tanh());
    }

    #[test]
    fn recurrent_hidden_state_affects_next_step() {
        let (k, b) = cell(
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            vec![0.0; 4],
            2,
            4,
        );
        let lstm = BasicLstmCell::from_tensors(&k, &b).unwrap();
        let mut state = lstm.zero_state();
        state.hidden[0] = 1.0;
        lstm.step(&[0.0], &mut state).unwrap();
        let expected_cell = 0.5 * 2.0f32.tanh();
        approx(state.cell[0], expected_cell);
    }

    #[test]
    fn shape_mismatch_and_invalid_input_are_errors() {
        let (k, b) = cell(vec![0.0; 8], vec![0.0; 4], 2, 4);
        let lstm = BasicLstmCell::from_tensors(&k, &b).unwrap();
        assert!(lstm.step(&[], &mut lstm.zero_state()).is_err());
        assert!(lstm.step(&[f32::NAN], &mut lstm.zero_state()).is_err());
        assert!(BasicLstmCell::from_tensors(
            &Tensor { shape: vec![1, 4], values: vec![0.0; 4] },
            &b
        ).is_err());
    }
}
