//! Autonomous drum-pattern generation from a *silent internal bootstrap step*.
//!
//! Follows music/src/music_rnn/model.ts sampleRnn() in Magenta.js:
//! consume silent class 0 once, sample from next-event logits, then feed each
//! sampled class back to the RNN, except after the final sample.
//! The model is trained; we do not require a user-written drum primer.

use crate::attention::{softmax, AttentionRnn};
use std::io::{self, ErrorKind};

pub const DRUM_PITCHES: [u8; 9] = [36, 38, 42, 46, 45, 48, 50, 49, 51];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrumHit {
    /// Zero-based step index, relative to the generated pattern (not seed).
    pub step: usize,
    pub pitch: u8,
}

#[derive(Debug, Clone, Copy)]
pub enum Sampling {
    Argmax,
    Temperature(f32),
}

#[derive(Debug, Clone)]
pub struct StepDiagnostic {
    pub step: usize,
    pub top_five: [(usize, f32); 5],
}

#[derive(Debug, Clone)]
pub struct GeneratedPattern {
    /// One class (0..511) per generated step, bitmask of nine drum instruments.
    pub classes: Vec<usize>,
    /// MIDI pitches emitted from each step's selected class.
    pub hits: Vec<DrumHit>,
    /// Top-five class probabilities at selected steps for comparison to Magenta.js.
    pub diagnostics: Vec<StepDiagnostic>,
}

/// SplitMix64: a small, reproducible generator for multinomial *sampling*.
/// No cryptographic properties are required; this is not used to train the network.
#[derive(Debug, Clone)]
struct SamplingRng {
    state: u64,
}

impl SamplingRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
    fn unit_f64(&mut self) -> f64 {
        // 53 random bits: 0.0 <= value < 1.0
        ((self.next_u64() >> 11) as f64) / 9_007_199_254_740_992.0
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message.into())
}

pub fn decode_class(class: usize, step: usize) -> io::Result<Vec<DrumHit>> {
    if class >= 512 {
        return Err(invalid("Drum class must be in 0..512"));
    }
    Ok(DRUM_PITCHES.iter().enumerate()
        .filter(|(bit, _)| (class & (1usize << *bit)) != 0)
        .map(|(_, &pitch)| DrumHit { step, pitch })
        .collect())
}

fn top_five(probs: &[f32]) -> [(usize, f32); 5] {
    let mut top = [(usize::MAX, f32::NEG_INFINITY); 5];
    for (index, &p) in probs.iter().enumerate() {
        for rank in 0..top.len() {
            if p > top[rank].1 {
                for next_rank in (rank + 1..top.len()).rev() {
                    top[next_rank] = top[next_rank - 1];
                }
                top[rank] = (index, p);
                break;
            }
        }
    }
    top
}

fn draw_class(probs: &[f32], rng: &mut SamplingRng) -> io::Result<usize> {
    if probs.len() != 512 {
        return Err(invalid("Expected a 512-class categorical distribution"));
    }
    let draw = rng.unit_f64();
    let mut total = 0.0f64;
    for (index, &prob) in probs.iter().enumerate() {
        total += prob as f64;
        if draw < total {
            return Ok(index);
        }
    }
    // Small floating-point summation differences may make sum slightly < 1.
    Ok(probs.len() - 1)
}

fn argmax(logits: &[f32]) -> io::Result<usize> {
    if logits.len() != 512 || logits.iter().any(|v| !v.is_finite()) {
        return Err(invalid("Expected 512 finite class logits"));
    }
    // Match tf.argMax's first-index behavior on ties.
    Ok(logits.iter().enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
        .expect("512 class logits are nonempty").0)
}

/// Generate a new drum pattern using only the weights and a silent bootstrap.
///
/// The class probabilities for argmax are displayed at temperature 1.0,
/// matching Magenta.js's reference page. A positive temperature uses
/// multinomial sampling (results can vary with the provided seed).
pub fn generate(
    network: &AttentionRnn<'_>,
    steps: usize,
    sampling: Sampling,
    seed: u64,
) -> io::Result<GeneratedPattern> {
    generate_with_observer(network, steps, sampling, seed, |_, _| Ok(()))
}

/// The same generation path as `generate`, exposing the complete 512-class
/// distribution before sampling on every step. The observer runs off the audio
/// thread; normal generation does not retain full probability matrices.
///
/// This is particularly useful for a deterministic Argmax trace against
/// Magenta.js `continueSequenceAndReturnProbabilities()`.
pub fn generate_with_observer<F>(
    network: &AttentionRnn<'_>,
    steps: usize,
    sampling: Sampling,
    seed: u64,
    mut observe: F,
) -> io::Result<GeneratedPattern>
where
    F: FnMut(usize, &[f32]) -> io::Result<()>,
{
    if steps == 0 || steps > 4096 {
        return Err(invalid("Steps must be between 1 and 4096"));
    }
    if let Sampling::Temperature(t) = sampling {
        if !t.is_finite() || t <= 0.0 {
            return Err(invalid("Temperature must be finite and > 0"));
        }
    }

    let mut state = network.zero_state();
    // This is the ONE internal, silent seed step, not a human-written beat.
    network.step(0, 0, &mut state)?;

    let mut rng = SamplingRng::new(seed);
    let mut pattern = GeneratedPattern {
        classes: Vec::with_capacity(steps),
        hits: Vec::new(),
        diagnostics: Vec::new(),
    };

    for step in 0..steps {
        let logits = state.logits();
        let temperature = match sampling {
            Sampling::Argmax => 1.0,
            Sampling::Temperature(t) => t,
        };
        let probabilities = softmax(logits, temperature)?;
        // Observer sees all 512 values on the exact same recurrent path
        // as ordinary generation; no additional inference is performed.
        observe(step, &probabilities)?;
        let selected = match sampling {
            Sampling::Argmax => argmax(logits)?,
            Sampling::Temperature(_) => draw_class(&probabilities, &mut rng)?,
        };
        // Diagnostic steps 0,1,7,15 allow recurrence+attention verification
        // without saving the full model probabilities or any JSON artifact.
        if step == 0 || step == 1 || step == 7 || step == 15 {
            pattern.diagnostics.push(StepDiagnostic {
                step,
                top_five: top_five(&probabilities),
            });
        }
        pattern.classes.push(selected);
        pattern.hits.extend(decode_class(selected, step)?);

        // Exactly as in Magenta.js: don't advance recurrent state after
        // sampling the final class (the sampled class is already the output).
        if step + 1 < steps {
            network.step(selected, step + 1, &mut state)?;
        }
    }
    Ok(pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_class_has_no_hits() {
        assert!(decode_class(0, 0).unwrap().is_empty());
    }

    #[test]
    fn single_bits_map_to_upstream_magenta_pitches() {
        for (index, pitch) in DRUM_PITCHES.iter().enumerate() {
            assert_eq!(
                decode_class(1 << index, 12).unwrap(),
                vec![DrumHit { step: 12, pitch: *pitch }]
            );
        }
    }

    #[test]
    fn all_nine_drums_and_invalid_class() {
        let hits = decode_class(511, 3).unwrap();
        assert_eq!(hits.len(), 9);
        assert!(decode_class(512, 0).is_err());
    }

    #[test]
    fn rng_is_reproducible_and_stays_in_unit_interval() {
        let mut a = SamplingRng::new(1234);
        let mut b = SamplingRng::new(1234);
        for _ in 0..1000 {
            let x = a.unit_f64();
            assert!(x >= 0.0 && x < 1.0);
            assert_eq!(x, b.unit_f64());
        }
    }

    #[test]
    fn top_five_orders_probabilities_descending() {
        let mut p = vec![0.0; 512];
        p[256] = 0.25;
        p[4] = 0.30;
        p[0] = 0.40;
        p[2] = 0.04;
        p[1] = 0.01;
        let top = top_five(&p);
        assert_eq!(top.map(|(i, _)| i), [0, 4, 256, 2, 1]);
    }

    #[test]
    fn argmax_picks_first_class_on_ties() {
        let mut logits = vec![-100.0; 512];
        logits[4] = 3.0;
        logits[2] = 3.0;
        assert_eq!(argmax(&logits).unwrap(), 2);
    }
}
