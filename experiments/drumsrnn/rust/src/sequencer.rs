//! Platform-independent live loop sequencer, separated from neural inference.
//!
//! A complete one-bar audio buffer is prepared on the control thread. Only
//! samples from the current buffer are read in the audio callback. New patterns
//! and BPM changes are queued to start at the next bar boundary.
//!
//! The real-time state machine lives in `realtime.rs`. This module is
//! intentionally offline/control-thread-only: it may allocate Vecs.

use crate::audio::render_samples_at_rate;
use crate::generation::GeneratedPattern;
use std::io::{self, ErrorKind};

const STEPS_PER_BAR: usize = 16;

fn invalid(message: &str) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message)
}

#[derive(Clone)]
pub struct PreparedLoop {
    pub samples: Vec<f32>,
    pub bpm: f32,
    pub sample_rate: u32,
}

/// Render all drum voices *off* the audio callback thread. Fold their decay
/// tails into the beginning of the next bar, giving seamless repeated loops.
/// A different pattern crossfades to this one at the next bar boundary.
pub fn prepare_loop(
    pattern: &GeneratedPattern,
    bpm: f32,
    sample_rate: u32,
) -> io::Result<PreparedLoop> {
    if pattern.classes.len() != STEPS_PER_BAR {
        return Err(invalid("Live prototype needs exactly 16 steps per bar"));
    }
    let full = render_samples_at_rate(pattern, bpm, sample_rate)?;
    let bar_frames = (60.0f64 / bpm as f64 * 4.0 * sample_rate as f64)
        .round() as usize;
    if bar_frames == 0 || full.len() < bar_frames {
        return Err(invalid("Invalid rendered bar duration"));
    }

    let mut samples = full[..bar_frames].to_vec();
    for (index, &value) in full[bar_frames..].iter().enumerate() {
        samples[index % bar_frames] += value;
    }

    // Folding tails can exceed 0.8 even though the base renderer normalizes.
    // Keep headroom for the brief crossfades at bar transitions.
    let peak = samples.iter().fold(0.0f32, |p, &v| p.max(v.abs()));
    if peak > 0.85 {
        for value in &mut samples {
            *value *= 0.85 / peak;
        }
    }
    Ok(PreparedLoop {
        samples, bpm, sample_rate,
    })
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::DrumHit;

    fn pattern() -> GeneratedPattern {
        let mut classes = vec![0; 16];
        classes[0] = 1;
        classes[15] = 8;
        GeneratedPattern {
            classes,
            hits: vec![
                DrumHit { step: 0, pitch: 36 },
                DrumHit { step: 15, pitch: 46 },
            ],
            diagnostics: vec![],
        }
    }

    #[test]
    fn duration_is_one_bar() {
        let bar = prepare_loop(&pattern(), 120.0, 8_000).unwrap();
        assert_eq!(bar.samples.len(), 16_000);
        assert!(bar.samples.iter().all(|&v| v.is_finite() && v.abs() <= 0.851));
    }

    #[test]
    fn bpm_controls_prepared_audio_length() {
        let fast = prepare_loop(&pattern(), 120.0, 8_000).unwrap();
        let slow = prepare_loop(&pattern(), 80.0, 8_000).unwrap();
        assert_eq!(fast.samples.len(), 16_000);
        assert_eq!(slow.samples.len(), 24_000);
    }

    #[test]
    fn audio_is_prepared_at_device_sample_rate() {
        let bar = prepare_loop(&pattern(), 120.0, 48_000).unwrap();
        assert_eq!(bar.samples.len(), 96_000);
        assert_eq!(bar.sample_rate, 48_000);
        assert!(bar.samples.iter().any(|&v| v.abs() > 0.05));
    }

    #[test]
    fn no_input_groove_is_required() {
        let silent = GeneratedPattern {
            classes: vec![0; 16],
            hits: vec![],
            diagnostics: vec![],
        };
        let bar = prepare_loop(&silent, 120.0, 8_000).unwrap();
        assert!(bar.samples.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn rejects_wrong_step_count_and_invalid_bpm() {
        let mut short = pattern();
        short.classes.pop();
        assert!(prepare_loop(&short, 120.0, 8_000).is_err());
        assert!(prepare_loop(&pattern(), 0.0, 8_000).is_err());
        assert!(prepare_loop(&pattern(), 250.0, 8_000).is_err());
    }
}
