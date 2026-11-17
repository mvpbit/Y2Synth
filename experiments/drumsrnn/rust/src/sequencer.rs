//! Platform-independent live loop sequencer, separated from neural inference.
//!
//! A complete one-bar audio buffer is prepared on the control thread. Only
//! samples from the current buffer are read in the audio callback. New patterns
//! and BPM changes are queued to start at the next bar boundary.
//!
//! The real-time state machine lives in `realtime.rs`. This module is
//! intentionally offline/control-thread-only: it may allocate Vecs.

use crate::audio::render_samples_with_kit_at_rate;
use crate::generation::GeneratedPattern;
use crate::kit::DrumKit;
use crate::synth1::{render_bar as render_synth1_bar, SynthTrack};
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
    prepare_loop_with_kit(pattern, bpm, sample_rate, &DrumKit::default())
}

/// Re-render one whole bar with the current drum-kit controls.
/// All allocations and DSP take place on the control thread.
pub fn prepare_loop_with_kit(
    pattern: &GeneratedPattern,
    bpm: f32,
    sample_rate: u32,
    kit: &DrumKit,
) -> io::Result<PreparedLoop> {
    if pattern.classes.len() != STEPS_PER_BAR {
        return Err(invalid("Live prototype needs exactly 16 steps per bar"));
    }
    let full = render_samples_with_kit_at_rate(pattern, bpm, sample_rate, kit)?;
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


/// Prepare a complete bar for the mixed project, preserving the old drums-only
/// renderer and its output when Synth 1 is not present. Each melodic tail is
/// folded into the loop in synth1::render_bar. DSP is done on the control thread.
pub fn prepare_project_loop(
    drums: Option<(&GeneratedPattern, &DrumKit)>,
    synth1: Option<&SynthTrack>,
    bpm: f32,
    sample_rate: u32,
) -> io::Result<PreparedLoop> {
    if drums.is_none() && synth1.is_none() {
        return Err(invalid("Cannot render a project with no tracks"));
    }
    let drum_bar = drums.map(|(pattern, kit)| {
        prepare_loop_with_kit(pattern, bpm, sample_rate, kit)
    }).transpose()?;
    let synth_bar = synth1.map(|track| {
        render_synth1_bar(track, bpm, sample_rate)
    }).transpose()?;

    match (drum_bar, synth_bar) {
        (Some(drums), None) => Ok(drums), // Preserve existing sound and tests.
        (None, Some(samples)) => Ok(PreparedLoop { samples, bpm, sample_rate }),
        (Some(mut drums), Some(synth)) => {
            if drums.samples.len() != synth.len() {
                return Err(invalid("Drums/Synth 1 rendered different bar lengths"));
            }
            // Deliberate instrument balance before global peak protection:
            // keep drums present, while leaving headroom for pad chords.
            for (a, b) in drums.samples.iter_mut().zip(synth) {
                *a = *a * 0.82 + b * 0.74;
            }
            let peak = drums.samples.iter().fold(0.0_f32, |p, &x| p.max(x.abs()));
            if peak > 0.85 {
                for x in &mut drums.samples { *x *= 0.85 / peak; }
            }
            Ok(drums)
        }
        (None, None) => unreachable!("empty project rejected above"),
    }
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
    fn kit_parameter_changes_audio_without_affecting_bar_length() {
        let mut kit = DrumKit::default();
        let baseline = prepare_loop_with_kit(&pattern(), 120.0, 48_000, &kit).unwrap();
        kit.set(36, "level", 0.0).unwrap();
        kit.set(46, "tone", 0.9).unwrap();
        let edited = prepare_loop_with_kit(&pattern(), 120.0, 48_000, &kit).unwrap();
        assert_eq!(baseline.samples.len(), edited.samples.len());
        assert_ne!(baseline.samples, edited.samples);
        assert!(edited.samples.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn project_mix_includes_both_instruments_without_changing_duration() {
        use crate::synth1::{SynthPreset, SynthTrack};
        let kit = DrumKit::default();
        let pluck = SynthTrack::demo(SynthPreset::Pluck);
        let drums_only = prepare_loop_with_kit(&pattern(), 120.0, 8_000, &kit).unwrap();
        let synth_only = prepare_project_loop(None, Some(&pluck), 120.0, 8_000).unwrap();
        let mixed = prepare_project_loop(
            Some((&pattern(), &kit)), Some(&pluck), 120.0, 8_000,
        ).unwrap();
        assert_eq!(drums_only.samples.len(), 16_000);
        assert_eq!(synth_only.samples.len(), 16_000);
        assert_eq!(mixed.samples.len(), 16_000);
        assert_ne!(mixed.samples, drums_only.samples);
        assert_ne!(mixed.samples, synth_only.samples);
        assert!(mixed.samples.iter().all(|v| v.is_finite() && v.abs() <= 0.851));
    }

    #[test]
    fn drums_without_synth_keep_previous_rendering() {
        let kit = DrumKit::default();
        let original = prepare_loop_with_kit(&pattern(), 92.0, 8_000, &kit).unwrap();
        let new = prepare_project_loop(Some((&pattern(), &kit)), None, 92.0, 8_000).unwrap();
        assert_eq!(original.samples, new.samples);
        assert!(prepare_project_loop(None, None, 120.0, 8_000).is_err());
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
