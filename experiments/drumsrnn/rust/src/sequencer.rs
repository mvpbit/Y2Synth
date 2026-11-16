//! Platform-independent live loop sequencer, separated from neural inference.
//!
//! A complete one-bar audio buffer is prepared on the control thread. Only
//! samples from the current buffer are read in the audio callback. New patterns
//! and BPM changes are queued to start at the next bar boundary.
//!
//! This is a prototype: CPAL's callback still needs RT profiling on target
//! hardware, especially around dropping old buffers at bar boundaries.

use crate::audio::render_samples_at_rate;
use crate::generation::GeneratedPattern;
use std::io::{self, ErrorKind};

const STEPS_PER_BAR: usize = 16;
const FADE_SECONDS: f32 = 0.015;
const MASTER_LEVEL: f32 = 0.75;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CueResult {
    Now,
    NextBar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Transport {
    Stopped,
    Playing,
    Stopping,
}

/// Deterministic mono sample engine, tested without an audio device.
///
/// Commands are called from the CPAL callback after draining its control
/// channel. No model inference or synthesis happens in tick().
pub struct LiveSequencer {
    sample_rate: u32,
    current: Option<PreparedLoop>,
    pending: Option<PreparedLoop>,
    fading_from: Option<PreparedLoop>,
    position: usize,
    crossfade_position: usize,
    crossfade_frames: usize,
    gain: f32,
    transport: Transport,
}

impl LiveSequencer {
    pub fn new(sample_rate: u32) -> io::Result<Self> {
        if !(8_000..=192_000).contains(&sample_rate) {
            return Err(invalid("Invalid output sample rate"));
        }
        Ok(Self {
            sample_rate,
            current: None,
            pending: None,
            fading_from: None,
            position: 0,
            crossfade_position: 0,
            crossfade_frames: ((sample_rate as f32 * FADE_SECONDS).round() as usize)
                .max(1),
            gain: 0.0,
            transport: Transport::Stopped,
        })
    }

    pub fn cue(&mut self, next: PreparedLoop) -> io::Result<CueResult> {
        if next.sample_rate != self.sample_rate || next.samples.is_empty() {
            return Err(invalid("Loop does not match output stream sample rate"));
        }
        if self.transport == Transport::Stopped || self.current.is_none() {
            self.pending = None;
            self.fading_from = None;
            self.position = 0;
            self.current = Some(next);
            Ok(CueResult::Now)
        } else {
            // Last requested pattern wins if multiple requests arrive in a bar.
            self.pending = Some(next);
            Ok(CueResult::NextBar)
        }
    }

    pub fn play(&mut self) {
        if self.current.is_some() {
            self.position = 0;
            self.fading_from = None;
            self.crossfade_position = 0;
            self.gain = 0.0;
            self.transport = Transport::Playing;
        }
    }

    pub fn stop(&mut self) {
        if self.transport == Transport::Playing {
            self.transport = Transport::Stopping;
        }
    }

    pub fn is_playing(&self) -> bool {
        self.transport == Transport::Playing
    }

    fn swap_at_bar(&mut self) {
        if let Some(next) = self.pending.take() {
            let previous = self.current.replace(next);
            self.fading_from = previous;
            self.crossfade_position = 0;
        }
    }

    /// One mono output sample. The audio stream duplicates it to each channel.
    pub fn tick(&mut self) -> f32 {
        if self.transport == Transport::Stopped {
            return 0.0;
        }
        if self.position == 0 && self.transport == Transport::Playing {
            self.swap_at_bar();
        }

        let Some(current) = self.current.as_ref() else {
            return 0.0;
        };
        let mut sample = current.samples[self.position];
        if let Some(previous) = self.fading_from.as_ref() {
            let index = self.crossfade_position % previous.samples.len();
            let old_sample = previous.samples[index];
            let alpha = (self.crossfade_position as f32 / self.crossfade_frames as f32)
                .clamp(0.0, 1.0);
            sample = old_sample * (1.0 - alpha) + sample * alpha;
            self.crossfade_position += 1;
            if self.crossfade_position >= self.crossfade_frames {
                self.fading_from = None;
            }
        }

        self.position += 1;
        if self.position == current.samples.len() {
            self.position = 0;
        }

        let gain_step = 1.0 / self.crossfade_frames as f32;
        match self.transport {
            Transport::Playing => self.gain = (self.gain + gain_step).min(1.0),
            Transport::Stopping => {
                self.gain = (self.gain - gain_step).max(0.0);
                if self.gain == 0.0 {
                    self.transport = Transport::Stopped;
                    self.position = 0;
                    self.fading_from = None;
                }
            }
            Transport::Stopped => {}
        }

        (sample * self.gain * MASTER_LEVEL).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::{DrumHit, StepDiagnostic};

    fn manual_pattern() -> GeneratedPattern {
        let mut classes = vec![0; 16];
        classes[0] = 1;
        classes[15] = 8;
        GeneratedPattern {
            classes,
            hits: vec![
                DrumHit { step: 0, pitch: 36 },
                DrumHit { step: 15, pitch: 46 },
            ],
            diagnostics: Vec::<StepDiagnostic>::new(),
        }
    }

    fn constant_loop(value: f32, size: usize) -> PreparedLoop {
        PreparedLoop { samples: vec![value; size], bpm: 120.0, sample_rate: 8_000 }
    }

    #[test]
    fn prepared_loop_matches_one_bar_duration() {
        let loop_data = prepare_loop(&manual_pattern(), 120.0, 8_000).unwrap();
        assert_eq!(loop_data.samples.len(), 16_000);
        assert!(loop_data.samples.iter().all(|v| v.is_finite() && v.abs() <= 0.86));
    }

    #[test]
    fn tempo_changes_one_bar_duration() {
        let a = prepare_loop(&manual_pattern(), 120.0, 8_000).unwrap();
        let b = prepare_loop(&manual_pattern(), 80.0, 8_000).unwrap();
        assert_eq!(a.samples.len(), 16_000);
        assert_eq!(b.samples.len(), 24_000);
    }

    #[test]
    fn cue_switches_only_at_bar_boundary_while_playing() {
        let mut engine = LiveSequencer::new(8_000).unwrap();
        assert_eq!(engine.cue(constant_loop(0.2, 1000)).unwrap(), CueResult::Now);
        engine.play();
        for _ in 0..100 {
            engine.tick();
        }
        assert_eq!(engine.position, 100);
        assert_eq!(
            engine.cue(constant_loop(0.7, 1000)).unwrap(),
            CueResult::NextBar
        );
        for _ in 0..899 {
            engine.tick();
        }
        assert_eq!(engine.position, 999);
        assert_eq!(engine.current.as_ref().unwrap().samples[0], 0.2);
        engine.tick();
        assert_eq!(engine.position, 0);
        assert_eq!(engine.current.as_ref().unwrap().samples[0], 0.2);
        engine.tick();
        assert_eq!(engine.current.as_ref().unwrap().samples[0], 0.7);
    }

    #[test]
    fn latest_pending_pattern_wins() {
        let mut engine = LiveSequencer::new(8_000).unwrap();
        engine.cue(constant_loop(0.1, 200)).unwrap();
        engine.play();
        engine.tick();
        engine.cue(constant_loop(0.3, 200)).unwrap();
        engine.cue(constant_loop(0.5, 200)).unwrap();
        for _ in 0..200 {
            engine.tick();
        }
        assert_eq!(engine.current.as_ref().unwrap().samples[0], 0.5);
    }

    #[test]
    fn stop_fades_to_silence_and_play_restarts() {
        let mut engine = LiveSequencer::new(8_000).unwrap();
        assert_eq!(engine.tick(), 0.0);
        engine.cue(constant_loop(0.7, 1000)).unwrap();
        engine.play();
        for _ in 0..150 {
            engine.tick();
        }
        assert!(engine.is_playing());
        engine.stop();
        for _ in 0..200 {
            engine.tick();
        }
        assert!(!engine.is_playing());
        assert_eq!(engine.tick(), 0.0);
        engine.play();
        assert!(engine.is_playing());
        assert_eq!(engine.position, 0);
    }

    #[test]
    fn rejects_mismatched_sample_rate() {
        let mut engine = LiveSequencer::new(8_000).unwrap();
        assert!(engine.cue(PreparedLoop {
            samples: vec![0.0; 100],
            bpm: 120.0,
            sample_rate: 44_100,
        }).is_err());
    }
}
