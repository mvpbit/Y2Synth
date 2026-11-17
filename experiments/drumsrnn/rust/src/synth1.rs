//! Synth 1: two musical instruments, not raw oscillator waveform presets.
//!
//! PLUCK is a deterministic Karplus–Strong-style damped delay string.
//! PAD uses layered, slightly detuned additive partials with slow envelopes
//! and subtle motion. Rendering is entirely offline / on the CONTROL thread.
//! No allocations, model inference or DSP from here run in the CPAL callback.

use std::f64::consts::TAU;
use std::io::{self, ErrorKind};

pub const STEPS_PER_BAR: usize = 16;
const MIN_PITCH: u8 = 36;
const MAX_PITCH: u8 = 96;
const MAX_NOTES: usize = 48;

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SynthPreset {
    Pluck,
    Pad,
}

impl SynthPreset {
    pub fn name(self) -> &'static str {
        match self {
            Self::Pluck => "Pluck",
            Self::Pad => "Pad",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoteEvent {
    /// Zero-based sixteenth-note position in a 16-step bar.
    pub step: usize,
    pub pitch: u8,
    /// Note gate length in sixteenth notes (tails may continue after gate-off).
    pub length_steps: usize,
    /// Note velocity in 0..=1.
    pub velocity: f32,
}

impl NoteEvent {
    pub fn validate(self) -> io::Result<()> {
        if self.step >= STEPS_PER_BAR || self.length_steps == 0
            || self.length_steps > STEPS_PER_BAR
            || self.pitch < MIN_PITCH || self.pitch > MAX_PITCH
            || !self.velocity.is_finite() || !(0.0..=1.0).contains(&self.velocity)
        {
            return Err(invalid("Invalid Synth 1 note: step 0..15, MIDI pitch 36..96, length 1..16, velocity 0..1"));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SynthTrack {
    pub preset: SynthPreset,
    pub notes: Vec<NoteEvent>,
    /// Instrument volume (0 means muted).
    pub level: f32,
}

impl SynthTrack {
    pub fn demo(preset: SynthPreset) -> Self {
        let notes = match preset {
            // Six little plucked notes spanning C minor, leaving space for drums.
            SynthPreset::Pluck => [(0, 60, 2), (3, 67, 2), (6, 63, 2),
                (9, 70, 2), (12, 67, 2), (14, 63, 2)]
                .into_iter()
                .map(|(step, pitch, length_steps)| NoteEvent {
                    step, pitch, length_steps, velocity: 0.8,
                })
                .collect(),
            // Two sustained chords: Cm -> Ab, softly crossfading through the bar.
            SynthPreset::Pad => [(0, 60), (0, 63), (0, 67),
                (8, 56), (8, 60), (8, 63)]
                .into_iter()
                .map(|(step, pitch)| NoteEvent {
                    step, pitch, length_steps: 8, velocity: 0.64,
                })
                .collect(),
        };
        Self { preset, notes, level: 0.75 }
    }

    pub fn validate(&self) -> io::Result<()> {
        if !self.level.is_finite() || !(0.0..=1.0).contains(&self.level) {
            return Err(invalid("Synth 1 level must be finite and in 0..=1"));
        }
        if self.notes.len() > MAX_NOTES {
            return Err(invalid("Synth 1 supports at most 48 note events per bar"));
        }
        for &note in &self.notes {
            note.validate()?;
        }
        Ok(())
    }
}

fn hz(pitch: u8) -> f64 {
    440.0 * 2.0_f64.powf((pitch as f64 - 69.0) / 12.0)
}

fn next_noise(seed: &mut u32) -> f64 {
    // A tiny repeatable xorshift noise source; no random or I/O dependencies.
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    (*seed as f64 / u32::MAX as f64) * 2.0 - 1.0
}

fn fade_end(frame: usize, frames: usize, rate: u32) -> f64 {
    let fade = (rate as usize / 100).max(1).min(frames); // final 10 ms
    if frame < frames - fade { 1.0 }
    else { (frames - frame) as f64 / fade as f64 }
}

fn render_pluck(
    samples: &mut [f32], start: usize, note: NoteEvent, rate: u32, step_sec: f64, level: f32,
) {
    let frequency = hz(note.pitch);
    let delay_len = (rate as f64 / frequency).round().max(3.0) as usize;
    let mut delay = vec![0.0_f64; delay_len];
    let mut seed = 0x9e37_79b9u32
        ^ ((note.pitch as u32) << 16) ^ (note.step as u32).wrapping_mul(0x85eb_ca6b);
    let mut smooth = 0.0;
    for value in &mut delay {
        // Pick excitation with filtered broadband noise, not a basic saw/square.
        smooth = 0.58 * smooth + 0.42 * next_noise(&mut seed);
        *value = smooth;
    }
    let mean = delay.iter().sum::<f64>() / delay_len as f64;
    for value in &mut delay { *value -= mean; }

    let gate_seconds = note.length_steps as f64 * step_sec;
    let tail_seconds = (0.85 + 0.55 * gate_seconds).min(2.1);
    let frames = ((tail_seconds * rate as f64).ceil() as usize)
        .min(samples.len().saturating_sub(start));
    let mut pointer = 0usize;
    let mut last = 0.0;
    let gain = (level * note.velocity * 0.38) as f64;
    for frame in 0..frames {
        let old = delay[pointer];
        // Averaged delay-line feedback emulates a lossy, plucked string.
        let filtered = 0.5 * (old + last) * 0.996;
        delay[pointer] = filtered;
        last = old;
        pointer = (pointer + 1) % delay_len;
        let t = frame as f64 / rate as f64;
        let envelope = (-t / (tail_seconds * 0.46)).exp()
            * fade_end(frame, frames, rate);
        samples[start + frame] += (old * gain * envelope) as f32;
    }
}

fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

fn render_pad(
    samples: &mut [f32], start: usize, note: NoteEvent, rate: u32, step_sec: f64, level: f32,
) {
    let fundamental = hz(note.pitch);
    let gate = note.length_steps as f64 * step_sec;
    let attack = 0.32;
    let release = 0.9;
    let frames = (((gate + release) * rate as f64).ceil() as usize)
        .min(samples.len().saturating_sub(start));
    let gain = (level * note.velocity * 0.16) as f64;

    for frame in 0..frames {
        let t = frame as f64 / rate as f64;
        let onset = smoothstep(t / attack);
        let release_env = if t < gate { 1.0 }
            else { 1.0 - smoothstep((t - gate) / release) };
        let envelope = onset * release_env * fade_end(frame, frames, rate);
        // Three gently detuned voices. The partial balance moves over time,
        // making a breathing texture rather than a fixed oscillator waveform.
        let lfo = (TAU * 0.31 * t).sin();
        let vibrato = (TAU * 4.1 * t).sin() * 0.0012;
        let brightness = (0.52 + 0.19 * (TAU * 0.23 * t).sin()).max(0.1);
        let mut voice = 0.0;
        for detune in [-0.0036, 0.0, 0.0036] {
            let phase = TAU * fundamental * (1.0 + detune + vibrato) * t;
            voice += phase.sin()
                + (2.0 * phase + 0.35 * lfo).sin() * brightness * 0.31
                + (3.0 * phase + 0.7 * lfo).sin() * brightness * 0.12;
        }
        samples[start + frame] += (voice * gain * envelope / 3.0) as f32;
    }
}

/// Produce exactly one loop bar (16 sixteenth notes), with late voice tails
/// wrapped into the next repetition. Never called from the audio callback.
pub fn render_bar(track: &SynthTrack, bpm: f32, sample_rate: u32) -> io::Result<Vec<f32>> {
    track.validate()?;
    if !bpm.is_finite() || !(40.0..=240.0).contains(&bpm) {
        return Err(invalid("Synth BPM must be finite and in 40..=240"));
    }
    if !(8_000..=192_000).contains(&sample_rate) {
        return Err(invalid("Synth sample rate must be 8000..=192000"));
    }
    let step_sec = 60.0 / bpm as f64 / 4.0;
    let frames = (step_sec * STEPS_PER_BAR as f64 * sample_rate as f64).round() as usize;
    let tail_frames = (2.2 * sample_rate as f64).ceil() as usize;
    let mut full = vec![0.0_f32; frames + tail_frames];
    for &note in &track.notes {
        let start = (note.step as f64 * step_sec * sample_rate as f64).round() as usize;
        match track.preset {
            SynthPreset::Pluck => render_pluck(
                &mut full, start, note, sample_rate, step_sec, track.level,
            ),
            SynthPreset::Pad => render_pad(
                &mut full, start, note, sample_rate, step_sec, track.level,
            ),
        }
    }
    let mut bar = full[..frames].to_vec();
    for (i, &value) in full[frames..].iter().enumerate() {
        bar[i % frames] += value;
    }
    // Scale down ONLY when needed; turning a track down stays meaningful.
    let peak = bar.iter().fold(0.0_f32, |p, &x| p.max(x.abs()));
    if peak > 0.8 {
        for x in &mut bar { *x *= 0.8 / peak; }
    }
    Ok(bar)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_actual_musical_phrases() {
        let pluck = SynthTrack::demo(SynthPreset::Pluck);
        let pad = SynthTrack::demo(SynthPreset::Pad);
        assert!(pluck.notes.len() >= 4);
        assert!(pad.notes.len() >= 4);
        assert_ne!(pluck.notes, pad.notes);
        assert!(pluck.validate().is_ok());
        assert!(pad.validate().is_ok());
    }

    #[test]
    fn both_presets_produce_finite_audio_with_one_bar_length() {
        for preset in [SynthPreset::Pluck, SynthPreset::Pad] {
            let track = SynthTrack::demo(preset);
            let bar = render_bar(&track, 120.0, 8_000).unwrap();
            assert_eq!(bar.len(), 16_000);
            assert!(bar.iter().all(|x| x.is_finite() && x.abs() <= 0.801));
            assert!(bar.iter().any(|x| x.abs() > 0.01));
        }
    }

    #[test]
    fn pad_has_a_slow_attack_and_pluck_a_fast_attack() {
        let event = NoteEvent { step: 0, pitch: 60, length_steps: 8, velocity: 0.8 };
        let mut pluck = SynthTrack::demo(SynthPreset::Pluck);
        pluck.notes = vec![event];
        let mut pad = SynthTrack::demo(SynthPreset::Pad);
        pad.notes = vec![event];
        let plucked = render_bar(&pluck, 120.0, 8_000).unwrap();
        let padded = render_bar(&pad, 120.0, 8_000).unwrap();
        let peak = |s: &[f32]| s.iter().fold(0.0_f32, |v, &x| v.max(x.abs()));
        assert!(peak(&plucked[0..200]) > peak(&padded[0..200]));
        assert!(peak(&padded[0..200]) < peak(&padded[2_000..4_000]));
    }

    #[test]
    fn mute_and_empty_notes_are_silent() {
        let mut track = SynthTrack::demo(SynthPreset::Pad);
        track.notes.clear();
        assert!(render_bar(&track, 110.0, 8_000).unwrap().iter().all(|&x| x == 0.0));
        track.notes = SynthTrack::demo(SynthPreset::Pluck).notes;
        track.level = 0.0;
        assert!(render_bar(&track, 110.0, 8_000).unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn tuning_and_bpm_change_audio_without_nan() {
        let track = SynthTrack::demo(SynthPreset::Pluck);
        let a = render_bar(&track, 120.0, 8_000).unwrap();
        let b = render_bar(&track, 80.0, 8_000).unwrap();
        assert_ne!(a.len(), b.len());
        let mut shifted = track.clone();
        shifted.notes[0].pitch += 1;
        let c = render_bar(&shifted, 120.0, 8_000).unwrap();
        assert_ne!(a, c);
    }

    #[test]
    fn bad_inputs_are_rejected() {
        let mut track = SynthTrack::demo(SynthPreset::Pad);
        track.level = f32::NAN;
        assert!(render_bar(&track, 120.0, 8_000).is_err());
        track.level = 0.75;
        track.notes[0].step = 16;
        assert!(track.validate().is_err());
        track.notes[0].step = 0;
        track.notes[0].pitch = 12;
        assert!(track.validate().is_err());
        track.notes[0].pitch = 60;
        track.notes[0].length_steps = 0;
        assert!(track.validate().is_err());
        track.notes[0].length_steps = 8;
        track.notes[0].velocity = f32::INFINITY;
        assert!(track.validate().is_err());
        assert!(render_bar(&SynthTrack::demo(SynthPreset::Pluck), 250.0, 8_000).is_err());
        assert!(render_bar(&SynthTrack::demo(SynthPreset::Pluck), 120.0, 100).is_err());
    }
}
