//! Offline master FX chain for the entire project: MIX -> DELAY -> REVERB.
//!
//! The processor runs exclusively on the control thread after all instrument
//! voices have been mixed. CPAL only reads the finished PreparedLoop, and no
//! per-sample FX state or allocations live in its callback.
//!
//! Each bar is fed repeatedly through the same processor before publishing
//! the final bar. This pre-roll settles the feedback networks into a periodic
//! state so delay/reverb tails continue naturally when the bar loops.

use std::io::{self, ErrorKind};

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DelaySettings {
    pub enabled: bool,
    /// Sixteenth-note units. 2 = eighth note, 4 = quarter note.
    pub steps: u8,
    pub feedback: f32,
    pub wet: f32,
}

impl Default for DelaySettings {
    fn default() -> Self {
        Self { enabled: true, steps: 2, feedback: 0.38, wet: 0.24 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReverbSettings {
    pub enabled: bool,
    /// Controls feedback/decay of the comb filters (0..=1).
    pub size: f32,
    /// Controls high-frequency damping in the feedback loops (0..=1).
    pub damping: f32,
    pub wet: f32,
}

impl Default for ReverbSettings {
    fn default() -> Self {
        Self { enabled: true, size: 0.6, damping: 0.45, wet: 0.2 }
    }
}

/// Project-wide, independent of Drums and Synth 1 tracks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MasterEffects {
    pub delay: DelaySettings,
    pub reverb: ReverbSettings,
}

impl MasterEffects {
    pub fn dry() -> Self {
        let mut effects = Self::default();
        effects.delay.enabled = false;
        effects.reverb.enabled = false;
        effects
    }

    pub fn validate(self) -> io::Result<()> {
        let delay = self.delay;
        let reverb = self.reverb;
        if !(1..=8).contains(&delay.steps) {
            return Err(invalid("Delay steps must be in 1..=8 (sixteenth-note units)"));
        }
        if !delay.feedback.is_finite() || !(0.0..=0.75).contains(&delay.feedback) {
            return Err(invalid("Delay feedback must be finite and in 0..=0.75"));
        }
        if !delay.wet.is_finite() || !(0.0..=0.7).contains(&delay.wet) {
            return Err(invalid("Delay wet must be finite and in 0..=0.7"));
        }
        if !reverb.size.is_finite() || !(0.0..=1.0).contains(&reverb.size) {
            return Err(invalid("Reverb size must be finite and in 0..=1"));
        }
        if !reverb.damping.is_finite() || !(0.0..=1.0).contains(&reverb.damping) {
            return Err(invalid("Reverb damping must be finite and in 0..=1"));
        }
        if !reverb.wet.is_finite() || !(0.0..=0.7).contains(&reverb.wet) {
            return Err(invalid("Reverb wet must be finite and in 0..=0.7"));
        }
        Ok(())
    }
}

struct DelayLine {
    samples: Vec<f32>,
    position: usize,
    feedback: f32,
    wet: f32,
}

impl DelayLine {
    fn new(settings: DelaySettings, bpm: f32, sample_rate: u32) -> Self {
        let seconds = 60.0_f64 / bpm as f64 / 4.0 * settings.steps as f64;
        let length = (seconds * sample_rate as f64).round().max(1.0) as usize;
        Self {
            samples: vec![0.0; length],
            position: 0,
            feedback: settings.feedback,
            wet: settings.wet,
        }
    }

    fn tick(&mut self, dry: f32) -> f32 {
        let echo = self.samples[self.position];
        self.samples[self.position] = dry + self.feedback * echo;
        self.position += 1;
        if self.position == self.samples.len() { self.position = 0; }
        dry * (1.0 - self.wet) + echo * self.wet
    }
}

struct Comb {
    buffer: Vec<f32>,
    pos: usize,
    lowpass: f32,
}

impl Comb {
    fn new(delay_samples: usize) -> Self {
        Self { buffer: vec![0.0; delay_samples.max(1)], pos: 0, lowpass: 0.0 }
    }

    fn tick(&mut self, input: f32, feedback: f32, damping: f32) -> f32 {
        let delayed = self.buffer[self.pos];
        self.lowpass = delayed * (1.0 - damping) + self.lowpass * damping;
        self.buffer[self.pos] = input * 0.3 + self.lowpass * feedback;
        self.pos += 1;
        if self.pos == self.buffer.len() { self.pos = 0; }
        delayed
    }
}

struct AllPass {
    buffer: Vec<f32>,
    pos: usize,
}

impl AllPass {
    fn new(delay_samples: usize) -> Self {
        Self { buffer: vec![0.0; delay_samples.max(1)], pos: 0 }
    }

    fn tick(&mut self, input: f32) -> f32 {
        const FEEDBACK: f32 = 0.5;
        let delayed = self.buffer[self.pos];
        // Canonical Schroeder all-pass: unit-magnitude diffusion stage.
        let written = input + delayed * FEEDBACK;
        let output = delayed - written * FEEDBACK;
        self.buffer[self.pos] = written;
        self.pos += 1;
        if self.pos == self.buffer.len() { self.pos = 0; }
        output
    }
}

struct Reverb {
    combs: [Comb; 4],
    allpasses: [AllPass; 2],
    feedback: f32,
    damping: f32,
    wet: f32,
}

impl Reverb {
    fn new(settings: ReverbSettings, sample_rate: u32) -> Self {
        let delay = |seconds: f64| -> usize {
            (seconds * sample_rate as f64).round().max(1.0) as usize
        };
        Self {
            // Slightly incommensurate delays provide an ambient decay without
            // hard-coded sample-rate-specific coefficients or IR files.
            combs: [
                Comb::new(delay(0.0297)), Comb::new(delay(0.0371)),
                Comb::new(delay(0.0411)), Comb::new(delay(0.0437)),
            ],
            allpasses: [AllPass::new(delay(0.005)), AllPass::new(delay(0.0017))],
            feedback: 0.53 + settings.size * 0.31,
            damping: 0.15 + settings.damping * 0.65,
            wet: settings.wet,
        }
    }

    fn tick(&mut self, dry: f32) -> f32 {
        let mut diffuse = 0.0;
        for comb in &mut self.combs {
            diffuse += comb.tick(dry, self.feedback, self.damping);
        }
        diffuse *= 0.25;
        for allpass in &mut self.allpasses {
            diffuse = allpass.tick(diffuse);
        }
        dry * (1.0 - self.wet) + diffuse * self.wet
    }
}

/// Process an already-mixed *mono* PCM bar through the global serial chain.
///
/// The dry signal comes from all current project tracks mixed together.
/// A fixed 12-bar pre-roll produces a stable periodic FX response before
/// publishing a single bar, rather than truncating tails at a bar boundary.
/// This is a bounded offline prototype, not a real-time stateful FX processor.
/// Peak protection reduces excessive levels, but never normalizes quiet music
/// upwards. Disabled FX are a bit-exact bypass.
pub fn render_master_bar(
    dry: &[f32],
    bpm: f32,
    sample_rate: u32,
    effects: MasterEffects,
) -> io::Result<Vec<f32>> {
    effects.validate()?;
    if dry.is_empty() || dry.iter().any(|sample| !sample.is_finite()) {
        return Err(invalid("Master FX requires a nonempty bar of finite audio"));
    }
    if !bpm.is_finite() || !(40.0..=240.0).contains(&bpm)
        || !(8_000..=192_000).contains(&sample_rate)
    {
        return Err(invalid("Invalid master FX tempo or sample rate"));
    }

    let with_delay = effects.delay.enabled && effects.delay.wet > 0.0;
    let with_reverb = effects.reverb.enabled && effects.reverb.wet > 0.0;
    if !with_delay && !with_reverb {
        return Ok(dry.to_vec());
    }

    let mut delay = with_delay.then(|| DelayLine::new(effects.delay, bpm, sample_rate));
    let mut reverb = with_reverb.then(|| Reverb::new(effects.reverb, sample_rate));
    let mut output = vec![0.0_f32; dry.len()];

    // Delay is limited to half a bar with at most 0.75 feedback; twelve
    // repetitions reduce lingering feedback to a very small level.
    for _ in 0..12 {
        for (index, &sample) in dry.iter().enumerate() {
            let delayed = match &mut delay {
                Some(engine) => engine.tick(sample),
                None => sample,
            };
            output[index] = match &mut reverb {
                Some(engine) => engine.tick(delayed),
                None => delayed,
            };
        }
    }

    if output.iter().any(|sample| !sample.is_finite()) {
        return Err(invalid("Nonfinite master FX output"));
    }
    let peak = output.iter().fold(0.0_f32, |max, &v| max.max(v.abs()));
    if peak > 0.85 {
        let gain = 0.85 / peak;
        for sample in &mut output { *sample *= gain; }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_have_both_effects_and_are_valid() {
        let fx = MasterEffects::default();
        assert!(fx.delay.enabled && fx.reverb.enabled);
        fx.validate().unwrap();
    }

    #[test]
    fn bypass_is_exact_and_does_not_change_gain() {
        let dry = vec![0.1, -0.3, 0.0, 0.6, -0.8];
        let unchanged = render_master_bar(&dry, 120.0, 8_000, MasterEffects::dry()).unwrap();
        assert_eq!(unchanged, dry);
    }

    #[test]
    fn beat_synced_delay_arrives_at_the_expected_sample() {
        let mut fx = MasterEffects::dry();
        fx.delay.enabled = true;
        fx.delay.steps = 2;
        fx.delay.wet = 0.5;
        fx.delay.feedback = 0.0;
        let mut input = vec![0.0; 16_000];
        input[0] = 1.0;
        let output = render_master_bar(&input, 120.0, 8_000, fx).unwrap();
        // Two sixteenth notes = 250 ms at 120 BPM = 2000 samples.
        assert!((output[0] - 0.5).abs() < 1e-6);
        assert!((output[2000] - 0.5).abs() < 1e-6);
        assert_eq!(output[1999], 0.0);
        assert_eq!(output[4000], 0.0);
    }

    #[test]
    fn reverb_produces_a_tail_without_widening_the_bar() {
        let mut fx = MasterEffects::dry();
        fx.reverb.enabled = true;
        fx.reverb.wet = 0.4;
        let mut input = vec![0.0; 16_000];
        input[0] = 0.7;
        let output = render_master_bar(&input, 120.0, 8_000, fx).unwrap();
        assert_eq!(output.len(), input.len());
        assert!(output[200..1600].iter().any(|x| x.abs() > 1e-5));
        assert!(output.iter().all(|x| x.is_finite() && x.abs() <= 0.851));
    }

    #[test]
    fn full_chain_changes_result_from_either_stage_alone() {
        let mut input = vec![0.0; 16_000];
        input[0] = 0.55;
        let fx = MasterEffects::default();
        let serial = render_master_bar(&input, 120.0, 8_000, fx).unwrap();
        let mut without_delay = fx;
        without_delay.delay.enabled = false;
        let mut without_reverb = fx;
        without_reverb.reverb.enabled = false;
        assert_ne!(serial, render_master_bar(&input, 120.0, 8_000, without_delay).unwrap());
        assert_ne!(serial, render_master_bar(&input, 120.0, 8_000, without_reverb).unwrap());
    }

    #[test]
    fn tempo_changes_delay_timing() {
        let mut fx = MasterEffects::dry();
        fx.delay.enabled = true;
        fx.delay.steps = 2;
        fx.delay.feedback = 0.0;
        fx.delay.wet = 0.5;
        let mut slow = vec![0.0; 24_000];
        slow[0] = 0.6;
        let mut fast = vec![0.0; 16_000];
        fast[0] = 0.6;
        let one = render_master_bar(&fast, 120.0, 8_000, fx).unwrap();
        let two = render_master_bar(&slow, 80.0, 8_000, fx).unwrap();
        assert!(one[2000] > 0.2);
        assert!(two[3000] > 0.2);
    }

    #[test]
    fn invalid_parameters_rejected_including_when_disabled() {
        let mut fx = MasterEffects::dry();
        fx.delay.feedback = f32::NAN;
        assert!(fx.validate().is_err());
        fx.delay.feedback = 0.4;
        fx.delay.steps = 9;
        assert!(fx.validate().is_err());
        fx.delay.steps = 2;
        fx.reverb.size = f32::INFINITY;
        assert!(fx.validate().is_err());
        assert!(render_master_bar(&[], 120.0, 8_000, MasterEffects::dry()).is_err());
        assert!(render_master_bar(&[f32::NAN], 120.0, 8_000, MasterEffects::dry()).is_err());
    }
}
