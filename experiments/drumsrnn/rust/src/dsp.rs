//! Small deterministic percussive synthesizer (no external samples).
//! All oscillator/filter/envelope work is OFFLINE, on the control thread.
//!
//! This is a musical prototype rather than an analog-modeling engine.
//! The live callback only reads the completed floating-point PCM buffer.

use crate::generation::DrumHit;
use crate::kit::VoiceParams;
use std::f32::consts::{PI, TAU};

pub fn voice_duration(pitch: u8, settings: VoiceParams) -> f32 {
    let base = match pitch {
        36 => 0.48,
        38 => 0.40,
        42 => 0.17,
        46 => 0.70,
        45 | 48 | 50 => 0.52,
        49 => 1.15,
        51 => 0.95,
        _ => 0.25,
    };
    base * settings.decay
}

/// SplitMix64 noise, seeded individually per hit so repeated renders match.
fn noise_next(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut x = *state;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    let n = (x ^ (x >> 31)) >> 40;
    (n as f32) * (2.0 / 16_777_216.0) - 1.0
}

fn one_pole_alpha(cutoff_hz: f32, sample_rate: f32) -> f32 {
    let pole = 1.0 - (-TAU * cutoff_hz / sample_rate).exp();
    pole.clamp(0.0, 1.0)
}

fn metallic_signal(t: f32, brightness: f32) -> f32 {
    // Inharmonic partials evoke a group of small metallic oscillators.
    const RATIOS: [f32; 6] = [1.00, 1.34, 1.71, 2.21, 2.63, 3.11];
    let base = 540.0 + brightness * 330.0;
    let mut total = 0.0;
    for (index, ratio) in RATIOS.iter().enumerate() {
        let frequency = (base * ratio).min(12_000.0);
        let phase_offset = index as f32 * 0.63;
        total += (TAU * t * frequency + phase_offset).sin();
    }
    total / RATIOS.len() as f32
}

/// Render an isolated drum voice into a destination buffer.
/// The destination contains the sum of other voices; we add, never overwrite.
pub fn render_hit(
    samples: &mut [f32],
    start: usize,
    hit: DrumHit,
    sample_rate: u32,
    settings: VoiceParams,
) {
    if start >= samples.len() || settings.level <= 0.0 {
        return;
    }

    let sr = sample_rate as f32;
    let length = ((voice_duration(hit.pitch, settings) * sr).ceil() as usize)
        .min(samples.len() - start);
    let mut rng = 0x8691_31e3_98b4_14a7u64
        ^ ((hit.step as u64) << 16)
        ^ hit.pitch as u64;
    let tone = settings.tone;
    let decay = settings.decay;

    let mut phase = 0.0f32;
    let mut noise_lp = 0.0f32;
    let mut previous_noise = 0.0f32;
    // Filter noise depending on tone. High tone removes more low-frequency noise.
    let noise_cutoff = match hit.pitch {
        36 => 1100.0 + tone * 2400.0,
        38 => 1200.0 + tone * 4000.0,
        _ => 2000.0 + tone * 6800.0,
    };
    let alpha = one_pole_alpha(noise_cutoff, sr);

    for index in 0..length {
        let t = index as f32 / sr;
        let noise = noise_next(&mut rng);
        noise_lp += alpha * (noise - noise_lp);
        let high_noise = noise - noise_lp;
        let transient = noise - previous_noise * 0.75;
        previous_noise = noise;

        let voice = match hit.pitch {
            36 => {
                // Kick: descending sine body + very short bright beater click.
                let base_hz = 43.0 + tone * 28.0;
                let sweep = 160.0 + tone * 130.0;
                let frequency = base_hz + sweep * (-t * 32.0).exp();
                phase += TAU * frequency / sr;
                let body = phase.sin() * (-t * (12.5 / decay)).exp();
                let click = transient * (-t * 125.0).exp();
                0.90 * body + (0.08 + tone * 0.18) * click
            }
            38 => {
                // Snare: resonant low drum shell plus broadband snare-wire noise.
                let frequency = 170.0 + tone * 70.0;
                phase += TAU * frequency / sr;
                let shell = phase.sin() * (-t * (22.0 / decay)).exp();
                let wires = high_noise * (-t * (15.0 / decay)).exp();
                let crack = transient * (-t * 90.0).exp();
                0.25 * shell + 0.46 * wires + (0.08 + tone * 0.10) * crack
            }
            42 | 46 => {
                // Hats: metallic inharmonic oscillators blended with high-pass noise.
                let closed = hit.pitch == 42;
                let decay_rate = if closed { 48.0 } else { 8.5 };
                let envelope = (-t * decay_rate / decay).exp();
                let metal = metallic_signal(t, tone);
                let bright = high_noise * (0.29 + tone * 0.14);
                (metal * 0.34 + bright) * envelope
            }
            45 | 48 | 50 => {
                // Toms: descending sine with harmonic shell and a short attack.
                let base = match hit.pitch {
                    45 => 102.0,
                    48 => 145.0,
                    _ => 198.0,
                };
                let frequency = base * (0.87 + tone * 0.25)
                    + 65.0 * (-t * 25.0).exp();
                phase += TAU * frequency / sr;
                let shell = phase.sin() + 0.18 * (phase * 1.95).sin();
                0.67 * shell * (-t * (11.0 / decay)).exp()
                    + 0.08 * transient * (-t * 90.0).exp()
            }
            49 | 51 => {
                // Cymbals: long metal + bright wash; ride has a more defined ping.
                let crash = hit.pitch == 49;
                let envelope_rate = if crash { 3.5 } else { 5.2 };
                let envelope = (-t * envelope_rate / decay).exp();
                let metal = metallic_signal(t, tone);
                let ping = (TAU * t * (1950.0 + tone * 900.0)).sin()
                    * (-t * 16.0 / decay).exp();
                0.37 * high_noise * envelope
                    + 0.21 * metal * envelope
                    + if crash { 0.0 } else { 0.19 * ping }
            }
            _ => high_noise * (-t * 36.0 / decay).exp() * 0.20,
        };

        // A short attack suppresses discontinuities when a hit begins.
        let attack = (t / 0.0007).min(1.0);
        let output = voice * settings.level * attack;
        samples[start + index] += output;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesis_is_deterministic() {
        let hit = DrumHit { step: 2, pitch: 36 };
        let mut a = vec![0.0; 24_000];
        let mut b = vec![0.0; 24_000];
        render_hit(&mut a, 0, hit, 48_000, VoiceParams::default());
        render_hit(&mut b, 0, hit, 48_000, VoiceParams::default());
        assert_eq!(a, b);
        assert!(a.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn muted_voice_is_silent_and_never_erases_existing_audio() {
        let mut samples = vec![0.2f32; 9000];
        render_hit(
            &mut samples, 0,
            DrumHit { step: 0, pitch: 38 }, 44_100,
            VoiceParams { level: 0.0, ..VoiceParams::default() },
        );
        assert!(samples.iter().all(|&x| x == 0.2));
    }

    #[test]
    fn all_nine_drum_voices_generate_finite_audio() {
        for pitch in [36, 38, 42, 46, 45, 48, 50, 49, 51] {
            let mut samples = vec![0.0f32; 72_000];
            render_hit(
                &mut samples, 0,
                DrumHit { step: 0, pitch }, 48_000,
                VoiceParams::default(),
            );
            assert!(samples.iter().all(|x| x.is_finite()), "pitch {pitch}");
            assert!(samples.iter().any(|x| x.abs() > 0.01), "pitch {pitch}");
        }
    }

    #[test]
    fn tone_and_decay_change_the_waveform_and_tail() {
        let hit = DrumHit { step: 0, pitch: 42 };
        let mut a = vec![0.0; 25_000];
        let mut b = vec![0.0; 25_000];
        render_hit(&mut a, 0, hit, 48_000, VoiceParams::default());
        render_hit(
            &mut b, 0, hit, 48_000,
            VoiceParams { tone: 0.9, decay: 2.0, ..VoiceParams::default() },
        );
        assert_ne!(a, b);
        assert!(voice_duration(42, VoiceParams { decay: 2.0, ..VoiceParams::default() })
            > voice_duration(42, VoiceParams::default()));
    }
}
