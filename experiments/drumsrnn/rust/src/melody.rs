//! Autonomous, seeded musical phrase composition for Synth 1.
//!
//! This is a small MUSIC-ALGORITHM generator (tonality, chord motion,
//! question/answer motifs, drum-aware rhythms), NOT a trained neural network.
//! Its interface is intentionally separate from Synth 1 timbre rendering, so
//! a later local ML model can be substituted without changing the audio path.
//! All work happens on the application/control thread.

use crate::generation::GeneratedPattern;
use crate::synth1::{NoteEvent, SynthPreset, SynthTrack, STEPS_PER_BAR};
use std::io::{self, ErrorKind};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RhythmContext {
    pub kicks: [bool; STEPS_PER_BAR],
    pub snares: [bool; STEPS_PER_BAR],
    pub hats: [bool; STEPS_PER_BAR],
}

impl RhythmContext {
    /// Sound-aware but not model-specific: the composer can run without drums.
    /// Use actual decoded MIDI hits; the AI generator is not called here.
    pub fn from_drums(pattern: Option<&GeneratedPattern>) -> Self {
        let mut context = Self::default();
        if let Some(pattern) = pattern {
            for hit in &pattern.hits {
                if hit.step >= STEPS_PER_BAR {
                    continue;
                }
                match hit.pitch {
                    36 => context.kicks[hit.step] = true,
                    38 => context.snares[hit.step] = true,
                    42 | 46 => context.hats[hit.step] = true,
                    _ => {}
                }
            }
        }
        context
    }
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }

    fn next(&mut self) -> u64 {
        // SplitMix64: fixed results across platforms with a given seed.
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }

    fn pick(&mut self, choices: usize) -> usize {
        (self.next() % choices as u64) as usize
    }
}

fn preferred_step(first: usize, second: usize, groove: RhythmContext, rng: &mut Rng) -> usize {
    fn strength(step: usize, groove: RhythmContext) -> i8 {
        i8::from(groove.kicks[step]) * 3
            + i8::from(groove.hats[step])
            - i8::from(groove.snares[step])
    }
    let one = strength(first, groove);
    let two = strength(second, groove);
    if one > two { first }
    else if two > one { second }
    else if rng.pick(2) == 0 { first }
    else { second }
}

const MINOR_TONICS: [u8; 6] = [48, 50, 52, 53, 55, 57];

// Semitone intervals for tonic minor, VI major, iv minor, VII major.
const PROGRESSION: [[u8; 4]; 4] = [
    [0, 3, 7, 10],
    [8, 12, 15, 19],
    [5, 8, 12, 15],
    [10, 14, 17, 21],
];

const MOTIFS: [[usize; 4]; 4] = [
    [0, 1, 2, 1],
    [0, 2, 1, 3],
    [2, 1, 0, 1],
    [0, 1, 3, 2],
];

fn add_note(notes: &mut Vec<NoteEvent>, step: usize, pitch: u8,
            length_steps: usize, velocity: f32) {
    notes.push(NoteEvent { step, pitch, length_steps, velocity });
}

/// Compose exactly one 16-step bar in a minor key with a two-chord movement.
/// seed chooses tonic, progression, motif and rhythmic variations. The source
/// of this track is set to AI by the coordinator *after* audio publication.
/// Retains the current user's instrument preset/level while replacing notes.
pub fn compose(
    seed: u64,
    preset: SynthPreset,
    level: f32,
    groove: RhythmContext,
) -> io::Result<SynthTrack> {
    if !level.is_finite() || !(0.0..=1.0).contains(&level) {
        return Err(io::Error::new(ErrorKind::InvalidInput, "Synth 1 level must be in 0..=1"));
    }
    let mut rng = Rng::new(seed);
    let root = MINOR_TONICS[rng.pick(MINOR_TONICS.len())];
    let second = 1 + rng.pick(3); // VI, iv, VII; not another tonic.
    let chords = [PROGRESSION[0], PROGRESSION[second]];
    let mut notes = Vec::with_capacity(16);

    match preset {
        SynthPreset::Pluck => {
            let motif = MOTIFS[rng.pick(MOTIFS.len())];
            // First and second eight-step halves form question/answer.
            let positions = [
                0, preferred_step(2, 3, groove, &mut rng),
                preferred_step(5, 6, groove, &mut rng), 7,
                8, preferred_step(10, 11, groove, &mut rng),
                preferred_step(13, 14, groove, &mut rng), 15,
            ];
            for (i, &step) in positions.iter().enumerate() {
                let half = i / 4;
                let choice = if half == 1 && i == 7 { 0 } else {
                    motif[i % 4]
                };
                let mut pitch = root + 12 + chords[half][choice];
                // Occasional octave color on upper motif note; still within
                // Synth 1's MIDI range and the current chord.
                if i == 3 && rng.pick(4) == 0 {
                    pitch += 12;
                }
                let accent = groove.kicks[step] || step == 0 || step == 8;
                let velocity = if accent { 0.82 } else { 0.66 };
                let length_steps = if step == 7 || step == 15 { 2 }
                    else { 1 + rng.pick(3) };
                add_note(&mut notes, step, pitch, length_steps, velocity);
            }
        }
        SynthPreset::Pad => {
            // Two broad voicings, including seventh/extension. Slow Pad
            // envelopes already live in the DSP, not in the composer.
            for half in 0..2 {
                let step = half * 8;
                for (i, &semitones) in chords[half].iter().enumerate() {
                    let pitch = root + semitones + if i == 0 { 0 } else { 12 };
                    let velocity = if i == 0 { 0.62 } else { 0.53 };
                    add_note(&mut notes, step, pitch, 8, velocity);
                }
            }
        }
    }

    let track = SynthTrack { preset, notes, level };
    track.validate()?;
    Ok(track)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::{DrumHit, GeneratedPattern};

    #[test]
    fn same_seed_is_fully_reproducible_and_different_seeds_vary() {
        let rhythm = RhythmContext::default();
        let a = compose(1234, SynthPreset::Pluck, 0.7, rhythm).unwrap();
        assert_eq!(a, compose(1234, SynthPreset::Pluck, 0.7, rhythm).unwrap());
        let variants: Vec<_> = (1..16)
            .map(|seed| compose(seed, SynthPreset::Pluck, 0.7, rhythm).unwrap())
            .collect();
        assert!(variants.iter().any(|v| *v != a));
    }

    #[test]
    fn generated_pluck_is_tonal_motivic_and_valid() {
        for seed in 0..64 {
            let track = compose(seed, SynthPreset::Pluck, 0.5, RhythmContext::default()).unwrap();
            assert_eq!(track.preset, SynthPreset::Pluck);
            assert_eq!(track.level, 0.5);
            assert_eq!(track.notes.len(), 8);
            assert_eq!(track.notes[0].step, 0);
            assert_eq!(track.notes[4].step, 8);
            assert!(track.notes.windows(2).all(|pair| pair[0].step < pair[1].step));
            assert!(track.notes.iter().all(|note| (48..=96).contains(&note.pitch)));
            track.validate().unwrap();
        }
    }

    #[test]
    fn generated_pad_contains_two_four_note_chords() {
        for seed in 0..64 {
            let track = compose(seed, SynthPreset::Pad, 0.7, RhythmContext::default()).unwrap();
            assert_eq!(track.notes.len(), 8);
            assert!(track.notes[..4].iter().all(|n| n.step == 0 && n.length_steps == 8));
            assert!(track.notes[4..].iter().all(|n| n.step == 8 && n.length_steps == 8));
            track.validate().unwrap();
        }
    }

    #[test]
    fn drum_hits_affect_rhythmic_emphasis_but_not_drums() {
        let pattern = GeneratedPattern {
            classes: vec![0; 16],
            hits: vec![
                DrumHit { step: 2, pitch: 36 },
                DrumHit { step: 3, pitch: 38 },
                DrumHit { step: 6, pitch: 36 },
                DrumHit { step: 11, pitch: 36 },
                DrumHit { step: 13, pitch: 42 },
            ],
            diagnostics: vec![],
        };
        let groove = RhythmContext::from_drums(Some(&pattern));
        assert!(groove.kicks[2] && groove.snares[3] && groove.hats[13]);
        let melody = compose(33, SynthPreset::Pluck, 0.75, groove).unwrap();
        assert_eq!(melody.notes[1].step, 2);
        assert_eq!(melody.notes[2].step, 6);
        assert_eq!(melody.notes[5].step, 11);
        assert_eq!(pattern.hits.len(), 5);
        assert_eq!(RhythmContext::from_drums(None), RhythmContext::default());
    }

    #[test]
    fn bad_levels_do_not_produce_tracks() {
        for level in [f32::NAN, -0.1, 1.1, f32::INFINITY] {
            assert!(compose(1, SynthPreset::Pluck, level, RhythmContext::default()).is_err());
        }
    }
}
