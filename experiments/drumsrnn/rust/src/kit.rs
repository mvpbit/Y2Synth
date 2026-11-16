//! Drum-kit settings shared by the WAV preview and the live sequencer.
//! Settings are applied when the control thread renders a new PCM loop.

use crate::generation::DRUM_PITCHES;
use std::io::{self, ErrorKind};

pub const VOICE_NAMES: [&str; 9] = [
    "kick", "snare", "chh", "ohh", "lowtom", "midtom", "hightom", "crash", "ride",
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoiceParams {
    /// Linear amplitude; 0 is mute, 1 is nominal, 1.5 is boost.
    pub level: f32,
    /// Timbre: 0 dark/deep .. 1 bright/high.
    pub tone: f32,
    /// Envelope duration multiplier: 0.5 short .. 2.0 long.
    pub decay: f32,
}

impl Default for VoiceParams {
    fn default() -> Self {
        Self { level: 1.0, tone: 0.5, decay: 1.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrumKit {
    voices: [VoiceParams; 9],
}

impl Default for DrumKit {
    fn default() -> Self {
        Self { voices: [VoiceParams::default(); 9] }
    }
}

impl DrumKit {
    pub fn voices(&self) -> &[VoiceParams; 9] {
        &self.voices
    }

    pub fn voice(&self, pitch: u8) -> Option<VoiceParams> {
        drum_index(pitch).map(|index| self.voices[index])
    }

    /// Mutates only after validating all arguments.
    pub fn set(&mut self, pitch: u8, parameter: &str, value: f32) -> io::Result<()> {
        let index = drum_index(pitch).ok_or_else(|| invalid("Unknown drum MIDI pitch"))?;
        if !value.is_finite() {
            return Err(invalid("Drum parameter must be a finite number"));
        }
        match parameter {
            "level" if (0.0..=1.5).contains(&value) => self.voices[index].level = value,
            "tone" if (0.0..=1.0).contains(&value) => self.voices[index].tone = value,
            "decay" if (0.5..=2.0).contains(&value) => self.voices[index].decay = value,
            "level" => return Err(invalid("level must be 0..=1.5")),
            "tone" => return Err(invalid("tone must be 0..=1")),
            "decay" => return Err(invalid("decay must be 0.5..=2")),
            _ => return Err(invalid("Unknown parameter (level/tone/decay)")),
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn validate(&self) -> io::Result<()> {
        for (index, params) in self.voices.iter().enumerate() {
            for (name, value, min, max) in [
                ("level", params.level, 0.0, 1.5),
                ("tone", params.tone, 0.0, 1.0),
                ("decay", params.decay, 0.5, 2.0),
            ] {
                if !value.is_finite() || value < min || value > max {
                    return Err(invalid(format!(
                        "{} {name} must be finite and between {min} and {max}",
                        VOICE_NAMES[index],
                    )));
                }
            }
        }
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message.into())
}

pub fn drum_index(pitch: u8) -> Option<usize> {
    DRUM_PITCHES.iter().position(|&candidate| candidate == pitch)
}

/// Recognizes short names suitable for terminal input, plus common aliases.
pub fn pitch_by_name(name: &str) -> Option<u8> {
    let normalized = name.trim().to_ascii_lowercase().replace('-', "").replace('_', "");
    let index = match normalized.as_str() {
        "kick" | "bd" => 0,
        "snare" | "sd" => 1,
        "chh" | "hat" | "closedhat" | "closedhh" => 2,
        "ohh" | "openhat" | "openhh" => 3,
        "lowtom" | "tom1" => 4,
        "midtom" | "tom2" => 5,
        "hightom" | "tom3" => 6,
        "crash" => 7,
        "ride" => 8,
        _ => return None,
    };
    Some(DRUM_PITCHES[index])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_voice_has_a_name_and_pitch() {
        let kit = DrumKit::default();
        kit.validate().unwrap();
        for (index, &name) in VOICE_NAMES.iter().enumerate() {
            let pitch = pitch_by_name(name).unwrap();
            assert_eq!(pitch, DRUM_PITCHES[index]);
            assert_eq!(kit.voice(pitch), Some(VoiceParams::default()));
        }
        assert_eq!(pitch_by_name("open-hat"), Some(46));
        assert_eq!(pitch_by_name("hi_hat"), None);
        assert_eq!(pitch_by_name("SNARE"), Some(38));
    }

    #[test]
    fn params_validate_without_partial_mutation() {
        let mut kit = DrumKit::default();
        kit.set(36, "level", 0.3).unwrap();
        kit.set(38, "tone", 1.0).unwrap();
        kit.set(42, "decay", 2.0).unwrap();
        assert_eq!(kit.voice(36).unwrap().level, 0.3);
        assert_eq!(kit.voice(38).unwrap().tone, 1.0);
        assert_eq!(kit.voice(42).unwrap().decay, 2.0);
        let before = kit;
        assert!(kit.set(36, "level", f32::NAN).is_err());
        assert!(kit.set(36, "level", -1.0).is_err());
        assert!(kit.set(36, "tone", 1.1).is_err());
        assert!(kit.set(36, "decay", 0.1).is_err());
        assert!(kit.set(36, "release", 1.0).is_err());
        assert!(kit.set(127, "level", 1.0).is_err());
        assert_eq!(kit, before);
    }

    #[test]
    fn reset_restores_defaults() {
        let mut kit = DrumKit::default();
        kit.set(36, "level", 0.0).unwrap();
        kit.reset();
        assert_eq!(kit, DrumKit::default());
    }
}
