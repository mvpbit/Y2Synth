//! Application coordinator between pure menu intent, musical state, AI and audio.
//!
//! No CPAL, hardware input or model checkpoint paths here. The control thread
//! executes operations, prepares a complete PCM bar and publishes it through
//! AudioPort; the real-time callback remains independent.
//!
//! Only the DRUMS AI generator exists today. Synth1/Synth2 and Generate All
//! return explicit 'not implemented' outcomes without mutating any track.

use crate::generation::GeneratedPattern;
use crate::kit::DrumKit;
use crate::project::{CreationMode, MusicSession, TrackId};
use crate::sequencer::{prepare_loop_with_kit, PreparedLoop};
use crate::ui::Command;
use std::io;

/// Explicitly uninhabited placeholder. We cannot accidentally put fake synth
/// notes in a project while the two synth engines are unimplemented.
#[derive(Clone, Debug, PartialEq)]
pub enum UnavailableSynth {}

#[derive(Clone, Debug)]
pub struct DrumTrack {
    pub pattern: GeneratedPattern,
    pub kit: DrumKit,
}

pub type AppSession = MusicSession<UnavailableSynth, UnavailableSynth, DrumTrack>;

/// A port for any platform's non-RT output handoff. Implementation lives
/// in the live application, not in the portable coordinator.
pub trait AudioPort {
    fn queue_bar(&mut self, bar: PreparedLoop) -> io::Result<()>;
    fn start(&mut self) -> io::Result<()>;
}

/// AI service boundary. Implementable by the real AttentionRnn or by a
/// deterministic fake for isolated tests.
pub trait DrumGenerator {
    fn generate_drums(&mut self) -> io::Result<GeneratedPattern>;
}

impl<F: FnMut() -> io::Result<GeneratedPattern>> DrumGenerator for F {
    fn generate_drums(&mut self) -> io::Result<GeneratedPattern> {
        self()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AppResponse {
    /// The saved project was queued and audio was told to start.
    PlayingLast { mode: CreationMode, bpm: f32 },
    /// A new drum pattern was queued for the next bar (or immediately if stopped).
    DrumGenerated { hits: usize },
    /// Drum editor exists only as the current CLI's kit controls.
    EditDrums,
    NotImplemented(&'static str),
    NothingSaved,
}

pub struct ApplicationController {
    session: AppSession,
    sample_rate: u32,
}

impl ApplicationController {
    pub fn new(sample_rate: u32, bpm: f32) -> io::Result<Self> {
        if !(8_000..=192_000).contains(&sample_rate) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Invalid sample rate"));
        }
        let mut session = AppSession::default();
        session.current_mut().set_bpm(bpm).map_err(invalid)?;
        Ok(Self { session, sample_rate })
    }

    pub fn session(&self) -> &AppSession {
        &self.session
    }

    /// The initial drum pattern is already AI-generated on the control thread.
    /// Start-up prepares audio and saves a RAM snapshot; it does NOT auto-play.
    pub fn install_initial_drums<A: AudioPort>(
        &mut self,
        pattern: GeneratedPattern,
        kit: DrumKit,
        audio: &mut A,
    ) -> io::Result<()> {
        let bar = prepare_loop_with_kit(
            &pattern, self.session.current().bpm(), self.sample_rate, &kit,
        )?;
        audio.queue_bar(bar)?;
        self.session.current_mut().replace_drums(
            DrumTrack { pattern, kit }, CreationMode::Ai,
        );
        self.session.save_current();
        Ok(())
    }

    /// All menu commands are requests handled on the control thread. A
    /// failed generator, render or output handoff leaves the saved state alone.
    pub fn dispatch<G: DrumGenerator, A: AudioPort>(
        &mut self,
        command: Command,
        generator: &mut G,
        audio: &mut A,
    ) -> io::Result<AppResponse> {
        match command {
            Command::PlayLast => {
                let Some(saved) = self.session.last_saved() else {
                    return Ok(AppResponse::NothingSaved);
                };
                let Some(drums) = saved.drums() else {
                    return Ok(AppResponse::NothingSaved);
                };
                let bar = prepare_loop_with_kit(
                    &drums.content.pattern, saved.bpm(), self.sample_rate, &drums.content.kit,
                )?;
                audio.queue_bar(bar)?;
                audio.start()?;
                Ok(AppResponse::PlayingLast { mode: saved.mode(), bpm: saved.bpm() })
            }
            Command::GenerateTrack(TrackId::Drums) => {
                // Preserve the current kit when generating a different beat.
                let kit = self.session.current().drums()
                    .map(|track| track.content.kit)
                    .unwrap_or_default();
                let pattern = generator.generate_drums()?;
                let hits = pattern.hits.len();
                let bar = prepare_loop_with_kit(
                    &pattern, self.session.current().bpm(), self.sample_rate, &kit,
                )?;
                audio.queue_bar(bar)?;
                self.session.current_mut().replace_drums(
                    DrumTrack { pattern, kit }, CreationMode::Ai,
                );
                self.session.save_current();
                Ok(AppResponse::DrumGenerated { hits })
            }
            Command::GenerateAll => Ok(AppResponse::NotImplemented(
                "Generate All requires both future AI synth generators",
            )),
            Command::GenerateTrack(TrackId::Synth1 | TrackId::Synth2) => {
                Ok(AppResponse::NotImplemented("AI synth generator not implemented"))
            }
            Command::EditUserTrack(TrackId::Drums) => Ok(AppResponse::EditDrums),
            Command::EditUserTrack(TrackId::Synth1 | TrackId::Synth2) => {
                Ok(AppResponse::NotImplemented("Synth editor not implemented"))
            }
        }
    }

    /// Change common BPM without regenerating AI or touching the callback.
    /// The tempo and RAM snapshot change only after the PCM queue accepts it.
    pub fn set_bpm<A: AudioPort>(&mut self, bpm: f32, audio: &mut A) -> io::Result<()> {
        // Check without mutating the current project.
        if !bpm.is_finite() || !(40.0..=240.0).contains(&bpm) {
            return Err(invalid("BPM must be finite and in 40..=240"));
        }
        if let Some(drums) = self.session.current().drums() {
            let bar = prepare_loop_with_kit(
                &drums.content.pattern, bpm, self.sample_rate, &drums.content.kit,
            )?;
            audio.queue_bar(bar)?;
        }
        self.session.current_mut().set_bpm(bpm).map_err(invalid)?;
        if self.session.current().has_content() {
            self.session.save_current();
        }
        Ok(())
    }

    /// Re-render the current drum pattern with a user-edited kit.
    /// User edit changes the most recent project mode to USER, as recorded by
    /// MusicProject; track contents are preserved.
    pub fn set_drum_kit<A: AudioPort>(
        &mut self,
        kit: DrumKit,
        audio: &mut A,
    ) -> io::Result<bool> {
        let Some(existing) = self.session.current().drums() else {
            return Ok(false);
        };
        let bar = prepare_loop_with_kit(
            &existing.content.pattern, self.session.current().bpm(), self.sample_rate, &kit,
        )?;
        audio.queue_bar(bar)?;
        let mut updated = existing.content.clone();
        updated.kit = kit;
        self.session.current_mut().replace_drums(updated, CreationMode::User);
        self.session.save_current();
        Ok(true)
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::DrumHit;
    use crate::ui::Command;

    fn pattern(pitch: u8) -> GeneratedPattern {
        let mut classes = vec![0; 16];
        classes[0] = 1;
        GeneratedPattern {
            classes,
            hits: vec![DrumHit { step: 0, pitch }],
            diagnostics: vec![],
        }
    }

    #[derive(Default)]
    struct TestAudio {
        queued: Vec<PreparedLoop>,
        starts: usize,
        reject_queue: bool,
    }

    impl AudioPort for TestAudio {
        fn queue_bar(&mut self, bar: PreparedLoop) -> io::Result<()> {
            if self.reject_queue {
                return Err(io::Error::new(io::ErrorKind::BrokenPipe, "device closed"));
            }
            self.queued.push(bar);
            Ok(())
        }
        fn start(&mut self) -> io::Result<()> {
            self.starts += 1;
            Ok(())
        }
    }

    #[test]
    fn play_without_saved_music_never_calls_generator_or_audio() {
        let mut app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut audio = TestAudio::default();
        let mut generator = || -> io::Result<GeneratedPattern> {
            panic!("PLAY must never generate");
        };
        assert_eq!(app.dispatch(Command::PlayLast, &mut generator, &mut audio).unwrap(),
            AppResponse::NothingSaved);
        assert_eq!(audio.starts, 0);
        assert!(audio.queued.is_empty());
    }

    #[test]
    fn initial_pattern_is_saved_without_auto_play() {
        let mut app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut audio = TestAudio::default();
        app.install_initial_drums(pattern(36), DrumKit::default(), &mut audio).unwrap();
        assert_eq!(audio.queued.len(), 1);
        assert_eq!(audio.starts, 0);
        assert_eq!(app.session().last_saved().unwrap().mode(), CreationMode::Ai);
        assert_eq!(app.session().last_saved().unwrap().drums().unwrap().content.pattern.hits[0].pitch, 36);
    }

    #[test]
    fn generate_drums_saves_new_pattern_and_play_uses_it_without_regenerating() {
        let mut app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut audio = TestAudio::default();
        app.install_initial_drums(pattern(36), DrumKit::default(), &mut audio).unwrap();
        let mut calls = 0usize;
        let mut generator = || {
            calls += 1;
            Ok(pattern(38))
        };
        assert_eq!(app.dispatch(Command::GenerateTrack(TrackId::Drums), &mut generator, &mut audio).unwrap(),
            AppResponse::DrumGenerated { hits: 1 });
        assert_eq!(calls, 1);
        assert_eq!(app.session().last_saved().unwrap().drums().unwrap().content.pattern.hits[0].pitch, 38);
        assert_eq!(app.dispatch(Command::PlayLast, &mut generator, &mut audio).unwrap(),
            AppResponse::PlayingLast { mode: CreationMode::Ai, bpm: 120.0 });
        assert_eq!(calls, 1);
        assert_eq!(audio.starts, 1);
        assert_eq!(audio.queued.len(), 3);
    }

    #[test]
    fn unsupported_actions_do_not_touch_project_or_generator() {
        let mut app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut audio = TestAudio::default();
        app.install_initial_drums(pattern(36), DrumKit::default(), &mut audio).unwrap();
        let mut generator = || -> io::Result<GeneratedPattern> { panic!("unexpected generation") };
        for command in [
            Command::GenerateAll,
            Command::GenerateTrack(TrackId::Synth1),
            Command::GenerateTrack(TrackId::Synth2),
            Command::EditUserTrack(TrackId::Synth1),
        ] {
            assert!(matches!(app.dispatch(command, &mut generator, &mut audio).unwrap(),
                AppResponse::NotImplemented(_)));
        }
        assert_eq!(audio.queued.len(), 1);
        assert_eq!(app.session().last_saved().unwrap().drums().unwrap().content.pattern.hits[0].pitch, 36);
    }

    #[test]
    fn failed_generation_or_output_does_not_replace_saved_pattern() {
        let mut app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut audio = TestAudio::default();
        app.install_initial_drums(pattern(36), DrumKit::default(), &mut audio).unwrap();
        let mut failed = || Err(io::Error::new(io::ErrorKind::Other, "generation failed"));
        assert!(app.dispatch(Command::GenerateTrack(TrackId::Drums), &mut failed, &mut audio).is_err());
        audio.reject_queue = true;
        let mut succeeding = || Ok(pattern(38));
        assert!(app.dispatch(Command::GenerateTrack(TrackId::Drums), &mut succeeding, &mut audio).is_err());
        assert_eq!(app.session().last_saved().unwrap().drums().unwrap().content.pattern.hits[0].pitch, 36);
    }

    #[test]
    fn kit_and_tempo_are_saved_and_restored_by_play_last() {
        let mut app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut audio = TestAudio::default();
        app.install_initial_drums(pattern(36), DrumKit::default(), &mut audio).unwrap();
        let mut kit = DrumKit::default();
        kit.set(36, "tone", 0.9).unwrap();
        assert!(app.set_drum_kit(kit, &mut audio).unwrap());
        app.set_bpm(90.0, &mut audio).unwrap();
        let mut unexpected = || -> io::Result<GeneratedPattern> { panic!("no regeneration expected") };
        assert_eq!(app.dispatch(Command::PlayLast, &mut unexpected, &mut audio).unwrap(),
            AppResponse::PlayingLast { mode: CreationMode::User, bpm: 90.0 });
        assert_eq!(audio.queued.last().unwrap().bpm, 90.0);
        assert_eq!(app.session().last_saved().unwrap().drums().unwrap().content.kit, kit);
    }

    #[test]
    fn tempo_and_kit_changes_roll_back_if_output_is_closed() {
        let mut app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut audio = TestAudio::default();
        app.install_initial_drums(pattern(36), DrumKit::default(), &mut audio).unwrap();
        audio.reject_queue = true;
        assert!(app.set_bpm(90.0, &mut audio).is_err());
        assert_eq!(app.session().current().bpm(), 120.0);
        let mut kit = DrumKit::default();
        kit.set(36, "tone", 0.8).unwrap();
        assert!(app.set_drum_kit(kit, &mut audio).is_err());
        assert_eq!(app.session().current().drums().unwrap().content.kit, DrumKit::default());
    }

    #[test]
    fn invalid_app_bpm_is_rejected_before_audio_queue() {
        assert!(ApplicationController::new(0, 120.0).is_err());
        assert!(ApplicationController::new(8_000, f32::NAN).is_err());
        let mut app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut audio = TestAudio::default();
        assert!(app.set_bpm(241.0, &mut audio).is_err());
        assert_eq!(app.session().current().bpm(), 120.0);
    }
}
