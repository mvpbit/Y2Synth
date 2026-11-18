//! Application coordinator between pure menu intent, musical state, AI and audio.
//!
//! No CPAL, hardware input or model checkpoint paths here. The control thread
//! executes operations, prepares a complete PCM bar and publishes it through
//! AudioPort; the real-time callback remains independent.
//!
//! Three independent tracks share a single master FX chain. Neural services
//! run outside the audio callback; publish/save is atomic.

use crate::generation::GeneratedPattern;
use crate::kit::DrumKit;
use crate::master_fx::{render_master_bar, MasterEffects};
use crate::melody::RhythmContext;
use crate::project::{CreationMode, MusicProject, MusicSession, TrackId};
use crate::sequencer::{prepare_three_track_loop, PreparedLoop};
use crate::synth1::{SynthPreset, SynthTrack};
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

pub type AppProject = MusicProject<SynthTrack, SynthTrack, DrumTrack>;
pub type AppSession = MusicSession<SynthTrack, SynthTrack, DrumTrack>;

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

/// Replaceable composition boundary. Unlike DrumsRNN, the first version is
/// deterministic rule-based composition; a neural service can later implement
/// this trait without changing MenuController or real-time audio code.
pub trait Synth1Generator {
    fn generate_synth1(
        &mut self, preset: SynthPreset, level: f32, rhythm: RhythmContext,
    ) -> io::Result<SynthTrack>;
}

impl<F: FnMut(SynthPreset, f32, RhythmContext) -> io::Result<SynthTrack>>
    Synth1Generator for F
{
    fn generate_synth1(
        &mut self, preset: SynthPreset, level: f32, rhythm: RhythmContext,
    ) -> io::Result<SynthTrack> {
        self(preset, level, rhythm)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AppResponse {
    /// The saved project was queued and audio was told to start.
    PlayingLast { mode: CreationMode, bpm: f32 },
    /// A new drum pattern was queued for the next bar (or immediately if stopped).
    DrumGenerated { hits: usize },
    /// A new autonomous melody/chord phrase, not a pretrained ML prediction.
    Synth1Generated { notes: usize, preset: SynthPreset },
    /// Drum editor exists as kit controls in the CLI.
    EditDrums,
    /// First USER synthesizer: Pluck/Pad and note editor on the control thread.
    EditSynth1,
    EditSynth2,
    Synth2Generated { notes: usize },
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

    fn prepare_project(&self, project: &AppProject) -> io::Result<PreparedLoop> {
        let drums = project.drums()
            .map(|track| (&track.content.pattern, &track.content.kit));
        let synth1 = project.synth1().map(|track| &track.content);
        let synth2 = project.synth2().map(|track| &track.content);
        let mut bar = prepare_three_track_loop(drums, synth1, synth2, project.bpm(), self.sample_rate)?;
        // A single serial insert after all track mixing: Drums + Synths
        // -> tempo-synced delay -> room reverb -> ready PCM loop.
        bar.samples = render_master_bar(
            &bar.samples, project.bpm(), self.sample_rate, project.master_fx(),
        )?;
        Ok(bar)
    }

    /// The initial drum pattern is already AI-generated on the control thread.
    /// Start-up prepares audio and saves a RAM snapshot; it does NOT auto-play.
    pub fn install_initial_drums<A: AudioPort>(
        &mut self,
        pattern: GeneratedPattern,
        kit: DrumKit,
        audio: &mut A,
    ) -> io::Result<()> {
        let mut proposed = self.session.current().clone();
        proposed.replace_drums(DrumTrack { pattern, kit }, CreationMode::Ai);
        let bar = self.prepare_project(&proposed)?;
        audio.queue_bar(bar)?;
        *self.session.current_mut() = proposed;
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
                if !saved.has_content() {
                    return Ok(AppResponse::NothingSaved);
                }
                let bar = self.prepare_project(saved)?;
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
                let mut proposed = self.session.current().clone();
                proposed.replace_drums(DrumTrack { pattern, kit }, CreationMode::Ai);
                let bar = self.prepare_project(&proposed)?;
                audio.queue_bar(bar)?;
                *self.session.current_mut() = proposed;
                self.session.save_current();
                Ok(AppResponse::DrumGenerated { hits })
            }
            Command::GenerateAll => Ok(AppResponse::NotImplemented(
                "Generate All requires Synth 2 and coordinated multi-track generation",
            )),
            Command::GenerateTrack(TrackId::Synth1) => Ok(AppResponse::NotImplemented(
                "Synth 1 composition requires dispatch_with_synth",
            )),
            Command::GenerateTrack(TrackId::Synth2) => Ok(AppResponse::NotImplemented(
                "Synth 2 generator not implemented",
            )),
            Command::EditUserTrack(TrackId::Drums) => Ok(AppResponse::EditDrums),
            Command::EditUserTrack(TrackId::Synth1) => Ok(AppResponse::EditSynth1),
            Command::EditUserTrack(TrackId::Synth2) => {
                Ok(AppResponse::NotImplemented("Synth 2 editor not implemented"))
            }
        }
    }

    /// Full application dispatch: adds composition of Synth 1 to the legacy
    /// Drums-only `dispatch` path. Old clients remain backwards compatible.
    /// Both generator services run ONLY on the control thread.
    pub fn dispatch_with_synth<G: DrumGenerator, S: Synth1Generator, A: AudioPort>(
        &mut self,
        command: Command,
        drums: &mut G,
        synth: &mut S,
        audio: &mut A,
    ) -> io::Result<AppResponse> {
        if command == Command::GenerateTrack(TrackId::Synth1) {
            let existing = self.session.current().synth1();
            // An existing USER or AI patch stays chosen on regeneration.
            let preset = existing
                .map(|track| track.content.preset)
                .unwrap_or(SynthPreset::Pluck);
            let level = existing.map(|track| track.content.level).unwrap_or(0.75);
            let rhythm = RhythmContext::from_drums(
                self.session.current().drums().map(|track| &track.content.pattern),
            );
            // Compose before touching either project snapshot or audio output.
            let generated = synth.generate_synth1(preset, level, rhythm)?;
            generated.validate()?;
            if generated.preset != preset || generated.level != level {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Synth generator must preserve chosen preset and level",
                ));
            }
            let note_count = generated.notes.len();
            self.replace_synth1(generated, CreationMode::Ai, audio)?;
            return Ok(AppResponse::Synth1Generated { notes: note_count, preset });
        }
        self.dispatch(command, drums, audio)
    }

    /// Live three-track route. Slot identities fix the instrument and model;
    /// the legacy dispatcher remains available for older experiments.
    pub fn dispatch_neural<G: DrumGenerator, S: Synth1Generator, A: AudioPort>(
        &mut self, command: Command, drums: &mut G, synth: &mut S, audio: &mut A,
    ) -> io::Result<AppResponse> {
        let target = match command {
            Command::GenerateTrack(TrackId::Synth1) => Some((TrackId::Synth1, SynthPreset::Pluck)),
            Command::GenerateTrack(TrackId::Synth2) => Some((TrackId::Synth2, SynthPreset::Pad)),
            Command::EditUserTrack(TrackId::Synth2) => return Ok(AppResponse::EditSynth2),
            _ => None,
        };
        if let Some((id, preset)) = target {
            let existing = if id == TrackId::Synth1 { self.session.current().synth1() }
                else { self.session.current().synth2() };
            let level = existing.map(|t| t.content.level).unwrap_or(0.75);
            let rhythm = RhythmContext::from_drums(self.session.current().drums().map(|t| &t.content.pattern));
            let track = synth.generate_synth1(preset, level, rhythm)?;
            track.validate()?;
            if track.preset != preset || track.level != level || track.notes.is_empty() {
                return Err(invalid("Neural service returned an invalid instrument, level or empty phrase"));
            }
            let notes = track.notes.len();
            if id == TrackId::Synth1 {
                self.replace_synth1(track, CreationMode::Ai, audio)?;
                return Ok(AppResponse::Synth1Generated { notes, preset });
            }
            self.replace_synth2(track, CreationMode::Ai, audio)?;
            return Ok(AppResponse::Synth2Generated { notes });
        }
        self.dispatch(command, drums, audio)
    }

    pub fn set_synth2<A: AudioPort>(&mut self, synth: SynthTrack, audio: &mut A) -> io::Result<()> {
        self.replace_synth2(synth, CreationMode::User, audio)
    }

    fn replace_synth2<A: AudioPort>(
        &mut self, synth: SynthTrack, source: CreationMode, audio: &mut A,
    ) -> io::Result<()> {
        synth.validate()?;
        if synth.preset != SynthPreset::Pad { return Err(invalid("PAD slot requires the Pad instrument")); }
        let mut proposed = self.session.current().clone();
        proposed.replace_synth2(synth, source);
        let bar = self.prepare_project(&proposed)?;
        audio.queue_bar(bar)?;
        *self.session.current_mut() = proposed;
        self.session.save_current();
        Ok(())
    }

    /// The same atomic publish/save logic is shared by manual note editing and
    /// autonomous composition; failed queue leaves saved music untouched.
    fn replace_synth1<A: AudioPort>(
        &mut self, synth: SynthTrack, source: CreationMode, audio: &mut A,
    ) -> io::Result<()> {
        synth.validate()?;
        let mut proposed = self.session.current().clone();
        proposed.replace_synth1(synth, source);
        let bar = self.prepare_project(&proposed)?;
        audio.queue_bar(bar)?;
        *self.session.current_mut() = proposed;
        self.session.save_current();
        Ok(())
    }

    /// Update shared master inserts without replacing or regenerating tracks.
    /// Like BPM and kit edits, this is atomic from the project perspective:
    /// failed DSP/AudioPort publication retains previous saved settings.
    pub fn set_master_fx<A: AudioPort>(
        &mut self, effects: MasterEffects, audio: &mut A,
    ) -> io::Result<()> {
        let mut proposed = self.session.current().clone();
        proposed.set_master_fx(effects)?;
        if proposed.has_content() {
            let bar = self.prepare_project(&proposed)?;
            audio.queue_bar(bar)?;
        }
        *self.session.current_mut() = proposed;
        if self.session.current().has_content() {
            self.session.save_current();
        }
        Ok(())
    }

    /// Change common BPM without regenerating AI or touching the callback.
    /// Drums and Synth 1 are re-rendered together; saved state changes only
    /// after the entire mixed bar has been accepted by AudioPort.
    pub fn set_bpm<A: AudioPort>(&mut self, bpm: f32, audio: &mut A) -> io::Result<()> {
        let mut proposed = self.session.current().clone();
        proposed.set_bpm(bpm).map_err(invalid)?;
        if proposed.has_content() {
            let bar = self.prepare_project(&proposed)?;
            audio.queue_bar(bar)?;
        }
        *self.session.current_mut() = proposed;
        if self.session.current().has_content() {
            self.session.save_current();
        }
        Ok(())
    }

    /// Preserve the drum pattern and any Synth 1 notes while editing the kit.
    pub fn set_drum_kit<A: AudioPort>(
        &mut self,
        kit: DrumKit,
        audio: &mut A,
    ) -> io::Result<bool> {
        let Some(existing) = self.session.current().drums() else {
            return Ok(false);
        };
        let mut changed = existing.content.clone();
        changed.kit = kit;
        let mut proposed = self.session.current().clone();
        proposed.replace_drums(changed, CreationMode::User);
        let bar = self.prepare_project(&proposed)?;
        audio.queue_bar(bar)?;
        *self.session.current_mut() = proposed;
        self.session.save_current();
        Ok(true)
    }

    /// Create or edit the USER Synth 1 instrument and notes. The AI drum track
    /// is not regenerated or lost, and queue failures leave both snapshots
    /// unchanged. This is the only synth editor implemented at present.
    pub fn set_synth1<A: AudioPort>(
        &mut self,
        synth: SynthTrack,
        audio: &mut A,
    ) -> io::Result<()> {
        self.replace_synth1(synth, CreationMode::User, audio)
    }

}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
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
        let calls = Cell::new(0usize);
        let mut generator = || -> io::Result<GeneratedPattern> {
            calls.set(calls.get() + 1);
            Ok(pattern(38))
        };
        assert_eq!(app.dispatch(Command::GenerateTrack(TrackId::Drums), &mut generator, &mut audio).unwrap(),
            AppResponse::DrumGenerated { hits: 1 });
        assert_eq!(calls.get(), 1);
        assert_eq!(app.session().last_saved().unwrap().drums().unwrap().content.pattern.hits[0].pitch, 38);
        assert_eq!(app.dispatch(Command::PlayLast, &mut generator, &mut audio).unwrap(),
            AppResponse::PlayingLast { mode: CreationMode::Ai, bpm: 120.0 });
        assert_eq!(calls.get(), 1);
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
            Command::EditUserTrack(TrackId::Synth2),
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
        let mut failed = || -> io::Result<GeneratedPattern> {
            Err(io::Error::new(io::ErrorKind::Other, "generation failed"))
        };
        assert!(app.dispatch(Command::GenerateTrack(TrackId::Drums), &mut failed, &mut audio).is_err());
        audio.reject_queue = true;
        let mut succeeding = || -> io::Result<GeneratedPattern> { Ok(pattern(38)) };
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
    fn menu_navigation_reaches_real_drum_ai_and_saved_playback() {
        use crate::ui::{InputEvent, MenuController};

        let mut app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut audio = TestAudio::default();
        app.install_initial_drums(pattern(36), DrumKit::default(), &mut audio).unwrap();
        let calls = Cell::new(0usize);
        let mut generator = || -> io::Result<GeneratedPattern> {
            calls.set(calls.get() + 1);
            Ok(pattern(42))
        };

        let mut menu = MenuController::default();
        menu.handle(InputEvent::CenterPressed);   // PLAY settings
        menu.handle(InputEvent::Next);            // AI
        menu.handle(InputEvent::CenterPressed);   // AI menu
        menu.handle(InputEvent::Next);            // Generate Track
        menu.handle(InputEvent::CenterPressed);   // Choose track
        menu.handle(InputEvent::Next);            // Synth 2
        menu.handle(InputEvent::Next);            // Drums
        let command = menu.handle(InputEvent::CenterPressed).unwrap();
        assert_eq!(
            app.dispatch(command, &mut generator, &mut audio).unwrap(),
            AppResponse::DrumGenerated { hits: 1 },
        );
        assert_eq!(calls.get(), 1);
        assert_eq!(app.session().last_saved().unwrap().drums().unwrap().content.pattern.hits[0].pitch, 42);

        menu.handle(InputEvent::BackPressed);     // AI menu
        menu.handle(InputEvent::BackPressed);     // PLAY settings
        menu.handle(InputEvent::BackPressed);     // Home
        let play = menu.handle(InputEvent::PlayPressed).unwrap();
        assert_eq!(
            app.dispatch(play, &mut generator, &mut audio).unwrap(),
            AppResponse::PlayingLast { mode: CreationMode::Ai, bpm: 120.0 }
        );
        assert_eq!(calls.get(), 1); // PLAY is replay, not regeneration
        assert_eq!(audio.starts, 1);
    }

    #[test]
    fn user_synth1_presets_mix_with_drums_and_play_saved_snapshot() {
        use crate::synth1::{SynthPreset, SynthTrack};
        let mut app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut audio = TestAudio::default();
        app.install_initial_drums(pattern(36), DrumKit::default(), &mut audio).unwrap();
        let previous_drums = app.session().current().drums().unwrap().content.pattern.classes.clone();
        let drums_only = audio.queued.last().unwrap().samples.clone();

        assert_eq!(app.dispatch(
            Command::EditUserTrack(TrackId::Synth1),
            &mut || -> io::Result<GeneratedPattern> { panic!("no AI needed") },
            &mut audio,
        ).unwrap(), AppResponse::EditSynth1);
        let pluck = SynthTrack::demo(SynthPreset::Pluck);
        app.set_synth1(pluck.clone(), &mut audio).unwrap();
        assert_eq!(audio.queued.last().unwrap().samples.len(), 16_000);
        assert_ne!(audio.queued.last().unwrap().samples, drums_only);
        assert_eq!(app.session().current().drums().unwrap().content.pattern.classes, previous_drums);
        assert_eq!(app.session().last_saved().unwrap().synth1().unwrap().content, pluck);

        let pad = SynthTrack::demo(SynthPreset::Pad);
        app.set_synth1(pad.clone(), &mut audio).unwrap();
        let pad_bar = audio.queued.last().unwrap().samples.clone();
        assert_ne!(pad_bar, drums_only);
        app.dispatch(
            Command::PlayLast,
            &mut || -> io::Result<GeneratedPattern> { panic!("PLAY must not generate") },
            &mut audio,
        ).unwrap();
        assert_eq!(audio.queued.last().unwrap().samples, pad_bar);
        assert_eq!(app.session().last_saved().unwrap().synth1().unwrap().content, pad);
        assert_eq!(audio.starts, 1);
    }

    #[test]
    fn synth1_survives_drums_regeneration_bpm_and_kit_edits() {
        use crate::synth1::{SynthPreset, SynthTrack};
        let mut app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut audio = TestAudio::default();
        app.install_initial_drums(pattern(36), DrumKit::default(), &mut audio).unwrap();
        let synth = SynthTrack::demo(SynthPreset::Pad);
        app.set_synth1(synth.clone(), &mut audio).unwrap();
        app.set_bpm(90.0, &mut audio).unwrap();
        assert_eq!(audio.queued.last().unwrap().samples.len(), 21_333);
        let mut kit = DrumKit::default();
        kit.set(36, "tone", 0.7).unwrap();
        app.set_drum_kit(kit, &mut audio).unwrap();
        app.dispatch(
            Command::GenerateTrack(TrackId::Drums),
            &mut || Ok(pattern(38)),
            &mut audio,
        ).unwrap();
        assert_eq!(app.session().last_saved().unwrap().synth1().unwrap().content, synth);
        assert_eq!(app.session().last_saved().unwrap().drums().unwrap().content.pattern.hits[0].pitch, 38);
        assert_eq!(app.session().last_saved().unwrap().bpm(), 90.0);
    }

    #[test]
    fn master_fx_applies_to_drums_only_and_synth_only() {
        use crate::synth1::{SynthPreset, SynthTrack};
        // Drum-only audio is affected, even before a synth has been created.
        let mut drums_app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut drums_audio = TestAudio::default();
        drums_app.install_initial_drums(pattern(36), DrumKit::default(), &mut drums_audio).unwrap();
        drums_app.set_master_fx(MasterEffects::dry(), &mut drums_audio).unwrap();
        let drums_dry = drums_audio.queued.last().unwrap().samples.clone();
        drums_app.set_master_fx(MasterEffects::default(), &mut drums_audio).unwrap();
        assert_ne!(drums_audio.queued.last().unwrap().samples, drums_dry);
        assert!(drums_app.session().current().synth1().is_none());

        // Synth-only audio is also processed, even without DrumsRNN.
        let mut synth_app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut synth_audio = TestAudio::default();
        synth_app.set_synth1(SynthTrack::demo(SynthPreset::Pluck), &mut synth_audio).unwrap();
        synth_app.set_master_fx(MasterEffects::dry(), &mut synth_audio).unwrap();
        let synth_dry = synth_audio.queued.last().unwrap().samples.clone();
        synth_app.set_master_fx(MasterEffects::default(), &mut synth_audio).unwrap();
        assert_ne!(synth_audio.queued.last().unwrap().samples, synth_dry);
        assert!(synth_app.session().current().drums().is_none());
        assert_eq!(synth_app.session().last_saved().unwrap().master_fx(), MasterEffects::default());
    }

    #[test]
    fn master_fx_processes_whole_mix_and_is_saved_for_play_and_tempo_changes() {
        use crate::synth1::{SynthPreset, SynthTrack};
        let mut app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut audio = TestAudio::default();
        app.install_initial_drums(pattern(36), DrumKit::default(), &mut audio).unwrap();
        app.set_synth1(SynthTrack::demo(SynthPreset::Pluck), &mut audio).unwrap();

        let mut dry = app.session().current().master_fx();
        dry.delay.enabled = false;
        dry.reverb.enabled = false;
        app.set_master_fx(dry, &mut audio).unwrap();
        let dry_bar = audio.queued.last().unwrap().samples.clone();

        let mut wet = MasterEffects::default();
        wet.delay.steps = 4;
        wet.delay.wet = 0.4;
        wet.reverb.wet = 0.32;
        app.set_master_fx(wet, &mut audio).unwrap();
        let mixed_wet = audio.queued.last().unwrap().samples.clone();
        assert_ne!(mixed_wet, dry_bar);
        assert_eq!(app.session().last_saved().unwrap().master_fx(), wet);
        assert!(app.session().last_saved().unwrap().synth1().is_some());
        assert!(app.session().last_saved().unwrap().drums().is_some());
        assert!(mixed_wet.iter().all(|x| x.is_finite() && x.abs() <= 0.851));

        app.set_bpm(80.0, &mut audio).unwrap();
        assert_eq!(audio.queued.last().unwrap().samples.len(), 24_000);
        assert_eq!(app.session().last_saved().unwrap().master_fx(), wet);
        let last = audio.queued.last().unwrap().samples.clone();
        app.dispatch(
            Command::PlayLast,
            &mut || -> io::Result<GeneratedPattern> { panic!("PLAY must not regenerate") },
            &mut audio,
        ).unwrap();
        assert_eq!(audio.queued.last().unwrap().samples, last);
    }

    #[test]
    fn invalid_fx_and_rejected_audio_never_change_saved_project() {
        let mut app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut audio = TestAudio::default();
        app.install_initial_drums(pattern(36), DrumKit::default(), &mut audio).unwrap();
        let settings = app.session().last_saved().unwrap().master_fx();

        let mut bad = settings;
        bad.delay.feedback = f32::NAN;
        assert!(app.set_master_fx(bad, &mut audio).is_err());
        assert_eq!(app.session().current().master_fx(), settings);

        audio.reject_queue = true;
        let mut valid = settings;
        valid.reverb.wet = 0.4;
        assert!(app.set_master_fx(valid, &mut audio).is_err());
        assert_eq!(app.session().current().master_fx(), settings);
        assert_eq!(app.session().last_saved().unwrap().master_fx(), settings);
    }

    #[test]
    fn ai_synth1_generates_new_phrase_and_preserves_drums_preset_and_level() {
        use crate::melody::compose;
        use crate::synth1::{SynthPreset, SynthTrack};
        let mut app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut audio = TestAudio::default();
        app.install_initial_drums(pattern(36), DrumKit::default(), &mut audio).unwrap();
        let mut patch = SynthTrack::demo(SynthPreset::Pad);
        patch.level = 0.5;
        app.set_synth1(patch.clone(), &mut audio).unwrap();
        let drum_before = app.session().current().drums().unwrap().content.pattern.classes.clone();
        let calls = Cell::new(0usize);
        let mut melody = |preset: SynthPreset, level: f32, rhythm: RhythmContext| {
            assert_eq!(preset, SynthPreset::Pad);
            assert_eq!(level, 0.5);
            assert!(rhythm.kicks[0]);
            calls.set(calls.get() + 1);
            compose(42, preset, level, rhythm)
        };
        let mut no_drums = || -> io::Result<GeneratedPattern> { panic!("no DrumsRNN call") };
        let response = app.dispatch_with_synth(
            Command::GenerateTrack(TrackId::Synth1),
            &mut no_drums, &mut melody, &mut audio,
        ).unwrap();
        assert_eq!(response, AppResponse::Synth1Generated { notes: 8, preset: SynthPreset::Pad });
        assert_eq!(calls.get(), 1);
        let saved = app.session().last_saved().unwrap();
        assert_eq!(saved.mode(), CreationMode::Ai);
        assert_eq!(saved.synth1().unwrap().source, CreationMode::Ai);
        assert_eq!(saved.synth1().unwrap().content.level, 0.5);
        assert_eq!(saved.synth1().unwrap().content.preset, SynthPreset::Pad);
        assert_ne!(saved.synth1().unwrap().content.notes, patch.notes);
        assert_eq!(saved.drums().unwrap().content.pattern.classes, drum_before);
        assert_eq!(saved.drums().unwrap().source, CreationMode::Ai);
        assert_eq!(audio.starts, 0); // generating doesn't auto-PLAY

        let count = audio.queued.len();
        let ignored = || -> io::Result<GeneratedPattern> { panic!("PLAY should not generate") };
        let ignored_synth = |_: SynthPreset, _: f32, _: RhythmContext| -> io::Result<SynthTrack> {
            panic!("PLAY must use saved notes")
        };
        let mut ignored = ignored;
        let mut ignored_synth = ignored_synth;
        assert_eq!(app.dispatch_with_synth(
            Command::PlayLast, &mut ignored, &mut ignored_synth, &mut audio,
        ).unwrap(), AppResponse::PlayingLast { mode: CreationMode::Ai, bpm: 120.0 });
        assert_eq!(audio.queued.len(), count + 1);
        assert_eq!(audio.starts, 1);
    }

    #[test]
    fn ai_synth1_failed_generator_or_audio_keeps_last_saved_project() {
        use crate::melody::compose;
        use crate::synth1::{SynthPreset, SynthTrack};
        let mut app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut audio = TestAudio::default();
        app.install_initial_drums(pattern(36), DrumKit::default(), &mut audio).unwrap();
        app.set_synth1(SynthTrack::demo(SynthPreset::Pluck), &mut audio).unwrap();
        let original = app.session().last_saved().unwrap().synth1().unwrap().content.clone();
        let mut no_drums = || -> io::Result<GeneratedPattern> { panic!("not drums") };
        let mut failure = |_: SynthPreset, _: f32, _: RhythmContext| -> io::Result<SynthTrack> {
            Err(io::Error::other("melody generator failed"))
        };
        assert!(app.dispatch_with_synth(
            Command::GenerateTrack(TrackId::Synth1), &mut no_drums, &mut failure, &mut audio,
        ).is_err());
        assert_eq!(app.session().last_saved().unwrap().synth1().unwrap().content, original);
        audio.reject_queue = true;
        let mut success = |preset, level, groove| compose(99, preset, level, groove);
        assert!(app.dispatch_with_synth(
            Command::GenerateTrack(TrackId::Synth1), &mut no_drums, &mut success, &mut audio,
        ).is_err());
        assert_eq!(app.session().last_saved().unwrap().synth1().unwrap().content, original);
        assert_eq!(app.session().last_saved().unwrap().synth1().unwrap().source, CreationMode::User);
    }

    #[test]
    fn ai_synth1_menu_route_generates_without_user_notes_or_drum_input() {
        use crate::melody::compose;
        use crate::ui::{InputEvent, MenuController};
        let mut app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut audio = TestAudio::default();
        let mut menu = MenuController::default();
        menu.handle(InputEvent::CenterPressed); // PLAY settings
        menu.handle(InputEvent::Next); // AI
        menu.handle(InputEvent::CenterPressed); // AI menu
        menu.handle(InputEvent::Next); // Generate Track
        menu.handle(InputEvent::CenterPressed); // Track picker, focuses Synth 1
        let command = menu.handle(InputEvent::CenterPressed).unwrap();
        assert_eq!(command, Command::GenerateTrack(TrackId::Synth1));
        let mut no_drums = || -> io::Result<GeneratedPattern> { panic!("no drums needed") };
        let mut melody = |preset, level, rhythm| compose(12, preset, level, rhythm);
        assert_eq!(app.dispatch_with_synth(
            command, &mut no_drums, &mut melody, &mut audio,
        ).unwrap(), AppResponse::Synth1Generated { notes: 8, preset: SynthPreset::Pluck });
        assert!(app.session().last_saved().unwrap().drums().is_none());
        assert_eq!(app.session().last_saved().unwrap().synth1().unwrap().source, CreationMode::Ai);
        assert_eq!(audio.queued.len(), 1);
    }

    #[test]
    fn synth1_can_play_alone_without_drum_pattern_or_ai() {
        use crate::synth1::{SynthPreset, SynthTrack};
        let mut app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut audio = TestAudio::default();
        let mut muted = SynthTrack::demo(SynthPreset::Pluck);
        muted.level = 0.85;
        app.set_synth1(muted.clone(), &mut audio).unwrap();
        assert!(app.session().current().drums().is_none());
        assert_eq!(app.session().last_saved().unwrap().synth1().unwrap().content, muted);
        assert_eq!(audio.queued.len(), 1);
        assert!(audio.queued[0].samples.iter().any(|x| x.abs() > 0.01));
        let mut generator = || -> io::Result<GeneratedPattern> {
            panic!("Synth-only PLAY must never run DrumsRNN")
        };
        assert_eq!(
            app.dispatch(Command::PlayLast, &mut generator, &mut audio).unwrap(),
            AppResponse::PlayingLast { mode: CreationMode::User, bpm: 120.0 },
        );
        assert_eq!(audio.starts, 1);
        assert_eq!(audio.queued[0].samples, audio.queued[1].samples);
    }

    #[test]
    fn synth1_invalid_settings_and_closed_audio_do_not_change_project() {
        use crate::synth1::{SynthPreset, SynthTrack};
        let mut app = ApplicationController::new(8_000, 120.0).unwrap();
        let mut audio = TestAudio::default();
        app.install_initial_drums(pattern(36), DrumKit::default(), &mut audio).unwrap();
        let old_queued = audio.queued.len();
        let mut broken = SynthTrack::demo(SynthPreset::Pluck);
        broken.level = f32::NAN;
        assert!(app.set_synth1(broken, &mut audio).is_err());
        assert_eq!(audio.queued.len(), old_queued);
        assert!(app.session().current().synth1().is_none());

        audio.reject_queue = true;
        assert!(app.set_synth1(SynthTrack::demo(SynthPreset::Pluck), &mut audio).is_err());
        assert!(app.session().last_saved().unwrap().synth1().is_none());
        assert!(app.session().current().synth1().is_none());
    }

    #[test]
    fn neural_pad_is_independent_and_failed_publication_preserves_both_snapshots() {
        let mut app = ApplicationController::new(8_000,120.0).unwrap();
        let mut audio = TestAudio::default();
        let pluck = SynthTrack::demo(SynthPreset::Pluck);
        app.set_synth1(pluck.clone(),&mut audio).unwrap();
        let mut drums = || -> io::Result<GeneratedPattern> { panic!("must not regenerate drums") };
        let mut synth = |preset, level, _| {
            let mut track=SynthTrack::demo(preset); track.level=level; Ok(track)
        };
        let response=app.dispatch_neural(Command::GenerateTrack(TrackId::Synth2), &mut drums,&mut synth,&mut audio).unwrap();
        assert_eq!(response,AppResponse::Synth2Generated {notes:6});
        let pad=app.session().last_saved().unwrap().synth2().unwrap().content.clone();
        assert_eq!(app.session().last_saved().unwrap().synth1().unwrap().content,pluck);
        let mix=audio.queued.last().unwrap().samples.clone();
        app.dispatch_neural(Command::PlayLast,&mut drums,&mut synth,&mut audio).unwrap();
        assert_eq!(audio.queued.last().unwrap().samples,mix);
        audio.reject_queue=true;
        let mut changed=pad.clone(); changed.notes[0].pitch=72;
        let mut success=|_,_,_| Ok(changed.clone());
        assert!(app.dispatch_neural(Command::GenerateTrack(TrackId::Synth2),&mut drums,&mut success,&mut audio).is_err());
        assert_eq!(app.session().current().synth2().unwrap().content,pad);
        assert_eq!(app.session().last_saved().unwrap().synth2().unwrap().content,pad);
        assert_eq!(app.session().last_saved().unwrap().synth1().unwrap().content,pluck);
        let mut fail=|_,_,_| Err(io::Error::other("model missing"));
        assert!(app.dispatch_neural(Command::GenerateTrack(TrackId::Synth2),&mut drums,&mut fail,&mut audio).is_err());
        assert_eq!(app.session().last_saved().unwrap().synth2().unwrap().content,pad);
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
