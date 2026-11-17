//! Engine-independent musical state for the future 3-track Y2Synth.
//!
//! This module stores the last *saved* musical snapshot, independent of the UI
//! and audio backend. It deliberately does not implement filesystem persistence,
//! a synth engine, or AI for synth tracks.

use crate::master_fx::MasterEffects;
use std::io;

/// Which track to edit or regenerate. The UI knows this enum, not the DSP.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackId {
    Synth1,
    Synth2,
    Drums,
}

impl TrackId {
    pub const ALL: [Self; 3] = [Self::Synth1, Self::Synth2, Self::Drums];

    pub fn next(self) -> Self {
        match self {
            Self::Synth1 => Self::Synth2,
            Self::Synth2 => Self::Drums,
            Self::Drums => Self::Synth1,
        }
    }

    pub fn previous(self) -> Self {
        match self {
            Self::Synth1 => Self::Drums,
            Self::Synth2 => Self::Synth1,
            Self::Drums => Self::Synth2,
        }
    }
}

/// The mode of the latest applied editing/generation operation.
/// The source of *each* track is tracked independently for mixed projects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CreationMode {
    User,
    Ai,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Track<T> {
    pub source: CreationMode,
    pub content: T,
}

impl<T> Track<T> {
    pub fn new(content: T, source: CreationMode) -> Self {
        Self { source, content }
    }
}

/// Each track has its own content type; future synth implementations need not
/// pretend to use the drum pattern format.
#[derive(Clone, Debug, PartialEq)]
pub struct MusicProject<Synth1, Synth2, Drums> {
    mode: CreationMode,
    bpm: f32,
    master_fx: MasterEffects,
    synth1: Option<Track<Synth1>>,
    synth2: Option<Track<Synth2>>,
    drums: Option<Track<Drums>>,
}

/// Commit an AI-generated three-track result only after all generators succeed.
/// Building this value alone cannot change the currently playing project.
pub struct GeneratedAll<Synth1, Synth2, Drums> {
    pub synth1: Synth1,
    pub synth2: Synth2,
    pub drums: Drums,
}

impl<Synth1, Synth2, Drums> Default for MusicProject<Synth1, Synth2, Drums> {
    fn default() -> Self {
        Self {
            mode: CreationMode::User,
            bpm: 120.0,
            master_fx: MasterEffects::default(),
            synth1: None,
            synth2: None,
            drums: None,
        }
    }
}

impl<Synth1, Synth2, Drums> MusicProject<Synth1, Synth2, Drums> {
    pub fn mode(&self) -> CreationMode {
        self.mode
    }

    pub fn bpm(&self) -> f32 {
        self.bpm
    }

    /// Project-wide insert chain: all mixed voices -> delay -> reverb.
    /// Independent of the source (USER/AI) of any individual track.
    pub fn master_fx(&self) -> MasterEffects {
        self.master_fx
    }

    pub fn set_master_fx(&mut self, effects: MasterEffects) -> io::Result<()> {
        effects.validate()?;
        self.master_fx = effects;
        Ok(())
    }

    /// Shared tempo of all three tracks. A saved project includes this value.
    pub fn set_bpm(&mut self, bpm: f32) -> Result<(), &'static str> {
        if !bpm.is_finite() || !(40.0..=240.0).contains(&bpm) {
            return Err("BPM must be finite and in 40..=240");
        }
        self.bpm = bpm;
        Ok(())
    }

    pub fn synth1(&self) -> Option<&Track<Synth1>> {
        self.synth1.as_ref()
    }

    pub fn synth2(&self) -> Option<&Track<Synth2>> {
        self.synth2.as_ref()
    }

    pub fn drums(&self) -> Option<&Track<Drums>> {
        self.drums.as_ref()
    }

    pub fn has_content(&self) -> bool {
        self.synth1.is_some() || self.synth2.is_some() || self.drums.is_some()
    }

    pub fn replace_synth1(&mut self, content: Synth1, source: CreationMode) {
        self.synth1 = Some(Track::new(content, source));
        self.mode = source;
    }

    pub fn replace_synth2(&mut self, content: Synth2, source: CreationMode) {
        self.synth2 = Some(Track::new(content, source));
        self.mode = source;
    }

    pub fn replace_drums(&mut self, content: Drums, source: CreationMode) {
        self.drums = Some(Track::new(content, source));
        self.mode = source;
    }

    /// Atomic from the project perspective: all three track slots switch in
    /// one operation. AI generation is performed elsewhere, before this call.
    pub fn apply_generated_all(&mut self, generated: GeneratedAll<Synth1, Synth2, Drums>) {
        self.synth1 = Some(Track::new(generated.synth1, CreationMode::Ai));
        self.synth2 = Some(Track::new(generated.synth2, CreationMode::Ai));
        self.drums = Some(Track::new(generated.drums, CreationMode::Ai));
        self.mode = CreationMode::Ai;
    }
}

/// In-memory last-saved snapshot only. Durable storage/restore on Y2 is a
/// separate future adapter. PLAY should use `last_saved()`, not regenerate.
pub struct MusicSession<Synth1, Synth2, Drums> {
    current: MusicProject<Synth1, Synth2, Drums>,
    last_saved: Option<MusicProject<Synth1, Synth2, Drums>>,
}

impl<Synth1, Synth2, Drums> Default for MusicSession<Synth1, Synth2, Drums> {
    fn default() -> Self {
        Self {
            current: MusicProject::default(),
            last_saved: None,
        }
    }
}

impl<Synth1, Synth2, Drums> MusicSession<Synth1, Synth2, Drums> {
    pub fn current(&self) -> &MusicProject<Synth1, Synth2, Drums> {
        &self.current
    }

    pub fn current_mut(&mut self) -> &mut MusicProject<Synth1, Synth2, Drums> {
        &mut self.current
    }

    pub fn last_saved(&self) -> Option<&MusicProject<Synth1, Synth2, Drums>> {
        self.last_saved.as_ref()
    }
}

impl<Synth1: Clone, Synth2: Clone, Drums: Clone> MusicSession<Synth1, Synth2, Drums> {
    pub fn save_current(&mut self) {
        self.last_saved = Some(self.current.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestProject = MusicProject<String, String, Vec<u8>>;

    #[test]
    fn no_empty_project_claims_to_have_music() {
        let project = TestProject::default();
        assert!(!project.has_content());
        assert_eq!(project.mode(), CreationMode::User);
        assert_eq!(project.bpm(), 120.0);
    }

    #[test]
    fn generate_one_track_preserves_both_other_tracks() {
        let mut project = TestProject::default();
        project.replace_synth1("lead".to_string(), CreationMode::User);
        project.replace_synth2("bass".to_string(), CreationMode::User);
        project.replace_drums(vec![36, 42], CreationMode::User);

        project.replace_drums(vec![38, 46], CreationMode::Ai);
        assert_eq!(project.mode(), CreationMode::Ai);
        assert_eq!(project.synth1().unwrap().content, "lead");
        assert_eq!(project.synth2().unwrap().content, "bass");
        assert_eq!(project.drums().unwrap().content, vec![38, 46]);
        assert_eq!(project.synth1().unwrap().source, CreationMode::User);
        assert_eq!(project.drums().unwrap().source, CreationMode::Ai);
    }

    #[test]
    fn generate_all_replaces_every_track_as_one_commit() {
        let mut project = TestProject::default();
        project.replace_drums(vec![36], CreationMode::User);
        project.apply_generated_all(GeneratedAll {
            synth1: "ai lead".to_string(),
            synth2: "ai bass".to_string(),
            drums: vec![42, 38],
        });
        assert_eq!(project.mode(), CreationMode::Ai);
        assert_eq!(project.synth1().unwrap().content, "ai lead");
        assert_eq!(project.synth2().unwrap().content, "ai bass");
        assert_eq!(project.drums().unwrap().content, vec![42, 38]);
        assert_eq!(project.synth1().unwrap().source, CreationMode::Ai);
        assert_eq!(project.synth2().unwrap().source, CreationMode::Ai);
        assert_eq!(project.drums().unwrap().source, CreationMode::Ai);
    }

    #[test]
    fn saved_snapshot_remains_independent_of_future_changes() {
        let mut session: MusicSession<String, String, Vec<u8>> = MusicSession::default();
        assert!(session.last_saved().is_none());
        session.current_mut().replace_drums(vec![36], CreationMode::User);
        session.save_current();
        session.current_mut().replace_drums(vec![38], CreationMode::Ai);
        assert_eq!(session.last_saved().unwrap().mode(), CreationMode::User);
        assert_eq!(session.last_saved().unwrap().drums().unwrap().content, vec![36]);
        assert_eq!(session.current().drums().unwrap().content, vec![38]);
    }

    #[test]
    fn tempo_is_saved_with_music_and_invalid_values_do_not_mutate_it() {
        let mut session: MusicSession<(), (), Vec<u8>> = MusicSession::default();
        session.current_mut().replace_drums(vec![36], CreationMode::User);
        session.current_mut().set_bpm(92.5).unwrap();
        session.save_current();
        session.current_mut().set_bpm(160.0).unwrap();
        assert_eq!(session.last_saved().unwrap().bpm(), 92.5);
        assert_eq!(session.current().bpm(), 160.0);
        for bpm in [0.0, 39.0, 241.0, f32::INFINITY, f32::NAN] {
            assert!(session.current_mut().set_bpm(bpm).is_err());
        }
        assert_eq!(session.current().bpm(), 160.0);
    }

    #[test]
    fn master_effects_are_project_wide_saved_and_validated() {
        let mut session: MusicSession<String, (), Vec<u8>> = MusicSession::default();
        assert!(session.current().master_fx().delay.enabled);
        assert!(session.current().master_fx().reverb.enabled);
        session.current_mut().replace_synth1("lead".into(), CreationMode::User);
        session.current_mut().replace_drums(vec![36], CreationMode::Ai);

        let mut effects = session.current().master_fx();
        effects.delay.steps = 4;
        effects.delay.wet = 0.4;
        effects.reverb.size = 0.9;
        session.current_mut().set_master_fx(effects).unwrap();
        session.save_current();

        let mut changed = effects;
        changed.delay.wet = 0.0;
        session.current_mut().set_master_fx(changed).unwrap();
        assert_eq!(session.last_saved().unwrap().master_fx(), effects);
        assert_eq!(session.current().synth1().unwrap().source, CreationMode::User);
        assert_eq!(session.current().drums().unwrap().source, CreationMode::Ai);

        changed.reverb.size = f32::NAN;
        assert!(session.current_mut().set_master_fx(changed).is_err());
        assert_eq!(session.current().master_fx().delay.wet, 0.0);
    }

    #[test]
    fn track_navigation_cycles_in_display_order() {
        assert_eq!(TrackId::Synth1.next(), TrackId::Synth2);
        assert_eq!(TrackId::Synth2.next(), TrackId::Drums);
        assert_eq!(TrackId::Drums.next(), TrackId::Synth1);
        assert_eq!(TrackId::Synth1.previous(), TrackId::Drums);
    }
}
