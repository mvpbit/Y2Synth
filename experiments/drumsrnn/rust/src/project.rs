//! Engine-independent musical state for the future 3-track Y2Synth.
//!
//! This module stores the last *saved* musical snapshot, independent of the UI
//! and audio backend. It deliberately does not implement filesystem persistence,
//! a synth engine, or AI for synth tracks.

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
    fn track_navigation_cycles_in_display_order() {
        assert_eq!(TrackId::Synth1.next(), TrackId::Synth2);
        assert_eq!(TrackId::Synth2.next(), TrackId::Drums);
        assert_eq!(TrackId::Drums.next(), TrackId::Synth1);
        assert_eq!(TrackId::Synth1.previous(), TrackId::Drums);
    }
}
