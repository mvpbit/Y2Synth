//! Pure event-driven menu controller for the Y2Synth PLAY-first design.
//!
//! Hardware drivers translate presses/rotations into InputEvent.
//! This controller does no I/O, AI inference, DSP or audio callback work.
//! Its Command values are *requests* to a future application coordinator:
//! emitting GenerateAll does NOT claim the synth AI generators exist.

use crate::project::TrackId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputEvent {
    /// The wheel's PLAY action, distinct from selecting menu entries.
    PlayPressed,
    /// Center press: enter settings or select the focused entry.
    CenterPressed,
    BackPressed,
    Next,
    Previous,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Screen {
    Home,
    PlaySettings,
    UserPage(TrackId),
    AiMenu,
    AiTrackSelect,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModeChoice {
    User,
    Ai,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AiChoice {
    GenerateAll,
    GenerateTrack,
}

/// Commands describe user intent, not successful asynchronous execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Command {
    PlayLast,
    EditUserTrack(TrackId),
    GenerateAll,
    GenerateTrack(TrackId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MenuController {
    screen: Screen,
    mode_focus: ModeChoice,
    ai_focus: AiChoice,
    track_focus: TrackId,
}

impl Default for MenuController {
    fn default() -> Self {
        Self {
            screen: Screen::Home,
            mode_focus: ModeChoice::User,
            ai_focus: AiChoice::GenerateAll,
            track_focus: TrackId::Synth1,
        }
    }
}

impl MenuController {
    pub fn screen(&self) -> Screen {
        self.screen
    }

    pub fn mode_focus(&self) -> ModeChoice {
        self.mode_focus
    }

    pub fn ai_focus(&self) -> AiChoice {
        self.ai_focus
    }

    pub fn track_focus(&self) -> TrackId {
        self.track_focus
    }

    /// Navigation mutates only UI state. An optional Command is dispatched
    /// by the control thread (after checking project/generator availability).
    pub fn handle(&mut self, event: InputEvent) -> Option<Command> {
        match event {
            InputEvent::PlayPressed => {
                if self.screen == Screen::Home {
                    Some(Command::PlayLast)
                } else {
                    None
                }
            }
            InputEvent::CenterPressed => self.activate(),
            InputEvent::BackPressed => {
                self.back();
                None
            }
            InputEvent::Next => {
                self.move_focus(true);
                None
            }
            InputEvent::Previous => {
                self.move_focus(false);
                None
            }
        }
    }

    fn activate(&mut self) -> Option<Command> {
        match self.screen {
            Screen::Home => {
                self.screen = Screen::PlaySettings;
                None
            }
            Screen::PlaySettings => {
                self.screen = match self.mode_focus {
                    ModeChoice::User => Screen::UserPage(TrackId::Synth1),
                    ModeChoice::Ai => Screen::AiMenu,
                };
                None
            }
            Screen::UserPage(track) => Some(Command::EditUserTrack(track)),
            Screen::AiMenu => match self.ai_focus {
                AiChoice::GenerateAll => Some(Command::GenerateAll),
                AiChoice::GenerateTrack => {
                    self.screen = Screen::AiTrackSelect;
                    None
                }
            },
            Screen::AiTrackSelect => Some(Command::GenerateTrack(self.track_focus)),
        }
    }

    fn back(&mut self) {
        self.screen = match self.screen {
            Screen::Home => Screen::Home,
            Screen::PlaySettings => Screen::Home,
            Screen::UserPage(_) => {
                self.mode_focus = ModeChoice::User;
                Screen::PlaySettings
            }
            Screen::AiMenu => {
                self.mode_focus = ModeChoice::Ai;
                Screen::PlaySettings
            }
            Screen::AiTrackSelect => {
                self.ai_focus = AiChoice::GenerateTrack;
                Screen::AiMenu
            }
        };
    }

    fn move_focus(&mut self, forward: bool) {
        match self.screen {
            Screen::Home => {}
            Screen::PlaySettings => {
                self.mode_focus = match self.mode_focus {
                    ModeChoice::User => ModeChoice::Ai,
                    ModeChoice::Ai => ModeChoice::User,
                };
            }
            Screen::UserPage(track) => {
                self.screen = Screen::UserPage(if forward { track.next() } else { track.previous() });
            }
            Screen::AiMenu => {
                self.ai_focus = match self.ai_focus {
                    AiChoice::GenerateAll => AiChoice::GenerateTrack,
                    AiChoice::GenerateTrack => AiChoice::GenerateAll,
                };
            }
            Screen::AiTrackSelect => {
                self.track_focus = if forward {
                    self.track_focus.next()
                } else {
                    self.track_focus.previous()
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enter_ai(menu: &mut MenuController) {
        assert_eq!(menu.handle(InputEvent::CenterPressed), None);
        assert_eq!(menu.screen(), Screen::PlaySettings);
        menu.handle(InputEvent::Next);
        assert_eq!(menu.mode_focus(), ModeChoice::Ai);
        menu.handle(InputEvent::CenterPressed);
        assert_eq!(menu.screen(), Screen::AiMenu);
    }

    #[test]
    fn top_level_play_has_no_generation_side_effects() {
        let mut menu = MenuController::default();
        assert_eq!(menu.handle(InputEvent::PlayPressed), Some(Command::PlayLast));
        assert_eq!(menu.screen(), Screen::Home);
        menu.handle(InputEvent::BackPressed);
        assert_eq!(menu.screen(), Screen::Home);
    }

    #[test]
    fn three_user_pages_are_navigable_and_edit_intents_are_explicit() {
        let mut menu = MenuController::default();
        menu.handle(InputEvent::CenterPressed);
        menu.handle(InputEvent::CenterPressed);
        assert_eq!(menu.screen(), Screen::UserPage(TrackId::Synth1));
        menu.handle(InputEvent::Next);
        assert_eq!(menu.screen(), Screen::UserPage(TrackId::Synth2));
        menu.handle(InputEvent::Next);
        assert_eq!(menu.screen(), Screen::UserPage(TrackId::Drums));
        assert_eq!(
            menu.handle(InputEvent::CenterPressed),
            Some(Command::EditUserTrack(TrackId::Drums))
        );
        menu.handle(InputEvent::Previous);
        assert_eq!(menu.screen(), Screen::UserPage(TrackId::Synth2));
        menu.handle(InputEvent::BackPressed);
        assert_eq!(menu.screen(), Screen::PlaySettings);
        menu.handle(InputEvent::BackPressed);
        assert_eq!(menu.screen(), Screen::Home);
    }

    #[test]
    fn ai_generate_all_and_single_track_are_distinct_requests() {
        let mut menu = MenuController::default();
        enter_ai(&mut menu);
        assert_eq!(menu.handle(InputEvent::CenterPressed), Some(Command::GenerateAll));
        assert_eq!(menu.screen(), Screen::AiMenu);
        menu.handle(InputEvent::Next);
        assert_eq!(menu.ai_focus(), AiChoice::GenerateTrack);
        menu.handle(InputEvent::CenterPressed);
        assert_eq!(menu.screen(), Screen::AiTrackSelect);
        assert_eq!(
            menu.handle(InputEvent::CenterPressed),
            Some(Command::GenerateTrack(TrackId::Synth1))
        );
        menu.handle(InputEvent::Next);
        assert_eq!(
            menu.handle(InputEvent::CenterPressed),
            Some(Command::GenerateTrack(TrackId::Synth2))
        );
        menu.handle(InputEvent::Next);
        assert_eq!(
            menu.handle(InputEvent::CenterPressed),
            Some(Command::GenerateTrack(TrackId::Drums))
        );
        // Generating a single track does not itself navigate away.
        assert_eq!(menu.screen(), Screen::AiTrackSelect);
    }

    #[test]
    fn back_restores_parent_selection_without_triggering_commands() {
        let mut menu = MenuController::default();
        enter_ai(&mut menu);
        menu.handle(InputEvent::Next);
        menu.handle(InputEvent::CenterPressed);
        assert_eq!(menu.screen(), Screen::AiTrackSelect);
        assert_eq!(menu.handle(InputEvent::BackPressed), None);
        assert_eq!(menu.screen(), Screen::AiMenu);
        assert_eq!(menu.ai_focus(), AiChoice::GenerateTrack);
        menu.handle(InputEvent::BackPressed);
        assert_eq!(menu.screen(), Screen::PlaySettings);
        assert_eq!(menu.mode_focus(), ModeChoice::Ai);
        menu.handle(InputEvent::BackPressed);
        assert_eq!(menu.screen(), Screen::Home);
    }

    #[test]
    fn play_does_not_start_while_browsing_settings() {
        let mut menu = MenuController::default();
        menu.handle(InputEvent::CenterPressed);
        assert_eq!(menu.handle(InputEvent::PlayPressed), None);
        assert_eq!(menu.screen(), Screen::PlaySettings);
    }
}
