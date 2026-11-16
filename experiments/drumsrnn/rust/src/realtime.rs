//! Lock-free handoff between the CLI/control thread and the audio callback.
//!
//! `triple_buffer` owns THREE PreparedLoop values and swaps their roles with
//! atomics. The producer may replace/destroy old Vec buffers, but only on its
//! control thread. The CPAL callback calls only Output::update(), reads the
//! currently published Vec by reference, and never owns or frees a Vec.
//!
//! Play/stop is a separate relaxed atomic flag. The latest published groove
//! wins; intermediate grooves may be discarded on the producer thread.
//! Changes are adopted at the start of a bar (or while stopped).
//!
//! There are no mutexes, channel receive calls or heap allocations in tick().
//! A bounded, preallocated 15ms crossfade scratch buffer preserves transitions.

use crate::sequencer::PreparedLoop;
use std::io::{self, ErrorKind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use triple_buffer::Output;

const FADE_SECONDS: f64 = 0.015;
const MASTER_LEVEL: f32 = 0.75;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Transport {
    Stopped,
    Playing,
    Stopping,
}

/// Single-consumer real-time player. The control thread only holds an
/// Input<PreparedLoop> and the Arc<AtomicBool>, never accesses this struct.
pub struct RealtimePlayer {
    output: Output<PreparedLoop>,
    requested_play: Arc<AtomicBool>,
    sample_rate: u32,
    position: usize,
    transport: Transport,
    gain: f32,
    fade_frames: usize,
    previous_head: Vec<f32>,
    crossfade_length: usize,
    crossfade_position: usize,
}

impl RealtimePlayer {
    pub fn new(
        output: Output<PreparedLoop>,
        requested_play: Arc<AtomicBool>,
        sample_rate: u32,
    ) -> io::Result<Self> {
        if !(8_000..=192_000).contains(&sample_rate) {
            return Err(io::Error::new(
                ErrorKind::InvalidInput,
                "Output sample rate must be 8000..=192000",
            ));
        }
        let fade_frames = ((sample_rate as f64 * FADE_SECONDS).round() as usize).max(1);
        Ok(Self {
            output,
            requested_play,
            sample_rate,
            position: 0,
            transport: Transport::Stopped,
            gain: 0.0,
            fade_frames,
            previous_head: vec![0.0; fade_frames],
            crossfade_length: 0,
            crossfade_position: 0,
        })
    }

    fn update_at_boundary(&mut self) {
        let old = self.output.output_buffer();
        let old_length = old.samples.len().min(self.fade_frames);
        if self.transport == Transport::Playing && self.gain > 0.0 {
            // Bounded copy; the buffer is allocated during initialization.
            self.previous_head[..old_length].copy_from_slice(&old.samples[..old_length]);
        }
        let changed = self.output.update();
        if changed && self.transport == Transport::Playing && self.gain > 0.0 {
            self.crossfade_length = old_length;
            self.crossfade_position = 0;
        } else {
            self.crossfade_length = 0;
        }
    }

    /// Render one mono sample. Intended to be invoked directly by CPAL's
    /// callback for every output frame.
    pub fn tick(&mut self) -> f32 {
        let requested = self.requested_play.load(Ordering::Relaxed);
        match (requested, self.transport) {
            (true, Transport::Stopped) => {
                self.transport = Transport::Playing;
                self.position = 0;
                self.gain = 0.0;
                self.crossfade_length = 0;
            }
            (true, Transport::Stopping) => {
                self.transport = Transport::Playing;
            }
            (false, Transport::Playing) => {
                self.transport = Transport::Stopping;
            }
            _ => {}
        }

        if self.transport == Transport::Stopped || self.position == 0 {
            // When stopped, latest published loop becomes current immediately.
            // While playing, publication only takes effect at the bar boundary.
            self.update_at_boundary();
        }
        if self.transport == Transport::Stopped {
            return 0.0;
        }

        let current = self.output.output_buffer();
        if current.sample_rate != self.sample_rate || current.samples.is_empty() {
            // A missing or mismatched loop plays silence. The CLI validates
            // sample rate before publishing; this is a defensive fallback.
            self.transport = Transport::Stopped;
            self.gain = 0.0;
            self.position = 0;
            return 0.0;
        }

        let mut value = current.samples[self.position];
        if self.crossfade_position < self.crossfade_length {
            let phase = self.crossfade_position as f32 / self.crossfade_length as f32;
            value = self.previous_head[self.crossfade_position] * (1.0 - phase) + value * phase;
            self.crossfade_position += 1;
        }

        self.position += 1;
        if self.position == current.samples.len() {
            self.position = 0;
        }

        let gain_step = 1.0 / self.fade_frames as f32;
        match self.transport {
            Transport::Playing => {
                self.gain = (self.gain + gain_step).min(1.0);
            }
            Transport::Stopping => {
                self.gain = (self.gain - gain_step).max(0.0);
                if self.gain == 0.0 {
                    self.position = 0;
                    self.transport = Transport::Stopped;
                    self.crossfade_length = 0;
                }
            }
            Transport::Stopped => {}
        }
        (value * self.gain * MASTER_LEVEL).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequencer::PreparedLoop;
    use std::sync::atomic::Ordering;
    use triple_buffer::triple_buffer;

    fn setup() -> (
        triple_buffer::Input<PreparedLoop>,
        Arc<AtomicBool>,
        RealtimePlayer,
    ) {
        let initial = PreparedLoop { samples: vec![], bpm: 120.0, sample_rate: 8_000 };
        let (producer, consumer) = triple_buffer(&initial);
        let playing = Arc::new(AtomicBool::new(false));
        let engine = RealtimePlayer::new(consumer, playing.clone(), 8_000).unwrap();
        (producer, playing, engine)
    }

    fn loop_with(value: f32, samples: usize) -> PreparedLoop {
        PreparedLoop { samples: vec![value; samples], bpm: 120.0, sample_rate: 8_000 }
    }

    #[test]
    fn publication_is_immediate_while_stopped() {
        let (mut input, playing, mut engine) = setup();
        input.write(loop_with(0.3, 1000));
        for _ in 0..20 {
            assert_eq!(engine.tick(), 0.0);
        }
        assert_eq!(engine.output.output_buffer().samples[0], 0.3);
        playing.store(true, Ordering::Relaxed);
        assert!(engine.tick() > 0.0);
    }

    #[test]
    fn buffered_pattern_waits_for_the_next_bar() {
        let (mut input, playing, mut engine) = setup();
        input.write(loop_with(0.2, 1000));
        engine.tick();
        playing.store(true, Ordering::Relaxed);
        for _ in 0..100 {
            engine.tick();
        }
        input.write(loop_with(0.6, 1000));
        for _ in 0..900 {
            engine.tick();
        }
        assert_eq!(engine.position, 0);
        assert_eq!(engine.output.output_buffer().samples[0], 0.2);
        engine.tick();
        assert_eq!(engine.output.output_buffer().samples[0], 0.6);
        // Crossfade should reach the new pattern without a hard jump.
        for _ in 0..200 { engine.tick(); }
        assert!(engine.tick() > 0.4 * MASTER_LEVEL);
    }

    #[test]
    fn newest_groove_overwrites_unconsumed_versions() {
        let (mut input, playing, mut engine) = setup();
        input.write(loop_with(0.2, 1000));
        engine.tick();
        playing.store(true, Ordering::Relaxed);
        engine.tick();
        input.write(loop_with(0.3, 1000));
        input.write(loop_with(0.7, 1000));
        for _ in 0..1000 { engine.tick(); }
        assert_eq!(engine.output.output_buffer().samples[0], 0.7);
    }

    #[test]
    fn stop_and_restart_have_short_fades() {
        let (mut input, playing, mut engine) = setup();
        input.write(loop_with(0.5, 1000));
        engine.tick();
        playing.store(true, Ordering::Relaxed);
        for _ in 0..140 { engine.tick(); }
        let before_stop = engine.tick();
        assert!(before_stop > 0.3 * MASTER_LEVEL);
        playing.store(false, Ordering::Relaxed);
        for _ in 0..160 { engine.tick(); }
        assert_eq!(engine.transport, Transport::Stopped);
        assert_eq!(engine.tick(), 0.0);
        playing.store(true, Ordering::Relaxed);
        let first = engine.tick();
        assert!(first > 0.0 && first < 0.05);
        assert_eq!(engine.position, 1);
    }

    #[test]
    fn tempo_change_can_use_a_different_bar_length() {
        let (mut input, playing, mut engine) = setup();
        input.write(loop_with(0.4, 800));
        engine.tick();
        playing.store(true, Ordering::Relaxed);
        engine.tick();
        input.write(loop_with(0.6, 1200));
        for _ in 0..799 { engine.tick(); }
        assert_eq!(engine.output.output_buffer().samples.len(), 800);
        engine.tick();
        assert_eq!(engine.output.output_buffer().samples.len(), 1200);
        for _ in 0..1200 { engine.tick(); }
        assert_eq!(engine.output.output_buffer().samples.len(), 1200);
    }

    #[test]
    fn invalid_sample_rate_is_rejected() {
        let initial = PreparedLoop { samples: vec![], bpm: 120.0, sample_rate: 8_000 };
        let (_, consumer) = triple_buffer(&initial);
        assert!(RealtimePlayer::new(consumer, Arc::new(AtomicBool::new(false)), 0).is_err());
    }
}
