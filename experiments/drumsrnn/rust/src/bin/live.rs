//! Interactive live DrumsRNN preview for macOS (CPAL / CoreAudio).
//!
//! The network and per-bar drum rendering run ONLY on the CLI/control thread.
//! Audio receives pre-rendered samples through a lock-free triple buffer.
//! Playback state is an atomic flag. The callback does not own or destroy
//! audio buffers. This remains a prototype pending xrun/latency profiling.

use std::cell::Cell;
use std::env;
use std::error::Error;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use triple_buffer::{triple_buffer, Input};
use std::time::{SystemTime, UNIX_EPOCH};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, Stream, StreamConfig};

use y2synth_drumsrnn_weights::{
    application::{AppResponse, ApplicationController, AudioPort, DrumGenerator, Synth1Generator},
    attention::AttentionRnn,
    generation::{generate, GeneratedPattern, Sampling, DRUM_PITCHES},
    kit::{pitch_by_name, DrumKit, VOICE_NAMES},
    load_checkpoint,
    melody::{compose, RhythmContext},
    project::TrackId,
    realtime::RealtimePlayer,
    sequencer::PreparedLoop,
    synth1::{NoteEvent, SynthPreset, SynthTrack},
    ui::{Command as MenuCommand, InputEvent, MenuController, Screen},
};

struct Options {
    weights: PathBuf,
    temperature: f32,
    bpm: f32,
    seed: u64,
}

fn usage() {
    println!("Y2Synth live DrumsRNN / macOS prototype");
    println!("cargo run --release --features live-audio --bin live -- [OPTIONS]");
    println!("  --bpm N            Initial tempo: 40..240 (default 120)");
    println!("  --temperature T    Model sampling temperature > 0 (default 1.2)");
    println!("  --seed N           Initial RNG seed (u64)");
    println!("  --weights PATH     Local downloaded Magenta checkpoint directory");
    println!("  -h, --help         Show help");
    println!();
    println!("While running:");
    println!("  play | stop | bpm <40..240> | regenerate | status | quit");
    println!("  kit | level <voice> <0..1.5> | tone <voice> <0..1>");
    println!("  decay <voice> <0.5..2> | reset-kit");
    println!("  synth1 pluck | synth1 pad | synth1 generate | synth1 show | synth1 clear");
    println!("  synth1 level <0..1> | synth1 note <step 0..15> <MIDI 36..96> <length 1..16>");
    println!("  menu | center | next | prev | back | wheel-play");
    println!("  (menu commands simulate the Y2 wheel; 'play' works anywhere)");
    println!("Voices: kick snare chh ohh lowtom midtom hightom crash ride");
}

fn parse_options() -> Result<Option<Options>, Box<dyn Error>> {
    let default_weights = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("Expected crate inside experiments/drumsrnn")?
        .join("models/drum_kit_rnn_tfjs");
    let seed = SystemTime::now().duration_since(UNIX_EPOCH)?
        .as_nanos() as u64
        ^ (u64::from(std::process::id()) << 32);
    let mut options = Options {
        weights: default_weights,
        temperature: 1.2,
        bpm: 120.0,
        seed,
    };

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                usage();
                return Ok(None);
            }
            "--weights" => {
                options.weights = PathBuf::from(args.next().ok_or("--weights needs PATH")?);
            }
            "--bpm" => {
                options.bpm = args.next().ok_or("--bpm needs N")?.parse()?;
            }
            "--temperature" => {
                options.temperature = args.next().ok_or("--temperature needs T")?.parse()?;
            }
            "--seed" => {
                options.seed = args.next().ok_or("--seed needs N")?.parse()?;
            }
            _ => return Err(format!("Unknown option: {arg} (use --help)").into()),
        }
    }
    validate_bpm(options.bpm)?;
    if !options.temperature.is_finite() || options.temperature <= 0.0 {
        return Err("Temperature must be finite and > 0".into());
    }
    Ok(Some(options))
}

fn validate_bpm(bpm: f32) -> Result<(), Box<dyn Error>> {
    if !bpm.is_finite() || !(40.0..=240.0).contains(&bpm) {
        return Err("BPM must be finite and in 40..=240".into());
    }
    Ok(())
}

/// No inference, audio preparation or files are touched by the audio callback.
fn build_stream<T>(
    device: &cpal::Device,
    config: StreamConfig,
    mut player: RealtimePlayer,
) -> Result<Stream, Box<dyn Error>>
where
    T: Sample + SizedSample + FromSample<f32>,
{
    let channels = config.channels as usize;
    let stream = device.build_output_stream(
        config,
        move |output: &mut [T], _info| {
            // Only atomic flag reads, triple-buffer slot swaps and copying
            // prepared samples; no channel locks, allocations or buffer frees.
            for frame in output.chunks_mut(channels) {
                let mono = player.tick();
                for sample in frame {
                    *sample = T::from_sample(mono);
                }
            }
        },
        move |error| eprintln!("CPAL output error: {error}"),
        None,
    )?;
    Ok(stream)
}

fn show_pattern(pattern: &GeneratedPattern, seed: u64) {
    println!("AI seed: {seed}; drum hits: {}", pattern.hits.len());
    for (bit, pitch) in DRUM_PITCHES.iter().enumerate() {
        let row: String = pattern.classes.iter()
            .map(|class| if *class & (1usize << bit) != 0 { 'X' } else { '.' })
            .collect();
        println!("  MIDI {pitch:>2}: {row}");
    }
    if pattern.hits.is_empty() {
        println!("  This random pattern is silent. Use 'regenerate' to try a new one.");
    }
}

fn show_kit(kit: &DrumKit) {
    println!("Current drum kit (level / tone / decay):");
    for (name, params) in VOICE_NAMES.iter().zip(kit.voices().iter()) {
        println!(
            "  {name:<8} {:>4.2} / {:>4.2} / {:>4.2}",
            params.level, params.tone, params.decay
        );
    }
}

/// Backend adapter: only the CONTROL thread uses this. The CoreAudio
/// callback continues to own the Output half of the triple buffer.
struct LiveAudio<'a> {
    input: &'a mut Input<PreparedLoop>,
    playing: &'a Arc<AtomicBool>,
}

impl AudioPort for LiveAudio<'_> {
    fn queue_bar(&mut self, bar: PreparedLoop) -> io::Result<()> {
        self.input.write(bar);
        Ok(())
    }

    fn start(&mut self) -> io::Result<()> {
        self.playing.store(true, Ordering::Relaxed);
        Ok(())
    }
}

fn apply_command<G: DrumGenerator, S: Synth1Generator>(
    app: &mut ApplicationController,
    command: MenuCommand,
    drum_generator: &mut G,
    synth_generator: &mut S,
    input: &mut Input<PreparedLoop>,
    playing: &Arc<AtomicBool>,
) -> io::Result<AppResponse> {
    let mut audio = LiveAudio { input, playing };
    app.dispatch_with_synth(command, drum_generator, synth_generator, &mut audio)
}

fn describe_result(result: AppResponse) {
    match result {
        AppResponse::PlayingLast { mode, bpm } => {
            println!("PLAY: saved project ({mode:?}), tempo {bpm} BPM.");
        }
        AppResponse::DrumGenerated { hits } => {
            println!("AI: {hits} drum hits; new loop queued and saved in RAM.");
        }
        AppResponse::Synth1Generated { notes, preset } => {
            println!("AI composer: {notes} notes for Synth 1 / {} (procedural, not ML).",
                preset.name());
            println!("New musical phrase queued for next bar and saved in RAM.");
        }
        AppResponse::EditDrums => {
            println!("Drums editor: kit | level kick 0.5 | tone snare 0.8 | decay chh 1.5");
        }
        AppResponse::EditSynth1 => {
            println!("Synth 1 USER editor: synth1 pluck | synth1 pad | synth1 show");
            println!("  synth1 level 0.7 | synth1 note 0 60 2 | synth1 clear");
        }
        AppResponse::NotImplemented(feature) => println!("Not available yet: {feature}"),
        AppResponse::NothingSaved => println!("No saved music project to play."),
    }
}

fn show_synth1(app: &ApplicationController) {
    if let Some(track) = app.session().current().synth1() {
        let synth = &track.content;
        println!("Synth 1: {} | level {:.2} | {} notes | source {:?}",
            synth.preset.name(), synth.level, synth.notes.len(), track.source);
        for note in &synth.notes {
            println!("  step {:>2}: MIDI {:>3}, length {:>2}, velocity {:.2}",
                note.step, note.pitch, note.length_steps, note.velocity);
        }
    } else {
        println!("Synth 1 is empty. Use 'synth1 pluck' or 'synth1 pad'.");
    }
}

fn show_menu(menu: &MenuController) {
    match menu.screen() {
        Screen::Home => println!("MENU: PLAY (wheel-play to play last, center to enter settings)"),
        Screen::PlaySettings => println!(
            "MENU: PLAY SETTINGS [USER / AI] — selected {:?} (next/prev, center, back)",
            menu.mode_focus(),
        ),
        Screen::UserPage(track) => println!(
            "MENU: USER / {track:?} (next/prev track, center to edit, back)"
        ),
        Screen::AiMenu => println!(
            "MENU: AI / {:?} (next/prev, center to select, back)",
            menu.ai_focus(),
        ),
        Screen::AiTrackSelect => println!(
            "MENU: AI / GENERATE TRACK / {:?} (next/prev, center to generate, back)",
            menu.track_focus(),
        ),
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let Some(options) = parse_options()? else { return Ok(()); };

    let host = cpal::default_host();
    let device = host.default_output_device()
        .ok_or("No default audio output device found (check macOS Sound settings)")?;
    let supported = device.default_output_config()?;
    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.into();
    let sample_rate = config.sample_rate;
    let channels = config.channels;
    if channels == 0 {
        return Err("Default audio output has zero channels".into());
    }

    println!("Y2Synth / live AI drum experiment + menu coordinator");
    println!("Output: {sample_rate} Hz, {channels} channels, {sample_format:?}");
    println!("Loading Magenta DrumsRNN checkpoint…");
    let checkpoint = load_checkpoint(&options.weights)?;
    let network = AttentionRnn::from_checkpoint(&checkpoint)?;
    let initial_pattern = generate(
        &network, 16, Sampling::Temperature(options.temperature), options.seed,
    )?;
    show_pattern(&initial_pattern, options.seed);

    // The triple buffer and real-time callback are unchanged. Application
    // commands publish finished loops from this control/CLI thread only.
    let initial = PreparedLoop {
        samples: vec![], bpm: options.bpm, sample_rate,
    };
    let (mut input, output) = triple_buffer(&initial);
    let playing = Arc::new(AtomicBool::new(false));
    let player = RealtimePlayer::new(output, playing.clone(), sample_rate)?;
    let stream = match sample_format {
        SampleFormat::F32 => build_stream::<f32>(&device, config, player)?,
        SampleFormat::I16 => build_stream::<i16>(&device, config, player)?,
        SampleFormat::U16 => build_stream::<u16>(&device, config, player)?,
        SampleFormat::I32 => build_stream::<i32>(&device, config, player)?,
        SampleFormat::U32 => build_stream::<u32>(&device, config, player)?,
        SampleFormat::F64 => build_stream::<f64>(&device, config, player)?,
        other => return Err(format!("Unsupported default audio sample format: {other:?}").into()),
    };

    let mut app = ApplicationController::new(sample_rate, options.bpm)?;
    {
        let mut audio = LiveAudio { input: &mut input, playing: &playing };
        app.install_initial_drums(initial_pattern, DrumKit::default(), &mut audio)?;
    }
    stream.play()?; // Audio stream runs, transport remains stopped until PLAY.

    let seed = Cell::new(options.seed);
    let mut drum_generator = || -> io::Result<GeneratedPattern> {
        let next_seed = seed.get().wrapping_add(0x9e37_79b9_7f4a_7c15);
        let pattern = generate(
            &network, 16, Sampling::Temperature(options.temperature), next_seed,
        )?;
        seed.set(next_seed);
        Ok(pattern)
    };
    // Separate seed sequence for composed melodic phrases; does not alter the
    // seed progression or checkpoint state of the existing DrumsRNN generator.
    let melody_seed = Cell::new(options.seed ^ 0xa57d_3e12_5b42_c908);
    let mut synth_generator = |preset: SynthPreset, level: f32, rhythm: RhythmContext|
        -> io::Result<SynthTrack> {
        let next_seed = melody_seed.get().wrapping_add(0x9e37_79b9_7f4a_7c15);
        let phrase = compose(next_seed, preset, level, rhythm)?;
        melody_seed.set(next_seed);
        Ok(phrase)
    };
    let mut menu = MenuController::default();

    println!();
    println!("Ready: play | stop | regenerate | synth1 generate | synth1 pluck | synth1 pad | menu | help | quit");
    println!("Wheel simulation: center | next | prev | back | wheel-play");
    println!("DrumsRNN: neural AI; Synth 1: autonomous rule-based melodic composer.");
    println!("New loops and edits are saved in RAM, not to disk.");
    let stdin = io::stdin();
    let mut lines = stdin.lock();
    loop {
        print!("y2> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if lines.read_line(&mut line)? == 0 { break; }
        let words: Vec<_> = line.split_whitespace().collect();
        if words.is_empty() { continue; }

        match words.as_slice() {
            ["play"] => {
                let result = apply_command(
                    &mut app, MenuCommand::PlayLast, &mut drum_generator, &mut synth_generator,
                    &mut input, &playing,
                )?;
                describe_result(result);
            }
            ["stop"] => {
                playing.store(false, Ordering::Relaxed);
                println!("Stopping with short fade.");
            }
            ["regenerate"] => {
                let result = apply_command(
                    &mut app, MenuCommand::GenerateTrack(TrackId::Drums),
                    &mut drum_generator, &mut synth_generator, &mut input, &playing,
                )?;
                describe_result(result);
                if let Some(drums) = app.session().current().drums() {
                    show_pattern(&drums.content.pattern, seed.get());
                }
            }
            ["bpm", raw] => match raw.parse::<f32>().ok()
                .and_then(|value| validate_bpm(value).ok().map(|_| value))
            {
                Some(bpm) => {
                    let mut audio = LiveAudio { input: &mut input, playing: &playing };
                    app.set_bpm(bpm, &mut audio)?;
                    println!("Tempo {bpm} BPM queued for next bar and saved in RAM.");
                }
                None => println!("Invalid BPM. Enter a number from 40 to 240."),
            },
            ["kit"] => {
                if let Some(drums) = app.session().current().drums() {
                    show_kit(&drums.content.kit);
                } else {
                    println!("No drum kit loaded.");
                }
            }
            ["synth1", "pluck"] | ["synth1", "pad"] => {
                let preset = if words[1] == "pluck" {
                    SynthPreset::Pluck
                } else {
                    SynthPreset::Pad
                };
                let synth = SynthTrack::demo(preset);
                let mut audio = LiveAudio { input: &mut input, playing: &playing };
                app.set_synth1(synth, &mut audio)?;
                println!("Synth 1: {} melodic phrase queued for next bar; saved in RAM.",
                    preset.name());
                show_synth1(&app);
            }
            ["synth1", "generate"] => {
                let result = apply_command(
                    &mut app, MenuCommand::GenerateTrack(TrackId::Synth1),
                    &mut drum_generator, &mut synth_generator, &mut input, &playing,
                )?;
                describe_result(result);
                show_synth1(&app);
            }
            ["synth1", "show"] => show_synth1(&app),
            ["synth1", "clear"] => {
                if let Some(existing) = app.session().current().synth1() {
                    let mut empty = existing.content.clone();
                    empty.notes.clear();
                    let mut audio = LiveAudio { input: &mut input, playing: &playing };
                    app.set_synth1(empty, &mut audio)?;
                    println!("Synth 1 notes cleared; drums remain unchanged.");
                } else {
                    println!("Synth 1 is already empty.");
                }
            }
            ["synth1", "level", raw] => {
                if let (Some(existing), Ok(level)) = (
                    app.session().current().synth1(), raw.parse::<f32>(),
                ) {
                    let mut edited = existing.content.clone();
                    edited.level = level;
                    if let Err(err) = edited.validate() {
                        println!("Invalid Synth 1 level: {err}");
                    } else {
                        let mut audio = LiveAudio { input: &mut input, playing: &playing };
                        app.set_synth1(edited, &mut audio)?;
                        println!("Synth 1 level {level:.2} queued and saved in RAM.");
                    }
                } else {
                    println!("Choose 'synth1 pluck'/'synth1 pad' first, then level 0..1.");
                }
            }
            ["synth1", "note", step, pitch, length] => {
                match (
                    step.parse::<usize>(),
                    pitch.parse::<u8>(),
                    length.parse::<usize>(),
                ) {
                    (Ok(step), Ok(pitch), Ok(length_steps)) => {
                        let note = NoteEvent {
                            step, pitch, length_steps, velocity: 0.75,
                        };
                        if let Err(error) = note.validate() {
                            println!("Invalid note: {error}");
                        } else if let Some(existing) = app.session().current().synth1() {
                            let mut synth = existing.content.clone();
                            synth.notes.push(note);
                            if let Err(error) = synth.validate() {
                                println!("Cannot add note: {error}");
                            } else {
                                let mut audio = LiveAudio { input: &mut input, playing: &playing };
                                app.set_synth1(synth, &mut audio)?;
                                println!("Synth 1 note added; new mixed bar queued.");
                            }
                        } else {
                            println!("Choose 'synth1 pluck' or 'synth1 pad' first.");
                        }
                    }
                    _ => println!("Syntax: synth1 note <step 0..15> <MIDI 36..96> <length 1..16>"),
                }
            }
            ["reset-kit"] => {
                let mut audio = LiveAudio { input: &mut input, playing: &playing };
                if app.set_drum_kit(DrumKit::default(), &mut audio)? {
                    println!("Kit reset and saved in RAM; update queued for the next bar.");
                }
            }
            ["level", name, raw] | ["tone", name, raw] | ["decay", name, raw] => {
                let parameter = words[0];
                let existing = app.session().current().drums()
                    .map(|drums| drums.content.kit).unwrap_or_default();
                match (pitch_by_name(name), raw.parse::<f32>()) {
                    (Some(pitch), Ok(value)) => {
                        let mut updated = existing;
                        match updated.set(pitch, parameter, value) {
                            Ok(()) => {
                                let mut audio = LiveAudio { input: &mut input, playing: &playing };
                                if app.set_drum_kit(updated, &mut audio)? {
                                    println!("{parameter} {name} = {value:.2} (queued, saved in RAM)");
                                }
                            }
                            Err(error) => println!("Invalid {parameter}: {error}"),
                        }
                    }
                    _ => println!("Invalid voice or value; enter 'help' for syntax."),
                }
            }
            ["menu"] => show_menu(&menu),
            ["center"] | ["next"] | ["prev"] | ["back"] | ["wheel-play"] => {
                let event = match words[0] {
                    "center" => InputEvent::CenterPressed,
                    "next" => InputEvent::Next,
                    "prev" => InputEvent::Previous,
                    "back" => InputEvent::BackPressed,
                    _ => InputEvent::PlayPressed,
                };
                if let Some(command) = menu.handle(event) {
                    let result = apply_command(
                        &mut app, command, &mut drum_generator, &mut synth_generator, &mut input, &playing,
                    )?;
                    describe_result(result);
                    if let AppResponse::Synth1Generated { .. } = result {
                        show_synth1(&app);
                    }
                    if let AppResponse::DrumGenerated { .. } = result {
                        if let Some(drums) = app.session().current().drums() {
                            show_pattern(&drums.content.pattern, seed.get());
                        }
                    }
                }
                show_menu(&menu);
            }
            ["status"] => {
                let current = app.session().current();
                let hits = current.drums()
                    .map(|drums| drums.content.pattern.hits.len()).unwrap_or(0);
                println!(
                    "Saved project: {} drum hits, {} BPM, mode {:?}, temperature {}, seed {}.",
                    hits, current.bpm(), current.mode(), options.temperature, seed.get(),
                );
                show_synth1(&app);
                show_menu(&menu);
            }
            ["quit"] | ["exit"] => break,
            ["help"] => usage(),
            _ => println!("Unknown command. Try 'help'."),
        }
    }
    playing.store(false, Ordering::Relaxed);
    drop(stream);
    println!("Bye. Saved snapshot was RAM-only; quitting clears it.");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ERROR: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tempo_validation_rejects_nan_and_out_of_range() {
        for bpm in [40.0, 120.0, 240.0] {
            assert!(validate_bpm(bpm).is_ok());
        }
        for bpm in [0.0, 39.9, 240.1, f32::NAN, f32::INFINITY] {
            assert!(validate_bpm(bpm).is_err());
        }
    }
}
