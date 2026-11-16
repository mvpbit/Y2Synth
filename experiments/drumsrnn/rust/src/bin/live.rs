//! Interactive live DrumsRNN preview for macOS (CPAL / CoreAudio).
//!
//! The network and per-bar drum rendering run ONLY on the CLI/control thread.
//! Audio receives pre-rendered samples through a lock-free triple buffer.
//! Playback state is an atomic flag. The callback does not own or destroy
//! audio buffers. This remains a prototype pending xrun/latency profiling.

use std::env;
use std::error::Error;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use triple_buffer::triple_buffer;
use std::time::{SystemTime, UNIX_EPOCH};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, Stream, StreamConfig};

use y2synth_drumsrnn_weights::{
    attention::AttentionRnn,
    generation::{generate, GeneratedPattern, Sampling, DRUM_PITCHES},
    kit::{pitch_by_name, DrumKit, VOICE_NAMES},
    load_checkpoint,
    realtime::RealtimePlayer,
    sequencer::{prepare_loop_with_kit, PreparedLoop},
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

fn next_pattern(
    network: &AttentionRnn<'_>,
    seed: u64,
    temperature: f32,
    bpm: f32,
    sample_rate: u32,
    kit: &DrumKit,
) -> Result<(GeneratedPattern, PreparedLoop), Box<dyn Error>> {
    // Always start from scratch: no user-supplied MIDI primer.
    let pattern = generate(network, 16, Sampling::Temperature(temperature), seed)?;
    let buffer = prepare_loop_with_kit(&pattern, bpm, sample_rate, kit)?;
    Ok((pattern, buffer))
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

    println!("Y2Synth / live AI drum experiment");
    println!("Output: {sample_rate} Hz, {channels} channels, {sample_format:?}");
    println!("Loading Magenta DrumsRNN checkpoint…");
    let checkpoint = load_checkpoint(&options.weights)?;
    let network = AttentionRnn::from_checkpoint(&checkpoint)?;
    let mut bpm = options.bpm;
    let mut seed = options.seed;
    let mut kit = DrumKit::default();
    let mut current_pattern = next_pattern(
        &network, seed, options.temperature, bpm, sample_rate, &kit,
    )?;
    show_pattern(&current_pattern.0, seed);

    // Two spare triple-buffer slots are owned by the control thread and one
    // by audio. Producer overwrites/deallocates old loops only on this thread.
    let initial = PreparedLoop {
        samples: vec![],
        bpm,
        sample_rate,
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
    // Publish first pattern before the CoreAudio callback starts.
    input.write(current_pattern.1);
    stream.play()?;

    println!();
    println!("Ready. Enter: play | stop | bpm <40..240> | regenerate | kit | help | quit");
    println!("AI inference and synth rendering run on this control thread, not the audio callback.");
    let stdin = io::stdin();
    let mut lines = stdin.lock();
    loop {
        print!("y2> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if lines.read_line(&mut line)? == 0 {
            break;
        }
        let words: Vec<_> = line.split_whitespace().collect();
        if words.is_empty() { continue; }
        match words.as_slice() {
            ["play"] => {
                playing.store(true, Ordering::Relaxed);
                println!("Playing (loop on).");
            }
            ["stop"] => {
                playing.store(false, Ordering::Relaxed);
                println!("Stopping with short fade.");
            }
            ["regenerate"] => {
                seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let (pattern, buffer) =
                    next_pattern(&network, seed, options.temperature, bpm, sample_rate, &kit)?;
                input.write(buffer);
                println!("New AI-generated pattern published for next bar (or immediately if stopped).");
                show_pattern(&pattern, seed);
                current_pattern.0 = pattern;
            }
            ["bpm", raw] => {
                match raw.parse::<f32>().ok().and_then(|value| {
                    validate_bpm(value).ok().map(|_| value)
                }) {
                    Some(next_bpm) => {
                        let buffer = prepare_loop_with_kit(
                            &current_pattern.0, next_bpm, sample_rate, &kit
                        )?;
                        input.write(buffer);
                        bpm = next_bpm;
                        println!("Tempo {bpm} BPM queued for next bar (or immediately if stopped).");
                    }
                    None => println!("Invalid BPM. Enter a number from 40 to 240."),
                }
            }
            ["kit"] => show_kit(&kit),
            ["reset-kit"] => {
                kit.reset();
                let buffer = prepare_loop_with_kit(
                    &current_pattern.0, bpm, sample_rate, &kit
                )?;
                input.write(buffer);
                println!("Kit restored to defaults; update scheduled at the next bar.");
            }
            ["level", name, raw] | ["tone", name, raw] | ["decay", name, raw] => {
                let parameter = words[0];
                match (pitch_by_name(name), raw.parse::<f32>()) {
                    (Some(pitch), Ok(value)) => {
                        match kit.set(pitch, parameter, value) {
                            Ok(()) => {
                                let buffer = prepare_loop_with_kit(
                                    &current_pattern.0, bpm, sample_rate, &kit,
                                )?;
                                input.write(buffer);
                                println!(
                                    "{parameter} {name} = {value:.2} (next bar or immediately while stopped)"
                                );
                            }
                            Err(error) => println!("Invalid {parameter}: {error}"),
                        }
                    }
                    _ => println!("Invalid voice or value; enter 'help' for syntax."),
                }
            }
            ["status"] => println!(
                "Requested pattern: {} hits, {bpm} BPM, temperature {}, last seed {seed}.",
                current_pattern.0.hits.len(), options.temperature
            ),
            ["quit"] | ["exit"] => break,
            ["help"] => usage(),
            _ => println!("Unknown command. Try 'help' for available drum controls."),
        }
    }
    // Stream shutdown happens on the CLI thread. No WAV file is created.
    playing.store(false, Ordering::Relaxed);
    drop(stream);
    println!("Bye.");
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
