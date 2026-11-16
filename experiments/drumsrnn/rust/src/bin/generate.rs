//! Minimal local experiment: generate MIDI-style drum events from scratch.
//! No browser, TensorFlow, audio synthesis or user primer is required.

use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use y2synth_drumsrnn_weights::{
    attention::AttentionRnn,
    generation::{generate, Sampling, DRUM_PITCHES},
    load_checkpoint,
};

struct Options {
    steps: usize,
    temperature: Option<f32>,
    seed: u64,
    checkpoint: PathBuf,
}

fn usage() {
    println!("Y2Synth / Magenta DrumsRNN — autonomous generator (experiment)");
    println!("Usage: cargo run --release --bin generate -- [OPTIONS]");
    println!("  --steps N           Generated steps, 1..=4096 (default 16)");
    println!("  --temperature T     Stochastic sampling with T > 0 (default 1.2)");
    println!("  --argmax            Deterministic most-likely-class selection");
    println!("  --seed N            RNG seed (u64); prints seed for reproducibility");
    println!("  --weights PATH      Directory containing already downloaded checkpoint");
    println!("  -h, --help          Show this help");
}

fn parse_options() -> Result<Option<Options>, Box<dyn Error>> {
    let default_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("Expected crate nested inside experiments/drumsrnn")?
        .join("models/drum_kit_rnn_tfjs");

    let seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
        ^ ((std::process::id() as u64) << 32);

    let mut options = Options {
        steps: 16,
        temperature: Some(1.2),
        seed,
        checkpoint: default_dir,
    };

    let mut args = env::args().skip(1);
    let mut explicit_temperature = false;
    let mut explicit_argmax = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                usage();
                return Ok(None);
            }
            "--steps" => options.steps = args.next().ok_or("Missing --steps N")?.parse()?,
            "--temperature" => {
                options.temperature = Some(args.next().ok_or("Missing --temperature T")?.parse()?);
                explicit_temperature = true;
            }
            "--argmax" => {
                options.temperature = None;
                explicit_argmax = true;
            }
            "--seed" => options.seed = args.next().ok_or("Missing --seed N")?.parse()?,
            "--weights" => options.checkpoint = PathBuf::from(
                args.next().ok_or("Missing --weights PATH")?
            ),
            _ => return Err(format!("Unknown option: {arg}. Use --help.").into()),
        }
    }
    if explicit_argmax && explicit_temperature {
        return Err("--argmax and --temperature cannot be combined".into());
    }
    if options.steps == 0 || options.steps > 4096 {
        return Err("--steps must be in 1..=4096".into());
    }
    if let Some(t) = options.temperature {
        if !t.is_finite() || t <= 0.0 {
            return Err("--temperature must be finite and > 0".into());
        }
    }
    Ok(Some(options))
}

fn run() -> Result<(), Box<dyn Error>> {
    let Some(options) = parse_options()? else {
        return Ok(());
    };

    let start = Instant::now();
    let checkpoint = load_checkpoint(&options.checkpoint)?;
    let network = AttentionRnn::from_checkpoint(&checkpoint)?;
    let load_time = start.elapsed();
    let sampling = match options.temperature {
        Some(t) => Sampling::Temperature(t),
        None => Sampling::Argmax,
    };

    let started = Instant::now();
    let pattern = generate(&network, options.steps, sampling, options.seed)?;
    let generation_time = started.elapsed();

    println!("Y2Synth / DrumsRNN (Rust experimental autonomous generation)");
    println!("Weights: {}", options.checkpoint.display());
    println!("Mode: from scratch (one internal silent bootstrap step)");
    println!("Steps: {}", options.steps);
    println!("Sampling: {}", match sampling {
        Sampling::Argmax => String::from("argmax"),
        Sampling::Temperature(t) => format!("temperature {t}"),
    });
    println!("RNG seed: {}", options.seed);
    println!("Load: {:.1} ms", load_time.as_secs_f64() * 1000.0);
    println!("Generation: {:.1} ms", generation_time.as_secs_f64() * 1000.0);
    println!("Generated hits: {}", pattern.hits.len());
    println!("Drum classes: {:?}", pattern.classes);

    // Simple 9-line text grid; each column is one sixteenth-note step.
    println!();
    println!("Rhythm grid (X=hit, .=rest):");
    let names = ["Kick", "Snare", "Closed HH", "Open HH", "Low tom", "Mid tom",
                 "High tom", "Crash", "Ride"];
    for (bit, &pitch) in DRUM_PITCHES.iter().enumerate() {
        let row: String = pattern.classes.iter()
            .map(|class| if (class & (1usize << bit)) != 0 { 'X' } else { '.' })
            .collect();
        println!("  {:<10} MIDI {:>2}: {}", names[bit], pitch, row);
    }

    println!();
    println!("Hits [step, MIDI pitch]:");
    for hit in &pattern.hits {
        println!("  [{}, {}]", hit.step, hit.pitch);
    }
    println!();
    println!("Diagnostic next-class probabilities:");
    for step in &pattern.diagnostics {
        println!("  Generated step {} — top 5 [id, probability]:", step.step);
        for (id, p) in step.top_five {
            println!("    [{}, {:.6}]", id, p);
        }
    }

    println!("NOTE: Rust outputs not yet verified across all steps vs Magenta.js.");
    println!("NOTE: No audio or Y2 firmware support is implemented.");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ERROR: {error}");
        std::process::exit(1);
    }
}
