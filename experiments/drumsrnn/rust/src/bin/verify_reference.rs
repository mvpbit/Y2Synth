//! Compare all 8192 class probabilities against Magenta.js from-scratch Argmax.
//!
//! Run from the repository root after exporting the full reference JSON:
//! cargo run --release --locked --manifest-path experiments/drumsrnn/rust/Cargo.toml \
//!   --bin verify_reference -- --reference ~/Downloads/drumsrnn-reference.json
//!
//! No sampling randomness: same silent bootstrap and Argmax trajectory.

use std::env;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufReader};
use std::path::{Path, PathBuf};
use std::process;

use y2synth_drumsrnn_weights::{
    attention::AttentionRnn,
    generation::{generate_with_observer, Sampling},
    load_checkpoint,
    verification::{compare_probabilities, infer_teacher_forced, MagentaReference, TraceMode,
        DEFAULT_TOLERANCE, STEPS, CLASSES},
};

struct Options {
    reference: PathBuf,
    weights: PathBuf,
    tolerance: f64,
}

fn usage() {
    println!("Compare all 16 × 512 DrumsRNN class probabilities: Rust vs pinned Magenta.js");
    println!("Usage: cargo run --release --bin verify_reference -- --reference FILE [OPTIONS]");
    println!("  --reference FILE    Saved JSON from reference.html (scratch Argmax)");
    println!("                      OR teacher-forcing.html (diagnostic forced inputs)");
    println!("                      Choose From scratch + Argmax in the browser");
    println!("  --weights DIR       Local checkpoint directory (default: experiments/drumsrnn/models/drum_kit_rnn_tfjs)");
    println!("  --tolerance N       Maximum allowed absolute error per probability (default: 1e-5)");
    println!("  -h, --help          Show help");
}

fn parse_options() -> Result<Option<Options>, Box<dyn Error>> {
    let default_weights = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().ok_or("Expected crate inside experiments/drumsrnn")?
        .join("models/drum_kit_rnn_tfjs");
    let mut reference = None;
    let mut weights = default_weights;
    let mut tolerance = DEFAULT_TOLERANCE;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                usage();
                return Ok(None);
            }
            "--reference" => reference = Some(PathBuf::from(args.next().ok_or("--reference requires FILE")?)),
            "--weights" => weights = PathBuf::from(args.next().ok_or("--weights requires DIR")?),
            "--tolerance" => tolerance = args.next().ok_or("--tolerance requires N")?.parse()?,
            other => return Err(format!("Unknown option: {other}; use --help").into()),
        }
    }
    if !tolerance.is_finite() || tolerance <= 0.0 || tolerance > 0.1 {
        return Err("--tolerance must be finite and in (0, 0.1]".into());
    }
    let reference = reference.ok_or("--reference FILE is required (export JSON from browser)")?;
    Ok(Some(Options { reference, weights, tolerance }))
}

fn run() -> Result<(), Box<dyn Error>> {
    let Some(options) = parse_options()? else { return Ok(()); };
    let reference: MagentaReference =
        serde_json::from_reader(BufReader::new(File::open(&options.reference)?))?;
    reference.validate()?;

    let checkpoint = load_checkpoint(&options.weights)?;
    let network = AttentionRnn::from_checkpoint(&checkpoint)?;
    let (probabilities, predicted_classes) = match reference.mode()? {
        TraceMode::ScratchArgmax => {
            let mut probabilities = Vec::with_capacity(STEPS);
            let pattern = generate_with_observer(
                &network, STEPS, Sampling::Argmax, 0,
                |step, row| {
                    if step != probabilities.len() || row.len() != CLASSES {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Unexpected inference trace shape/order",
                        ));
                    }
                    probabilities.push(row.to_vec());
                    Ok(())
                },
            )?;
            (probabilities, pattern.classes)
        }
        TraceMode::TeacherForced => {
            let forced = reference.forced_classes.as_deref().expect("validated above");
            let trace = infer_teacher_forced(&network, forced)?;
            (trace.probabilities, trace.predicted_classes)
        }
    };

    let report = compare_probabilities(
        &reference, &probabilities, &predicted_classes, options.tolerance,
    )?;
    println!("Y2Synth DrumsRNN full-distribution numerical verification");
    println!("Reference: {}", options.reference.display());
    println!("JS: {} (backend: {})", reference.implementation,
        reference.tfjs_backend.as_deref().unwrap_or("unknown"));
    println!("Checkpoint: {}", reference.checkpoint);
    match reference.mode()? {
        TraceMode::ScratchArgmax => {
            println!("Mode: From scratch, 1 internal silent step, Argmax");
        }
        TraceMode::TeacherForced => {
            println!("Mode: DIAGNOSTIC teacher forcing, one silent bootstrap then forced history");
            println!("Forced preceding classes: {:?}", reference.forced_classes.as_ref().unwrap());
            println!("These are fixed inputs, NOT generated output or user performance.");
        }
    }
    println!("Compared: {STEPS} steps × {CLASSES} classes = {} probabilities", STEPS * CLASSES);
    println!("Tolerance (absolute): {:.2e}", report.tolerance);
    println!("Rust classes: {:?}", predicted_classes);
    println!("Magenta.js:   {:?}", reference.classes);
    println!();
    println!("step  max_abs_error  worst_class  mean_abs_error   total_variation  >tolerance");
    for step in &report.steps {
        println!("{:>4}  {:>13.6e}  {:>11}  {:>14.6e}  {:>15.6e}  {:>10}",
            step.step, step.max_abs, step.worst_class, step.mean_abs,
            step.total_variation, step.over_tolerance);
    }
    println!();
    println!("Worst probability: step {}, class {}, absolute error {:.9e}",
        report.worst_step, report.worst_class, report.max_abs);
    println!("Mean absolute error: {:.9e}", report.mean_abs);
    println!("RMS absolute error:  {:.9e}", report.rms_abs);
    println!("Probability values above tolerance: {} / {}", report.over_tolerance, STEPS * CLASSES);
    println!("Different Argmax classes: {} / {STEPS}", report.mismatched_classes);

    if report.passed() {
        println!("PASS: all {STEPS} × {CLASSES} values within tolerance, and all classes match");
        Ok(())
    } else {
        println!("FAIL: numerical differences exceed tolerance and/or selected classes differ");
        Err(io::Error::new(io::ErrorKind::Other, "Magenta.js / Rust numerical comparison failed").into())
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ERROR: {error}");
        process::exit(1);
    }
}
