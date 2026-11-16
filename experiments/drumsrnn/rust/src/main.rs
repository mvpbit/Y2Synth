use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};

use y2synth_drumsrnn_weights::load_checkpoint;

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let default_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate directory is inside experiments/drumsrnn")
        .join("models/drum_kit_rnn_tfjs");
    let dir = match args.next() {
        Some(arg) if arg == "--help" || arg == "-h" => {
            println!("Usage: y2synth-drumsrnn-weights [CHECKPOINT_DIR]");
            println!("Reads offline Magenta.js weights. Does not generate music yet.");
            return Ok(());
        }
        Some(arg) => PathBuf::from(arg),
        None => default_path,
    };
    if args.next().is_some() {
        return Err("Expected at most one CHECKPOINT_DIR argument".into());
    }

    let model = load_checkpoint(&dir)?;
    println!("Y2Synth / Magenta.js DrumsRNN weight loader");
    println!("Checkpoint: {}", dir.display());
    println!("Attention length: {}", model.attention_length);
    println!("Tensors: {}", model.tensors.len());
    println!("Float32 parameters: {}", model.parameter_count());
    println!(
        "Decoded weight bytes: {} ({:.2} MiB)",
        model.weights_bytes(),
        model.weights_bytes() as f64 / 1024.0 / 1024.0
    );
    for (name, tensor) in &model.tensors {
        println!(
            "  {}: {:?}, {} values",
            name,
            tensor.shape,
            tensor.element_count()
        );
    }

    // A tiny sanity check beyond the JSON metadata: read actual decoded floats.
    if let Some(tensor) = model.tensors.get("fully_connected/biases") {
        let n = tensor.values.len().min(4);
        println!("First {n} output-bias values: {:?}", &tensor.values[..n]);
    }
    if model.tensors.len() != 16 || model.parameter_count() != 2_967_850 {
        return Err(
            "Loaded checkpoint does not match the 16-tensor/2,967,850-parameter DrumsRNN studied on Mac".into(),
        );
    }

    println!("OK: model weights decoded as finite little-endian float32.");
    println!("Not implemented: LSTM, attention, pattern generation, audio, Y2 support.");
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("ERROR: {err}");
        std::process::exit(1);
    }
}
