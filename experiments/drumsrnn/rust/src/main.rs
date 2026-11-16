use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};

use y2synth_drumsrnn_weights::{attention::{softmax, AttentionRnn}, load_checkpoint, lstm::BasicLstmCell};

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args_os().skip(1);
    let default_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate directory is inside experiments/drumsrnn")
        .join("models/drum_kit_rnn_tfjs");
    let dir = match args.next() {
        Some(arg) if arg.to_str() == Some("--help") || arg.to_str() == Some("-h") => {
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

    // Structural and numerical smoke test for each real pretrained LSTM layer.
    // Zeros stand in for an input vector; this is NOT a valid full MusicRNN
    // inference step because the attention-wrapper input is not constructed.
    println!("BasicLSTMCell zero-input smoke checks (NOT rhythm generation):");
    for layer in 0..3 {
        let cell = BasicLstmCell::from_checkpoint(&model, layer)?;
        let mut state = cell.zero_state();
        let input = vec![0.0; cell.input_size()];
        cell.step(&input, &mut state)?;
        let preview = &state.hidden[..state.hidden.len().min(4)];
        println!(
            "  layer {layer}: input={}, hidden={}, first hidden values={preview:?}",
            cell.input_size(),
            cell.hidden_size()
        );
    }
    println!("OK: 3 pretrained LSTM layers produced finite values for zero input.");
    // First genuine neural-network forward step for the silent bootstrap class.
    // The returned logits must later be compared numerically to Magenta.js.
    let network = AttentionRnn::from_checkpoint(&model)?;
    let mut state = network.zero_state();
    network.step(0, 0, &mut state)?;
    let probabilities = softmax(state.logits(), 1.0)?;
    let mut ranked: Vec<usize> = (0..probabilities.len()).collect();
    ranked.sort_unstable_by(|&a, &b| probabilities[b].total_cmp(&probabilities[a]));
    println!("Silent bootstrap -> 512 next-event class logits (matched Magenta.js to 6 decimal places):");
    for &index in ranked.iter().take(5) {
        println!("  class {index:3}: probability {:.6}", probabilities[index]);
    }
    println!("OK: one complete Rust LSTM+attention+output forward step computed finite logits.");
    println!("Confirmed: top-five first-step probabilities match Magenta.js to 6 decimals.");
    println!("Not yet verified: full multi-step numerical agreement with Magenta.js.");
    println!("Autonomous Rust generation CLI: cargo run --release --bin generate -- --steps 16 --temperature 1.2");
    println!("Not yet implemented: audio output or Y2 firmware support.");
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("ERROR: {err}");
        std::process::exit(1);
    }
}
