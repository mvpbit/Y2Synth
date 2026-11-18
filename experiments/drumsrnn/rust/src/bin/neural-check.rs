//! CI-only numerical comparison to official Magenta.js, using downloaded weights.
use std::{env,fs,path::Path};
use y2synth_drumsrnn_weights::{neural::{load_model,melody_tokens,multitrack_tokens,NeuralSynths},generation::Sampling,synth1::SynthPreset};
fn main() -> Result<(),Box<dyn std::error::Error>> {
    let args: Vec<_>=env::args().collect();
    let root=Path::new(args.get(1).ok_or("model root")?);
    let reference: serde_json::Value=serde_json::from_str(&fs::read_to_string(args.get(2).ok_or("reference JSON")?)?)?;
    let melody=load_model(&root.join("melody_rnn_tfjs"),SynthPreset::Pluck)?;
    let actual=melody_tokens(&melody,Sampling::Argmax,0)?;
    let expected: Vec<usize>=serde_json::from_value(reference["melody"].clone())?;
    assert_eq!(actual,expected,"MelodyRNN argmax differs from Magenta.js");
    drop(melody);
    let pad=load_model(&root.join("multitrack_chords_tfjs"),SynthPreset::Pad)?;
    let z:Vec<f32>=serde_json::from_value(reference["z"].clone())?;
    let actual=multitrack_tokens(&pad,&z,13,Sampling::Argmax,0)?;
    let expected:Vec<usize>=serde_json::from_value(reference["pad"].clone())?;
    for (i,(&a,&b)) in actual.iter().zip(&expected).enumerate() { assert_eq!(a,b,"PAD token {i} differs from Magenta.js"); }
    assert_eq!(actual.len(),expected.len());
    drop(pad);
    let mut models=NeuralSynths::new(root,42);
    for preset in [SynthPreset::Pluck,SynthPreset::Pad] {
        let first=models.generate(preset,0.75)?;
        let second=models.generate(preset,0.75)?;
        first.validate()?;second.validate()?;
        assert!(!first.notes.is_empty());assert!(!second.notes.is_empty());
        assert_ne!(first.notes,second.notes,"Repeated seed advancement must change generated notes");
        println!("{preset:?}: {} / {} sampled notes",first.notes.len(),second.notes.len());
    }
    println!("PASS: MelodyRNN 16 + MusicVAE 512 argmax tokens match official Magenta.js; stochastic inference valid.");
    Ok(())
}
