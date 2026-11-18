//! Native inference for the selected PLUCK and PAD checkpoints.
//! Scalar reference implementation. All loading/inference stays on the control thread.
use crate::{Checkpoint, Tensor, load_checkpoint_filtered};
use crate::lstm::{BasicLstmCell, LstmState};
use crate::generation::{Sampling, SamplingRng};
use crate::harmony::Key;
use crate::synth1::{NoteEvent, SynthPreset, SynthTrack};
use std::{io, path::{Path, PathBuf}};

fn invalid(s: impl Into<String>) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, s.into()) }
fn tensor<'a>(cp: &'a Checkpoint, name: &str) -> io::Result<&'a Tensor> {
    cp.tensors.get(name).ok_or_else(|| invalid(format!("Missing tensor {name}")))
}
fn dense(cp: &Checkpoint, prefix: &str, input: &[f32], rnn: bool) -> io::Result<Vec<f32>> {
    let k = tensor(cp, &format!("{prefix}/{}", if rnn { "weights" } else { "kernel" }))?;
    let b = tensor(cp, &format!("{prefix}/{}", if rnn { "biases" } else { "bias" }))?;
    let n = b.values.len();
    if k.shape != [input.len(), n] || b.shape != [n] { return Err(invalid("Dense shape mismatch")); }
    let mut out = b.values.clone();
    for (row, &x) in input.iter().enumerate() {
        for (y, &w) in out.iter_mut().zip(&k.values[row*n..(row+1)*n]) { *y += x*w; }
    }
    if out.iter().any(|x| !x.is_finite()) { return Err(invalid("Non-finite dense result")); }
    Ok(out)
}
struct Stack<'a> { cells: Vec<BasicLstmCell<'a>>, states: Vec<LstmState> }
impl<'a> Stack<'a> {
    fn new(cp: &'a Checkpoint, prefix: &str, suffix: &str, layers: usize, input: usize, hidden: usize) -> io::Result<Self> {
        let mut cells = Vec::new();
        for i in 0..layers {
            let p = format!("{prefix}/cell_{i}/{suffix}");
            let c = BasicLstmCell::from_tensors(tensor(cp,&format!("{p}/kernel"))?, tensor(cp,&format!("{p}/bias"))?)?;
            if c.hidden_size() != hidden || c.input_size() != if i == 0 { input } else { hidden } {
                return Err(invalid("Unexpected checkpoint LSTM dimensions"));
            }
            cells.push(c);
        }
        let states = cells.iter().map(|c| c.zero_state()).collect();
        Ok(Self { cells, states })
    }
    fn initialize(&mut self, values: &[f32]) -> io::Result<()> {
        let n = self.cells[0].hidden_size();
        if values.len() != self.cells.len()*2*n { return Err(invalid("Initial state shape mismatch")); }
        for (i, state) in self.states.iter_mut().enumerate() {
            for j in 0..n {
                state.cell[j] = values[2*i*n+j].tanh();
                state.hidden[j] = values[(2*i+1)*n+j].tanh();
            }
        }
        Ok(())
    }
    fn step(&mut self, input: &[f32]) -> io::Result<Vec<f32>> {
        let mut x = input.to_vec();
        for (cell, state) in self.cells.iter().zip(self.states.iter_mut()) {
            cell.step(&x, state)?;
            x.clone_from(&state.hidden);
        }
        Ok(x)
    }
}
fn draw(logits: &[f32], sampling: Sampling, rng: &mut SamplingRng) -> io::Result<usize> {
    if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) { return Err(invalid("Invalid logits")); }
    let top = logits.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0))).unwrap().0;
    let Sampling::Temperature(t) = sampling else { return Ok(top); };
    if !t.is_finite() || t <= 0.0 { return Err(invalid("Invalid temperature")); }
    let weights: Vec<f64> = logits.iter().map(|&v| ((v-logits[top]) as f64 / t as f64).exp()).collect();
    let mut target = rng.unit_f64()*weights.iter().sum::<f64>();
    for (i, p) in weights.iter().enumerate() { target -= p; if target < 0.0 { return Ok(i); } }
    Ok(weights.len()-1)
}
/// A silent internal bootstrap (class zero), followed by 16 predicted events.
pub fn melody_tokens(cp: &Checkpoint, sampling: Sampling, seed: u64) -> io::Result<Vec<usize>> {
    let mut stack = Stack::new(cp, "rnn/multi_rnn_cell", "basic_lstm_cell", 2, 130, 512)?;
    let mut rng = SamplingRng::new(seed);
    let mut input = vec![0.0;130]; input[0] = 1.0;
    let mut result = Vec::with_capacity(16);
    for _ in 0..16 {
        let h = stack.step(&input)?;
        let logits = dense(cp, "fully_connected", &h, true)?;
        if logits.len() != 130 { return Err(invalid("Expected melody_rnn vocabulary")); }
        let token = draw(&logits, sampling, &mut rng)?;
        result.push(token); input.fill(0.0); input[token] = 1.0;
    }
    Ok(result)
}
/// Decode a fixed latent vector, useful for deterministic Magenta.js parity checks.
/// Chord index follows TriadChordEncoder: NC=0, major=1+root, minor=13+root.
pub fn multitrack_tokens(cp: &Checkpoint, z: &[f32], chord: usize, sampling: Sampling, seed: u64) -> io::Result<Vec<usize>> {
    if z.len() != 256 || z.iter().any(|v| !v.is_finite()) || chord >= 49 { return Err(invalid("Invalid latent/chord")); }
    let mut conductor = Stack::new(cp,"decoder/hierarchical_level_0","lstm_cell",3,1,256)?;
    conductor.initialize(&dense(cp,"decoder/hierarchical_level_0/initial_state",z,false)?)?;
    let mut core = Stack::new(cp,"core_decoder/decoder/multi_rnn_cell","lstm_cell",3,795,256)?;
    let mut previous = vec![0.0;490];
    let mut rng = SamplingRng::new(seed);
    let mut tokens = Vec::with_capacity(512);
    for _ in 0..8 {
        let embedding = conductor.step(&[0.0])?;
        core.initialize(&dense(cp,"core_decoder/decoder/z_to_initial_state",&embedding,false)?)?;
        for event in 0..64 {
            let mut input = previous.clone();
            input.extend_from_slice(&embedding);
            let mut control = [0.0;49]; control[if event == 0 {0} else {chord}] = 1.0;
            input.extend_from_slice(&control);
            let h = core.step(&input)?;
            let logits = dense(cp,"core_decoder/decoder/output_projection",&h,false)?;
            if logits.len() != 490 { return Err(invalid("Expected multitrack_chords vocabulary")); }
            let token = draw(&logits,sampling,&mut rng)?;
            tokens.push(token); previous.fill(0.0); previous[token] = 1.0;
        }
    }
    Ok(tokens)
}
pub fn melody_track(tokens: &[usize], level: f32) -> io::Result<SynthTrack> {
    if tokens.len() != 16 || tokens.iter().any(|&t| t >= 130) { return Err(invalid("Invalid melody tokens")); }
    let mut notes = Vec::new();
    let mut active: Option<(usize,u8)> = None;
    for step in 0..=16 {
        let t = if step == 16 {1} else {tokens[step]};
        if t != 0 {
            if let Some((start,pitch)) = active.take() {
                if step > start { notes.push(NoteEvent {step:start,pitch,length_steps:step-start,velocity:0.75}); }
            }
            if t >= 2 { active = Some((step,{ let mut p=(t-2) as i32; while p<55 {p+=12;} while p>84 {p-=12;} p as u8 })); }
        }
    }
    let track = SynthTrack {preset:SynthPreset::Pluck,notes,level};
    track.validate()?;
    if track.notes.is_empty() { return Err(invalid("MelodyRNN returned silence")); }
    Ok(track)
}
#[cfg(test)]
fn normalize_pad(raw: &[(usize,usize,u8,f32)]) -> Vec<NoteEvent> { normalize_pad_in_key(raw,Key::default()) }
fn normalize_pad_in_key(raw: &[(usize,usize,u8,f32)], key: Key) -> Vec<NoteEvent> {
    let mut result: Vec<NoteEvent> = Vec::new();
    for &(start,end,pitch,velocity) in raw {
        if end <= start { continue; }
        let step = (start+3)/6;
        if step >= 16 { continue; }
        let end = ((end+3)/6).max(step+1).min(16);
        let pitch = key.snap(pitch as i32,48);
        if let Some(old) = result.iter_mut().find(|n| n.step == step && n.pitch == pitch) {
            old.length_steps = old.length_steps.max(end-step);
        } else { result.push(NoteEvent {step,pitch,length_steps:end-step,velocity}); }
    }
    result.sort_by_key(|n| (n.step,n.pitch));
    let mut accepted: Vec<NoteEvent> = Vec::new();
    let mut voices = [0usize;16];
    for n in result {
        if accepted.len() >= 48 { break; }
        if voices[n.step..n.step+n.length_steps].iter().any(|&v| v >= 6) { continue; }
        for v in &mut voices[n.step..n.step+n.length_steps] { *v += 1; }
        accepted.push(n);
    }
    accepted
}
/// Split eight performance streams; select a genuinely polyphonic nondrum part.
pub fn multitrack_track(tokens: &[usize], level: f32) -> io::Result<SynthTrack> {
    multitrack_track_in_key(tokens,level,Key::default())
}
pub fn multitrack_track_in_key(tokens: &[usize], level: f32, key: Key) -> io::Result<SynthTrack> {
    if tokens.len() != 512 || tokens.iter().any(|&t| t >= 490) { return Err(invalid("Invalid multitrack tokens")); }
    let mut best = Vec::new();
    let mut best_score = (0,0);
    for part in tokens.chunks_exact(64) {
        let mut active: Vec<Vec<(usize,f32)>> = vec![Vec::new();128];
        let mut raw = Vec::new();
        let mut time = 0usize;
        let mut velocity = 0.75;
        let mut program = None;
        for &t in part {
            match t {
                0..=127 => active[t].push((time,velocity)),
                128..=255 => {
                    let p = t-128;
                    if !active[p].is_empty() {
                        let (start,v) = active[p].remove(0);
                        raw.push((start,time.min(96),p as u8,v));
                    }
                }
                256..=351 => time += t-256+1,
                352..=359 => velocity = (1+16*(t-352)) as f32/127.0,
                360..=488 => { if program.is_none() { program=Some(t-360); } }
                489 => break,
                _ => unreachable!(),
            }
        }
        if program == Some(128) { continue; }
        for (p, queue) in active.iter().enumerate() {
            for &(start,v) in queue { raw.push((start,96,p as u8,v)); }
        }
        let notes = normalize_pad_in_key(&raw,key);
        let mut overlaps = 0;
        for step in 0..16 {
            let count = notes.iter().filter(|n| n.step<=step && n.step+n.length_steps>step).count();
            if count >= 2 { overlaps += count; }
        }
        let score = (overlaps,notes.iter().map(|n| n.length_steps).sum());
        if overlaps > 0 && score > best_score { best_score=score; best=notes; }
    }
    if best.is_empty() { return Err(invalid("MusicVAE returned no polyphonic PAD part")); }
    let track = SynthTrack {preset:SynthPreset::Pad,notes:best,level};
    track.validate()?; Ok(track)
}
/// Lazily loaded model instances have independent random streams and fixed roles.
pub struct NeuralSynths {
    root: PathBuf, melody: Option<Checkpoint>, pad: Option<Checkpoint>, melody_seed: u64, pad_seed: u64, last_pad_key: Option<Key>,
}
impl NeuralSynths {
    pub fn new(root: impl Into<PathBuf>, seed: u64) -> Self { Self {root:root.into(),melody:None,pad:None,melody_seed:seed,pad_seed:seed ^ 0x504144,last_pad_key:None} }
    pub fn generate(&mut self, preset: SynthPreset, level: f32) -> io::Result<SynthTrack> {
        let state = match preset { SynthPreset::Pluck => &mut self.melody_seed, SynthPreset::Pad => &mut self.pad_seed };
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let seed = *state;
        let (slot,name) = match preset {
            SynthPreset::Pluck => (&mut self.melody,"melody_rnn"),
            SynthPreset::Pad => (&mut self.pad,"multitrack_chords"),
        };
        if slot.is_none() {
            let path = self.root.join(format!("{name}_tfjs"));
            *slot = Some(load_model(&path, preset).map_err(|e| invalid(format!("{name}: {e}. Download with ./download-model.sh {name}")))?);
        }
        let cp = slot.as_ref().unwrap();
        let mut rng = SamplingRng::new(seed);
        let choices: Vec<Key>=(0..24).map(|n|Key::new(n%12,n<12)).filter(|key|Some(*key)!=self.last_pad_key).collect();
        let key=choices[(rng.unit_f64()*choices.len() as f64) as usize];
        let mut error = invalid("No usable neural phrase");
        for attempt in 0..3 {
            let generated = match preset {
                SynthPreset::Pluck => melody_track(&melody_tokens(cp,Sampling::Temperature(1.05),seed.wrapping_add(attempt))?,level),
                SynthPreset::Pad => {
                    let mut z = Vec::with_capacity(256);
                    for _ in 0..128 {
                        let radius = (-2.0*(1.0-rng.unit_f64()).ln()).sqrt();
                        let angle = std::f64::consts::TAU*rng.unit_f64();
                        z.push((radius*angle.cos()) as f32); z.push((radius*angle.sin()) as f32);
                    }
                    multitrack_track_in_key(&multitrack_tokens(cp,&z,key.chord_index(),Sampling::Temperature(0.8),seed.wrapping_add(attempt))?,level,key)
                }
            };
            match generated { Ok(track) => { if preset==SynthPreset::Pad {self.last_pad_key=Some(key);} return Ok(track); }, Err(e) => error=e }
        }
        Err(error)
    }
}
impl crate::application::Synth1Generator for NeuralSynths {
    fn generate_synth1(&mut self, preset: SynthPreset, level: f32, _: crate::melody::RhythmContext) -> io::Result<SynthTrack> {
        self.generate(preset,level)
    }
    fn harmony_key(&self) -> Option<Key> {self.last_pad_key}
}
pub fn load_model(path: &Path, preset: SynthPreset) -> io::Result<Checkpoint> {
    load_checkpoint_filtered(path, |name| preset == SynthPreset::Pluck ||
        name.starts_with("decoder/hierarchical_level_0/") ||
        name.starts_with("core_decoder/decoder/"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn melody_sustain_noteoff_and_last_gate() {
        let mut t = vec![0;16]; t[0]=62; t[4]=1; t[7]=69;
        let n = melody_track(&t,0.7).unwrap().notes;
        assert_eq!((n[0].step,n[0].pitch,n[0].length_steps),(0,60,4));
        assert_eq!((n[1].step,n[1].pitch,n[1].length_steps),(7,67,9));
        assert!(melody_track(&[0;16],0.7).is_err());
    }
    #[test]
    fn pad_keeps_chords_converts_96_steps_and_excludes_drums() {
        let mut tokens=vec![489;512];
        tokens[..9].copy_from_slice(&[360,356,60,63,67,303,188,191,195]);
        // Competing drum stream is excluded even with long overlapping notes.
        tokens[64..69].copy_from_slice(&[488,36,38,42,351]);
        let track=multitrack_track(&tokens,0.5).unwrap();
        assert_eq!(track.notes.len(),3);
        assert!(track.notes.iter().all(|n| n.step==0 && n.length_steps==8));
        tokens[..9].fill(489);
        assert!(multitrack_track(&tokens,0.5).is_err());
    }
    #[test]
    fn pad_voice_limit_deduplicates_octave_folded_notes() {
        let raw: Vec<_> = (0..128).map(|p|(0,96,p,0.75)).collect();
        let notes=normalize_pad(&raw);
        assert_eq!(notes.len(),6);
        assert!(notes.iter().all(|n| n.length_steps==16));
    }
}
