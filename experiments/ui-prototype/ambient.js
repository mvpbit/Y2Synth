"use strict";
// Each section lasts eight bars. Hold chord voicings for two bars and
// revisit the same short motif before introducing another model phrase.
class AmbientSection {
  constructor(pad, pluck, drums) {
    this.key = pad.key;
    this.chords = [...new Set(pad.pattern.map(note => note.step))].map(step =>
      [...new Set(pad.pattern.filter(note => note.step <= step && step < note.step + note.len)
        .map(note => note.pitch))].sort((a,b) => a-b).slice(0,4)
    ).filter(chord => chord.length >= 2);
    if (!this.chords.length) throw Error("No polyphonic PAD phrase");
    this.motif = pluck.pattern.filter((note,index,notes) =>
      index === 0 || note.step >= notes[index-1].step + 2).slice(0,4);
    this.hats = drums.pattern.hat.filter(step => step % 4 === 2).slice(0,2);
    this.bar = 0;
  }
  next() {
    const index = this.bar++;
    const chord = this.chords[Math.floor(index / 2) % this.chords.length];
    const pad = chord.map(pitch => ({step:0, pitch, len:16, velocity:.52, instrument:"pad"}));
    const notes = index % 2 ? [] : pad.map(note => ({...note,len:32}));
    const phrase = this.motif.map((note,i) => ({
      ...note, step:note.step, len:Math.min(note.len,2),
      pitch:note.pitch + (index % 4 === 3 && i === this.motif.length-1 ? 5 : 0),
      velocity:.45 * note.velocity, instrument:"pluck"
    }));
    notes.push(...NeuralPatterns.harmonizePluck(phrase,pad,this.key));
    for (const step of [0,8]) notes.push({instrument:"pulse",step,velocity:step===0?.85:.6});
    if(index % 2) for(const step of this.hats) notes.push({instrument:"tick",step,velocity:.35});
    return notes;
  }
}
