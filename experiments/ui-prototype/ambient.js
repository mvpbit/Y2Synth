"use strict";
class AmbientStream {
  constructor(key) {
    this.key = key;
    this.history = [];
    this.previous = [];
    this.hats = [];
    this.chord = [];
    this.padDue = true;
  }
  setHarmony(pattern) {
    const onsets = [...new Set(pattern.map(note => note.step))];
    const chords = onsets.map(step => [...new Set(pattern
      .filter(note => note.step <= step && step < note.step + note.len)
      .map(note => note.pitch))].sort((a,b) => a-b).slice(0,4)).filter(chord => chord.length >= 2);
    if (!chords.length) throw Error("No polyphonic PAD phrase");
    const old = this.chord;
    const distance = chord => chord.reduce((sum,pitch) =>
      sum + Math.min(...old.map(prior => Math.abs(prior-pitch))),0) / chord.length;
    const fresh = chords.filter(chord => chord.join() !== old.join());
    this.chord = !old.length ? chords[0] : (fresh.length ? fresh : chords).sort((a,b)=>distance(a)-distance(b))[0];
    this.harmonyFor = 2 + Math.floor(Math.random()*4);
    this.padDue = true;
  }
  primer() {
    if (!this.previous.length) return undefined;
    return {quantizationInfo:{stepsPerQuarter:4},totalQuantizedSteps:16,
      notes:this.previous.map(n => ({pitch:n.pitch,quantizedStartStep:n.step,quantizedEndStep:n.step+n.len}))};
  }
  next(pattern) {
    const pad = this.chord.map(pitch => ({step:0,pitch,len:16,velocity:.52,instrument:"pad"}));
    const phrase = [];
    for (const n of pattern) {
      if (phrase.length && n.step < phrase.at(-1).step + 3) continue;
      phrase.push({...n,len:Math.min(2,n.len),velocity:n.velocity*.47,instrument:"pluck"});
      if (phrase.length === 4) break;
    }
    const melody = NeuralPatterns.harmonizePluck(phrase,pad,this.key);
    // Compare audible pitches and onsets. Changing only velocity is not a new phrase.
    const signature = () => melody.map(n => n.step+"/"+n.pitch).join(",");
    if (this.history.includes(signature())) {
      const last = melody.at(-1);
      const lower = melody.length > 1 ? melody.at(-2).step + melody.at(-2).len : 0;
      const pitches = NeuralPatterns.scaleFor(this.key);
      // A new answer can be a moved final note or an additional, quiet pickup.
      const variants = [];
      for(let step=lower;step<16;step++) for(let pitch=57;pitch<=81;pitch++)
        if(pitches.includes(pitch%12)) variants.push({step,pitch});
      const offset = Math.floor(Math.random()*variants.length);
      for(let i=0;i<variants.length && this.history.includes(signature());i++) {
        Object.assign(last, variants[(i+offset)%variants.length]);
        last.len = Math.min(last.len,16-last.step);
      }
      if(this.history.includes(signature())) {
        // A crowded bar can leave too few endings. Thin it to a two-note answer.
        melody.splice(1);
        melody[0].step=0; melody[0].len=1;
        const answer={...melody[0],step:4};
        melody.push(answer);
        outer: for(let step=3;step<16;step++) for(let pitch=57;pitch<=81;pitch++) {
          if(!pitches.includes(pitch%12))continue;
          Object.assign(answer,{step,pitch});
          if(!this.history.includes(signature()))break outer;
        }
      }
    }
    this.history.push(signature());
    if(this.history.length>64)this.history.shift();
    this.previous=melody;
    const notes=[...melody];
    // Renew sustained voices within the scheduler's two-bar gate limit.
    if(this.padDue) {
      const hold=Math.min(this.harmonyFor,2);
      notes.push(...pad.map(n=>({...n,len:16*hold})));
      this.padWait=hold;
    }
    this.harmonyFor--;
    this.padDue=--this.padWait===0;
    notes.push({instrument:"pulse",step:0,velocity:.75+Math.random()*.12});
    if(Math.random()<.86) notes.push({instrument:"pulse",step:8,velocity:.48+Math.random()*.14});
    for(const step of this.hats.filter(s=>s%2===0 && Math.random()<.22).slice(0,2))
      notes.push({instrument:"tick",step,velocity:.23+Math.random()*.12});
    return notes;
  }
}
