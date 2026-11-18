/* Shared note adaptation for browser inference and remote integration checks.
 * Neural checkpoints produce notes, never the Pluck/Pad audio timbres.
 */
(function(root){
  "use strict";
  const SCALE=[0,2,3,5,7,8,10];
  function pitchInRange(pitch,low,high){
    while(pitch<low)pitch+=12;
    while(pitch>high)pitch-=12;
    return pitch;
  }
  function minorPitch(pitch){
    let best=pitch;
    for(let distance=0;distance<=2;distance++){
      if(SCALE.includes((pitch-distance+120)%12)){best=pitch-distance;break;}
      if(SCALE.includes((pitch+distance)%12)){best=pitch+distance;break;}
    }
    return best;
  }
  function normalize(sequence,track){
    if(!sequence||!Array.isArray(sequence.notes))throw Error("Invalid model output");
    const resolution=sequence.quantizationInfo?.stepsPerQuarter;
    if(!Number.isFinite(resolution)||resolution<=0)throw Error("Invalid model timing");
    const ratio=4/resolution,notes=[];
    for(const note of sequence.notes){
      if(note.isDrum)continue;
      const start=note.quantizedStartStep,end=note.quantizedEndStep;
      if(!Number.isInteger(note.pitch)||note.pitch<0||note.pitch>127||
         !Number.isFinite(start)||!Number.isFinite(end)||start<0||end<=start)continue;
      const step=Math.round(start*ratio),stop=Math.min(16,Math.round(end*ratio));
      if(step<0||step>=16||stop<=step)continue;
      const pitch=pitchInRange(minorPitch(note.pitch),track==="pad"?48:55,track==="pad"?84:84);
      const velocity=Number.isFinite(note.velocity)&&note.velocity>0?
        Math.max(.2,Math.min(1,note.velocity/127)):.75;
      notes.push({step,pitch,len:stop-step,velocity,instrument:note.instrument??0});
    }
    return notes.sort((a,b)=>a.step-b.step||a.pitch-b.pitch);
  }
  function polyphony(notes){
    let maximum=0,overlap=0;
    for(let step=0;step<16;step++){
      const count=new Set(notes.filter(n=>n.step<=step&&step<n.step+n.len).map(n=>n.pitch)).size;
      maximum=Math.max(maximum,count);overlap+=Math.max(0,count-1);
    }
    return {maximum,overlap};
  }
  function instrumentNotes(sequence,track){
    let notes=normalize(sequence,track);
    if(track==="pad"){
      const groups=new Map();
      for(const note of notes){
        if(!groups.has(note.instrument))groups.set(note.instrument,[]);
        groups.get(note.instrument).push(note);
      }
      const candidates=[...groups.values()].filter(group=>polyphony(group).maximum>=2);
      candidates.sort((a,b)=>polyphony(b).overlap-polyphony(a).overlap||
        b.reduce((sum,n)=>sum+n.len,0)-a.reduce((sum,n)=>sum+n.len,0));
      if(!candidates.length)throw Error("No polyphonic PAD phrase; try again");
      notes=candidates[0];
    }
    // Merge quantization collisions; cap voices to protect the prototype mixer.
    const unique=new Map(),out=[];
    for(const note of notes){
      const key=note.step+":"+note.pitch,existing=unique.get(key);
      if(existing){existing.len=Math.max(existing.len,note.len);continue;}
      const active=out.filter(n=>n.step<=note.step&&note.step<n.step+n.len);
      if(active.length>=(track==="pad"?6:1)){
        if(track==="pluck")for(const n of active)n.len=note.step-n.step;
        else continue;
      }
      const item={step:note.step,pitch:note.pitch,len:note.len,velocity:note.velocity};
      unique.set(key,item);out.push(item);
      if(out.length>=48)break;
    }
    const result=out.filter(n=>n.len>0);
    if(!result.length)throw Error("Model returned silence; try again");
    if(track==="pad"&&polyphony(result).maximum<2)throw Error("No polyphonic PAD phrase; try again");
    return result;
  }
  function drumNotes(sequence){
    if(!sequence||!Array.isArray(sequence.notes))throw Error("Invalid drum output");
    const result={kick:[],snare:[],hat:[]},hits={kick:new Set(),snare:new Set(),hat:new Set()};
    const resolution=sequence.quantizationInfo?.stepsPerQuarter;
    if(!Number.isFinite(resolution)||resolution<=0)throw Error("Invalid drum timing");
    for(const note of sequence.notes){
      const step=Math.round(note.quantizedStartStep*4/resolution);
      if(!Number.isInteger(step)||step<0||step>15)continue;
      // The browser kit has three voices; the Rust kit retains all nine.
      const voice=note.pitch===36?"kick":note.pitch===38?"snare":
        [42,46,49,51].includes(note.pitch)?"hat":[45,48,50].includes(note.pitch)?"kick":null;
      if(voice)hits[voice].add(step);
    }
    for(const voice of Object.keys(result))result[voice]=[...hits[voice]].sort((a,b)=>a-b);
    if(!Object.values(result).some(hits=>hits.length))throw Error("Model returned silence; try again");
    return result;
  }
  root.NeuralPatterns={instrumentNotes,drumNotes,polyphony};
})(typeof self!=="undefined"?self:globalThis);
