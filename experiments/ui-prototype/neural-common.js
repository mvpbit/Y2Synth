/* Shared note adaptation for browser inference and remote integration checks.
 * Neural checkpoints produce notes, never the Pluck/Pad audio timbres.
 */
(function(root){
  "use strict";
  const NAMES=["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"];
  function validKey(key){return key&&Number.isInteger(key.root)&&key.root>=0&&key.root<12&&["major","minor"].includes(key.mode);}
  function scaleFor(key){
    if(!validKey(key))throw Error("Invalid PAD key");
    return (key.mode==="minor"?[0,2,3,5,7,8,10]:[0,2,4,5,7,9,11]).map(n=>(n+key.root)%12);
  }
  function keyName(key){return NAMES[key.root]+" "+key.mode;}
  function randomKey(previous){
    const keys=Array.from({length:24},(_,i)=>({root:i%12,mode:i<12?"minor":"major"}))
      .filter(k=>!previous||k.root!==previous.root||k.mode!==previous.mode);
    return keys[Math.floor(Math.random()*keys.length)];
  }
  function chordFor(key){return NAMES[key.root]+(key.mode==="minor"?"m":"");}
  function pitchInRange(pitch,low,high){
    while(pitch<low)pitch+=12;
    while(pitch>high)pitch-=12;
    return pitch;
  }
  function nearestPitch(pitch,classes,low,high){
    let best=null;
    for(let candidate=low;candidate<=high;candidate++){
      if(classes.includes(candidate%12)&&(best===null||Math.abs(candidate-pitch)<Math.abs(best-pitch)))best=candidate;
    }
    if(best===null)throw Error("No playable harmony tones");
    return best;
  }
  function harmonyAt(pad,step,key){
    return [...new Set(pad.filter(n=>n.step<=step&&step<n.step+n.len).map(n=>n.pitch%12))]
      .filter(pc=>scaleFor(key).includes(pc)).sort((a,b)=>a-b);
  }
  function harmonizePluck(notes,pad,key){
    const scale=scaleFor(key),out=[];
    for(const note of notes){
      let start=note.step;
      const stop=Math.min(16,note.step+note.len);
      while(start<stop){
        const tones=harmonyAt(pad,start,key);
        let end=start+1;
        while(end<stop&&harmonyAt(pad,end,key).join(",")===tones.join(","))end++;
        // Strong beats and held notes use the sounding PAD tones.
        // Short off-beat passing notes may use other tones of the shared key.
        const allowed=tones.length&&(start%4===0||end-start>1)?tones:scale;
        const pitch=nearestPitch(pitchInRange(note.pitch,55,84),allowed,55,84);
        out.push({...note,step:start,len:end-start,pitch});
        start=end;
      }
    }
    return out;
  }
  function normalize(sequence,track,key){
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
      const folded=pitchInRange(note.pitch,track==="pad"?48:55,84);
      const pitch=track==="pad"?nearestPitch(folded,scaleFor(key),48,84):folded;
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
  function instrumentNotes(sequence,track,key={root:0,mode:"minor"}){
    let notes=normalize(sequence,track,key);
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
  root.NeuralPatterns={instrumentNotes,drumNotes,polyphony,validKey,scaleFor,keyName,randomKey,chordFor,harmonizePluck,harmonyAt};
})(typeof self!=="undefined"?self:globalThis);
