/* NanoMaestro event decoding. Model: Utku Coban, CC BY-NC 4.0.
   https://huggingface.co/utkucoban/NanoMaestro-Realtime */
(function(root){
"use strict";
function sample(logits,allowed,random=Math.random){
  const candidates=allowed.map(id=>({id,value:Number(logits[id])/.9}));
  if(!candidates.length||candidates.some(x=>!Number.isFinite(x.value)))throw Error("Invalid model probabilities");
  candidates.sort((a,b)=>b.value-a.value);
  const top=candidates.slice(0,32),weights=top.map(x=>Math.exp(x.value-top[0].value));
  let pick=random()*weights.reduce((a,b)=>a+b,0);
  for(let i=0;i<top.length;i++){pick-=weights[i];if(pick<=0)return top[i].id;}
  return top.at(-1).id;
}
function validBar(bar){
  return Array.isArray(bar)&&bar.length<=64&&bar.every(n=>
    Number.isFinite(n.step)&&n.step>=0&&n.step<16&&
    Number.isInteger(n.pitch)&&n.pitch>=21&&n.pitch<=108&&
    Number.isFinite(n.len)&&n.len>0&&n.len<=32&&
    Number.isFinite(n.velocity)&&n.velocity>0&&n.velocity<=1&&
    (n.instrument==="pluck"||n.instrument==="pad"));
}
class Decoder{
  constructor(){this.phase="event";this.position=0;this.notes=[];this.pitch=null;this.duration=null;this.seenNote=false;}
  allows(token){
    if(this.phase==="duration")return /^DUR_\d+$/.test(token)&&+token.slice(4)>0&&+token.slice(4)<=256;
    if(this.phase==="velocity")return /^VEL_[0-7]$/.test(token);
    if(this.phase==="position")return /^POS_\d+$/.test(token)&&+token.slice(4)<64;
    if(/^NOTE_\d+$/.test(token))return +token.slice(5)>=21&&+token.slice(5)<=108&&this.notes.length<64;
    if(/^POS_\d+$/.test(token))return +token.slice(4)>this.position&&+token.slice(4)<64;
    return (token==="BAR"||token==="EOS")&&this.seenNote;
  }
  finish(){
    // Same-onset chord: highest voice to PLUCK, up to four lower voices to PAD.
    // This is orchestration of model notes, not a second composition model.
    const groups=new Map();
    for(const n of this.notes){if(!groups.has(n.step))groups.set(n.step,[]);groups.get(n.step).push(n);}
    const bar=[];
    for(const group of groups.values()){
      group.sort((a,b)=>b.pitch-a.pitch);
      const pitches=new Set();
      for(const n of group){if(pitches.has(n.pitch))continue;pitches.add(n.pitch);
        if(pitches.size<=5)bar.push({...n,instrument:pitches.size===1?"pluck":"pad"});
      }
    }
    return bar;
  }
  push(token){
    if(!this.allows(token))throw Error("Invalid NanoMaestro event: "+token);
    if(token==="BAR"||token==="EOS"){
      const bar=this.finish();this.notes=[];this.seenNote=false;this.position=0;this.phase="position";
      return {bar,end:token==="EOS"};
    }
    if(token.startsWith("POS_")){this.position=+token.slice(4);this.phase="event";}
    else if(token.startsWith("NOTE_")){this.pitch=+token.slice(5);this.phase="duration";}
    else if(token.startsWith("DUR_")){this.duration=+token.slice(4);this.phase="velocity";}
    else if(token.startsWith("VEL_")){
      this.notes.push({step:this.position/4,pitch:this.pitch,len:Math.min(32,this.duration/4),velocity:(+token.slice(4)+1)/8});
      this.seenNote=true;this.phase="event";
    }
    return null;
  }
}
root.NanoMusic={sample,validBar,Decoder};
})(globalThis);
