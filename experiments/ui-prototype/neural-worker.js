"use strict";
// All inference runs outside the UI/audio scheduling thread. Official pinned
// Magenta implementations, with separate model objects for the three tracks.
importScripts("neural-common.js","ambient.js");
const CHECKPOINTS={
  drums:"music_rnn/drum_kit_rnn",
  pluck:"music_rnn/melody_rnn",
  pad:"music_vae/multitrack_chords",
};
const models={};
let runtimeReady=false;
async function initializeRuntime(){
  if(runtimeReady)return;
  importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.7.0/dist/tf.min.js");
  importScripts("https://cdn.jsdelivr.net/npm/@magenta/music@1.23.1/es6/core.js");
  importScripts("https://cdn.jsdelivr.net/npm/@magenta/music@1.23.1/es6/music_rnn.js");
  importScripts("https://cdn.jsdelivr.net/npm/@magenta/music@1.23.1/es6/music_vae.js");
  // Offscreen WebGL where supported; CPU remains a compatible fallback.
  try{await tf.setBackend("webgl");await tf.ready();}
  catch(error){await tf.setBackend("cpu");await tf.ready();}
  runtimeReady=true;
}
async function generate(request){
  const {id,track}=request;
  if(!Object.hasOwn(CHECKPOINTS,track))throw Error("Unknown track");
  postMessage({id,status:"Loading "+CHECKPOINTS[track].split("/")[1]+"…"});
  await initializeRuntime();
  if(!models[track]){
    const url="https://storage.googleapis.com/magentadata/js/checkpoints/"+CHECKPOINTS[track];
    const model=track==="pad"?new music_vae.MusicVAE(url):new music_rnn.MusicRNN(url);
    try{await model.initialize();models[track]=model;}
    catch(error){model.dispose();throw error;}
  }
  postMessage({id,status:"Generating "+track.toUpperCase()+"…"});
  const model=models[track];
  const key=track==="pad"?(request.flowKey || NeuralPatterns.randomKey(request.previousKey)):null;
  const chord=key?NeuralPatterns.chordFor(key):null;
  let lastError;
  for(let attempt=0;attempt<3;attempt++){
    try{
      let sequence;
      if(track==="pad"){
        [sequence]=await model.sample(1,.8,{chordProgression:[chord]},24);
      }else{
        // One internal silent bootstrap step, never a hand-written user phrase.
        const primer=request.primer || {notes:[],quantizationInfo:{stepsPerQuarter:4},totalQuantizedSteps:1};
        sequence=await model.continueSequence(primer,16,track==="drums"?1.2:1.05);
      }
      const pattern=track==="drums"?NeuralPatterns.drumNotes(sequence):
        NeuralPatterns.instrumentNotes(sequence,track,key||undefined);
      return {id,pattern,chord,key,model:CHECKPOINTS[track].split("/")[1]};
    }catch(error){lastError=error;}
  }
  throw lastError||Error("Generation failed");
}
let stream, barsGenerated, nextHarmony, nextTexture;
async function flowBatch(request) {
  const generateTrack = (track, extra={}) => generate({id:request.id,track,...extra});
  if (request.command === "start") {
    const pad = await generateTrack("pad");
    stream = new AmbientStream(pad.key);
    stream.setHarmony(pad.pattern);
    stream.hats = (await generateTrack("drums")).pattern.hat;
    barsGenerated = 0;
    nextHarmony = stream.harmonyFor;
    nextTexture = 5 + Math.floor(Math.random()*7);
  }
  const bars = [];
  while (bars.length < 2) {
    if (barsGenerated === nextHarmony) {
      stream.setHarmony((await generateTrack("pad",{flowKey:stream.key})).pattern);
      nextHarmony += stream.harmonyFor;
    }
    if (barsGenerated === nextTexture) {
      stream.hats = (await generateTrack("drums")).pattern.hat;
      nextTexture += 5 + Math.floor(Math.random()*7);
    }
    const melody = await generateTrack("pluck",{primer:stream.primer()});
    bars.push(stream.next(melody.pattern));
    barsGenerated++;
  }
  return {id:request.id,bars,barsGenerated};
}
self.onmessage = async ({data}) => {
  try { postMessage(await (data.command ? flowBatch(data) : generate(data))); }
  catch(error) { postMessage({id:data.id,error:error?.message || String(error)}); }
};
