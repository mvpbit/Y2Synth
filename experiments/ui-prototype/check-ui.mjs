// Static checks for the exact Innioasis Y2 framebuffer coordinate system.
// No npm dependencies, no browser automation and no audio playback needed.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Script, runInNewContext } from "node:vm";

const html = readFileSync(new URL("./index.html", import.meta.url), "utf8");
const screen = html.match(/\.screen\s*\{([^}]+)\}/);
assert.ok(screen, "Native screen CSS rule is missing");
for (const [name, value] of [
  ["width", "480px"],
  ["height", "360px"],
  ["min-width", "480px"],
  ["max-width", "480px"],
  ["min-height", "360px"],
  ["max-height", "360px"],
]) {
  assert.match(screen[1], new RegExp(`\\b${name}:${value.replace(".", "\\.")}(?:;|$)`),
    `Native screen must keep ${name}:${value}`);
}
const sizes = [
  { selector: ".screen-head", height: 36 },
  { selector: "#screen-main", height: 292 },
  { selector: ".screen-foot", height: 32 },
];
for (const { selector, height } of sizes) {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const css = html.match(new RegExp(`${escaped}\\s*\\{([^}]+)\\}`));
  assert.ok(css, `Missing section ${selector}`);
  assert.ok(css[1].includes(`height:${height}px`), `Wrong height for ${selector}`);
}
assert.equal(sizes.reduce((total, item) => total + item.height, 0), 360);
assert.match(html, /data-native-width="480" data-native-height="360"/);
assert.match(html, /AudioContext/);
assert.match(html, /GENERATE PLUCK/);
assert.match(html, /fx:\{delay:true/);
// Click wheel hardware geometry: all controls are integrated in one ring.
for (const id of ["wheel", "return", "prev", "next", "play", "center"]) {
  assert.match(html, new RegExp(`id="${id}"`), `Missing click-wheel control: ${id}`);
}
for (const id of ["back", "home", "up", "down", "left", "right"]) {
  assert.doesNotMatch(html, new RegExp(`id="${id}"`),
    `Old separate control must not return: ${id}`);
}
assert.match(html, /\.wheel \{width:340px;height:340px;/);
assert.match(html, /function startWheelGesture\(event\)/);
assert.match(html, /function moveWheelGesture\(event\)/);
assert.match(html, /function finishWheelGesture\(event\)/);
assert.match(html, /Math\.atan2\(\s*Math\.sin\(nextAngle-wheelMotion\.lastAngle\),/);
assert.match(html, /touch-action:none;user-select:none/);
assert.match(html, /wheelElement\.addEventListener\("wheel"/);
assert.match(html, /state\.editing=true/);
assert.match(html, /function wheelTurn\(steps\)/);
assert.match(html, /function stopTransport\(\)/);
assert.match(html, /longPressTimeout=setTimeout/);
// Instrument keys are stable, semantic and shared across state, routes,
// the sequencer, generators and the audio scheduler. Do not alias to synth1/2.
assert.match(html, /pluck:\{level:[^}]*notes:\[\]\}, pad:\{level:[^}]*notes:\[\]\}/);
assert.match(html, /for\(const track of \["pluck","pad"\]\)/);
assert.match(html, /function playInstrumentNote\(note,time,instrument\)/);
assert.match(html, /function generateInstrument\(instrument\)/);
assert.match(html, /go\("pluck"\)/);
assert.match(html, /go\("pad"\)/);
assert.match(html, /pluck:"PLUCK",pad:"PAD"/);
assert.match(html, /data-transport/);
assert.doesNotMatch(html, /demoNotes|restoreDemo|stepGrid|data-voice|data-step|label:"USER"|seekStep/);
assert.doesNotMatch(html, /synth1|synth2|hitSynth|generateSynth|PRESET: PLUCK|PRESET: PAD|state\.preset|state\.synthLevel|label:"SYNTH ?[12]"/i);
const script = html.match(/<script>([\s\S]*?)<\/script>/);
assert.ok(script, "Inline Web Audio program not found");
new Script(script[1], { filename: "ui-prototype-inline.js" });
assert.equal((html.match(/<script>/g) || []).length, 1);
// A DOM-light runtime smoke test catches errors static matching cannot:
// the home screen is transport-only and all instrument notes come from AI.
const elements = new Map();
const document = {
  getElementById(id) {
    if (!elements.has(id)) {
      elements.set(id, {
        textContent: "", innerHTML: "", onclick: null,
        addEventListener() {},
        querySelector() { return null; },
        getBoundingClientRect() { return {left:0, top:0, width:340, height:340}; },
        classList: { add() {}, remove() {} },
        setPointerCapture() {},
        hasPointerCapture() { return false; },
        releasePointerCapture() {},
      });
    }
    return elements.get(id);
  },
  addEventListener() {},
  querySelectorAll() { return []; },
};
await runInNewContext(["neural-common.js","nano-common.js","nano-flow.js"].map(file=>readFileSync(new URL("./"+file,import.meta.url),"utf8")).join("\n")+"\n(async()=>{\n" + script[1] + String.raw`
const assertRuntime=(ok,message)=>{if(!ok)throw Error(message);};
assertRuntime(!hasMusic(),"Fresh session must not contain demo notes");
await transport();
assertRuntime(!state.playing&&audio.ctx===null,"Empty PLAY must not start audio");
const homeMarkup=renderHome();
assertRuntime((homeMarkup.match(/<button/g)||[]).length===1&&homeMarkup.includes("data-transport"),"HOME must contain only the transport button");
assertRuntime(!homeMarkup.includes("data-step")&&!homeMarkup.includes("seq-row"),"HOME must not render a note grid");
assertRuntime(choices.settings().map(row=>row.label).join(",")==="GENERATE ALL,FLOW,DRUMS,PAD,PLUCK,FX,TEMPO","Settings must expose Generate All first and the five destinations");
assertRuntime(!choices.instruments,"No intermediate instrument submenu");
for(const instrument of ["pluck","pad","drums"]){
  assertRuntime(choices[instrument]().every(row=>row.label.startsWith("GENERATE")||row.label==="LEVEL"),"No manual notes/demo controls");
}
assertRuntime(choices.fx().map(row=>row.key).join(",")==="delay,delaySteps,delayFeedback,delayWet,reverb,reverbSize,reverbDamping,reverbWet","FX controls must stay intact");
activate();activate(4);
assertRuntime(state.page==="pluck","OK navigation must reach PLUCK");
back();assertRuntime(state.page==="settings","BACK must return directly to settings");
for(const [index,page] of ["drums","pad","pluck","fx","tempo"].entries()){
  activate(index+2);assertRuntime(state.page===page,"Direct settings destination: "+page);
  back();assertRuntime(state.page==="settings","Direct BACK from "+page);
}
home();
assertRuntime(await generateInstrument("pluck")===false&&!state.generating,"PLUCK requires PAD first");
const pluckBefore=JSON.stringify(state.pluck.notes),drumsBefore=JSON.stringify(state.drums);
const samplePad=[{step:0,pitch:60,len:16,velocity:.7},{step:0,pitch:63,len:16,velocity:.7},{step:0,pitch:67,len:16,velocity:.7}];
let pending=generateInstrument("pad");
assertRuntime(state.generating,"Loading must lock generation");
assertRuntime(await generateInstrument("pluck")===false,"Concurrent generation must be rejected");
neuralWorker.onmessage({data:{id:neuralJob.id,pattern:samplePad,model:"multitrack_chords",key:{root:0,mode:"minor"},chord:"Cm"}});
assertRuntime(await pending,"PAD result not committed");
assertRuntime(JSON.stringify(state.pluck.notes)===pluckBefore,"PAD changed PLUCK");
assertRuntime(JSON.stringify(state.drums)===drumsBefore,"PAD changed drums");
const padBefore=JSON.stringify(state.pad.notes);
pending=generateInstrument("pluck");
neuralWorker.onmessage({data:{id:neuralJob.id,error:"Network unavailable"}});
assertRuntime(await pending===false,"Error must fail generation");
assertRuntime(JSON.stringify(state.pluck.notes)===pluckBefore,"Error replaced old music");
pending=generateInstrument("pluck");
neuralWorker.onmessage({data:{id:neuralJob.id,pattern:[{step:17,pitch:60,len:1,velocity:.7}]}});
assertRuntime(await pending===false,"Invalid notes accepted");
pending=generateInstrument("pad");
++trackRevision.pad;
const savedPad=JSON.stringify(state.pad.notes);
neuralWorker.onmessage({data:{id:neuralJob.id,pattern:samplePad,model:"multitrack_chords",key:{root:0,mode:"minor"}}});
assertRuntime(await pending===false&&JSON.stringify(state.pad.notes)===savedPad,"Stale AI replaced the saved phrase");
state.playing=true;
pending=generateInstrument("pad");
neuralWorker.onmessage({data:{id:neuralJob.id,pattern:samplePad,model:"multitrack_chords",key:{root:0,mode:"minor"}}});
assertRuntime(await pending,"Queued generation failed");
assertRuntime(JSON.stringify(state.pad.notes)===savedPad,"Pattern changed before bar boundary");
applyPendingPatterns();
assertRuntime(JSON.stringify(state.pad.notes)===JSON.stringify(samplePad),"Queued pattern lost");
state.playing=false;
// Install a neural melody and verify it is harmonized, not used as fixed C minor.
const rawMelody=[{step:0,pitch:61,len:16,velocity:.8}];
pending=generateInstrument("pluck");
neuralWorker.onmessage({data:{id:neuralJob.id,pattern:rawMelody,model:"melody_rnn"}});
assertRuntime(await pending,"PLUCK generation failed");
assertRuntime(state.pluck.notes[0].pitch%12===0,"Strong PLUCK note must follow PAD chord");
const oldPluck=JSON.stringify(state.pluck.notes),oldPad=JSON.stringify(state.pad.notes);
state.playing=true;
pending=generateInstrument("pad");
const nextPad=[{step:0,pitch:62,len:8,velocity:.7},{step:0,pitch:66,len:8,velocity:.7},
 {step:8,pitch:69,len:8,velocity:.7},{step:8,pitch:73,len:8,velocity:.7}];
neuralWorker.onmessage({data:{id:neuralJob.id,pattern:nextPad,key:{root:2,mode:"major"},model:"multitrack_chords"}});
assertRuntime(await pending,"New PAD generation failed");
assertRuntime(JSON.stringify(state.pluck.notes)===oldPluck&&JSON.stringify(state.pad.notes)===oldPad&&state.padKey.root===0,"Harmony changed before bar boundary");
applyPendingPatterns();
assertRuntime(state.padKey.root===2&&state.padKey.mode==="major","New key was not installed");
assertRuntime(state.pluck.notes.length===2&&state.pluck.notes[0].pitch%12===2&&state.pluck.notes[1].pitch%12===1,"Held PLUCK must follow PAD chord changes");
assertRuntime(JSON.stringify(state.rawPluck)===JSON.stringify(rawMelody),"Retuning must preserve model melody for future PAD changes");
state.playing=false;
pending=generateDrums();
neuralWorker.onerror();
assertRuntime(await pending===false&&!state.generating&&neuralWorker===null,"Worker crash did not recover");
assertRuntime(JSON.stringify(state.drums)===drumsBefore,"Worker failure changed drums");

// Exercise real note scheduling with silent test doubles. Short Pad gates at
// high BPM must not put the sustain event before the end of the attack.
const envelopes=[];
audio.mix={};
audio.ctx={
  createOscillator(){return {type:"",frequency:{value:0,setValueAtTime(){}},
    connect(){return this;},start(){},stop(){}};},
  createGain(){
    const events=[];
    envelopes.push(events);
    const record=(value,time)=>events.push({value,time});
    return {gain:{setValueAtTime:record,linearRampToValueAtTime:record,
      exponentialRampToValueAtTime:record},connect(){return this;}};
  }
};
for(const bpm of [40,120,240]){
  state.bpm=bpm;
  for(const instrument of ["pluck","pad"]){
    for(const len of [1,2,8,16])playInstrumentNote({pitch:60,len,velocity:.7},1,instrument);
  }
}
for(const events of envelopes){
  if(events.some((event,i)=>!Number.isFinite(event.value)||event.value<=0||
      !Number.isFinite(event.time)||(i&&event.time<events[i-1].time))){
    throw new Error("Invalid audio envelope timing");
  }
}
for(const instrument of ["pluck","pad"]){
  envelopes.length=0;
  playInstrumentNote({pitch:60,len:2,velocity:1},1,instrument);
  const loud=Math.max(...envelopes[0].map(event=>event.value));
  envelopes.length=0;
  playInstrumentNote({pitch:60,len:2,velocity:.5},1,instrument);
  const soft=Math.max(...envelopes[0].map(event=>event.value));
  if(Math.abs(soft/loud-.5)>.00001)throw new Error("Velocity not applied: "+instrument);
}
audio.ctx=null;state.bpm=120;
// Drum level is a track gain before the common FX, including both snare layers.
const drumNotesBefore=JSON.stringify(state.drums),pluckLevelBefore=state.pluck.level,padLevelBefore=state.pad.level;
go("drums");activate(1);wheelTurn(-2);activate();
assertRuntime(Math.abs(state.drumsLevel-.9)<.00001,"Drum LEVEL must use wheel edit mode");
assertRuntime(JSON.stringify(state.drums)===drumNotesBefore&&state.pluck.level===pluckLevelBefore&&state.pad.level===padLevelBefore,"Drum LEVEL must not alter notes or other levels");
for(let i=0;i<25;i++)edit("drumsLevel",-1);
assertRuntime(state.drumsLevel===0,"Drums must reach mute");
const edges=[],gainChanges=[];
const parameter={value:0,setValueAtTime(){},exponentialRampToValueAtTime(){}};
function graphNode(){return {frequency:{...parameter},gain:{...parameter},connect(target){edges.push([this,target]);return target;},start(){},stop(){}};}
audio.mix=graphNode();
audio.drums=graphNode();
audio.drums.gain.setTargetAtTime=(value,time,ramp)=>gainChanges.push({value,time,ramp});
audio.ctx={currentTime:10,createOscillator:graphNode,createGain:graphNode,createBufferSource:graphNode,createBiquadFilter:graphNode};
updateDrumsLevel();
assertRuntime(gainChanges[0].value===0&&gainChanges[0].ramp>0,"Mute must reach the real drum gain with smoothing");
for(const voice of ["kick","snare","hat"])hitDrum(voice,10);
assertRuntime(edges.filter(([,target])=>target===audio.drums).length===4,"Kick, both snare layers and hat must route through drum LEVEL");
assertRuntime(!edges.some(([,target])=>target===audio.mix),"No drum voice may bypass its level");
audio.ctx=null;
for(let i=0;i<25;i++)edit("drumsLevel",1);
assertRuntime(state.drumsLevel===1,"Drum LEVEL must clamp to unity");
state.drumsLevel=.4;
pending=generateDrums();
neuralWorker.onmessage({data:{id:neuralJob.id,pattern:{kick:[0],snare:[4],hat:[2]},model:"drum_kit_rnn"}});
assertRuntime(await pending&&state.drumsLevel===.4,"Regeneration must preserve drum LEVEL");
audio.drums=null;state.drumsLevel=1;


// Menu movement, value edit mode and round-the-ring gestures are independent.
home();
wheelTurn(1);
if (state.page !== "settings" || state.focus !== 0) throw new Error("Home wheel navigation");
wheelTurn(5);
if (state.focus !== 5) throw new Error("Clockwise scroll should select FX");
wheelTurn(-1);
if (state.focus !== 4) throw new Error("Anticlockwise scroll should select PLUCK");
go("tempo");
if (state.editing) throw new Error("New page must exit edit mode");
activate();
if (!state.editing) throw new Error("OK must start editing the BPM");
const bpmBefore = state.bpm;
wheelTurn(2);
if (state.bpm !== bpmBefore + 4) throw new Error("Rotation must edit tempo");
activate();
if (state.editing) throw new Error("Second OK must finish numeric editing");
home();
const stepBefore = state.playhead;
leftRight(1);
if (state.playhead !== stepBefore || state.page !== "settings") throw new Error("NEXT must open settings without seeking notes");
// Test a clockwise quadrant drag through the exposed ring.
home();
const rect = {left:0,top:0,width:340,height:340};
if (!pointerOnRing({clientX:320,clientY:170},rect)) throw new Error("Ring zone missing");
if (pointerOnRing({clientX:170,clientY:170},rect)) throw new Error("OK zone included in ring");
const touch = {pointerId:7,clientX:320,clientY:170,preventDefault(){},
  target:{closest(){return null;}}};
startWheelGesture(touch);
if (!wheelMotion.active) throw new Error("Drag did not start");
moveWheelGesture({...touch,clientX:170,clientY:320});
if (state.page !== "settings" || state.focus !== 3) {
  throw new Error("Clockwise drag must scroll multiple menu steps");
}
finishWheelGesture(touch);
if (wheelMotion.active) throw new Error("Pointer release must end gesture");
state.playing=true;state.paused=true;state.playhead=9;audio.step=9;
stopTransport();
if (state.playing || state.paused || state.playhead !== 0 || audio.step !== 0) {
  throw new Error("Long-press STOP must reset playhead and transport");
}

// Generate All is one transaction, including at startup without an existing PAD.
state.pad.notes=[];state.pluck.notes=[];state.rawPluck=[];state.padKey=null;
state.drums={kick:[],snare:[],hat:[]};
const allPad=[{step:0,pitch:62,len:16,velocity:.7},{step:0,pitch:66,len:16,velocity:.7},{step:0,pitch:69,len:16,velocity:.7}];
const allMelody=[{step:0,pitch:61,len:16,velocity:.8}];
const allDrums={kick:[0,8],snare:[4,12],hat:[2,6,10,14]};
function musicSnapshot(){return JSON.stringify({pad:state.pad.notes,pluck:state.pluck.notes,raw:state.rawPluck,key:state.padKey,drums:state.drums});}
async function answerAll(track,pattern,key){
  assertRuntime(neuralWorker.lastRequest.track===track,"Generate All model order: "+track);
  neuralWorker.onmessage({data:{id:neuralJob.id,pattern,key,model:track}});
  await Promise.resolve();
}
let whole=generateAll();
assertRuntime(state.generating&&choices.settings()[0].disabled,"Generate All must lock the UI");
assertRuntime(await generateAll()===false,"Concurrent Generate All must be rejected");
assertRuntime(await generateDrums()===false,"Single-track generation must not interleave");
await answerAll("pad",allPad,{root:2,mode:"major"});
assertRuntime(!hasMusic(),"PAD must not publish a partial startup result");
await answerAll("pluck",allMelody);
assertRuntime(!hasMusic(),"PLUCK must not publish before DRUMS succeeds");
await answerAll("drums",allDrums);
assertRuntime(await whole&&!state.generating&&hasMusic(),"Generate All startup failed");
assertRuntime(state.padKey.root===2&&state.pluck.notes[0].pitch%12===2,"All PLUCK must follow the new PAD");
assertRuntime(!state.playing,"Generate All must not start transport");
const savedAll=musicSnapshot();
const savedLevels=JSON.stringify([state.drumsLevel,state.pluck.level,state.pad.level,state.fx]);
whole=generateAll();
await answerAll("pad",samplePad,{root:0,mode:"minor"});
await answerAll("pluck",allMelody);
neuralWorker.onmessage({data:{id:neuralJob.id,error:"Drum model unavailable"}});
assertRuntime(await whole===false&&musicSnapshot()===savedAll,"Last-stage failure must preserve the entire previous composition");
state.playing=true;
whole=generateAll();
await answerAll("pad",samplePad,{root:0,mode:"minor"});
await answerAll("pluck",allMelody);
await answerAll("drums",{kick:[1],snare:[5],hat:[3]});
assertRuntime(await whole&&musicSnapshot()===savedAll,"All tracks must wait for the bar boundary");
applyPendingPatterns();
assertRuntime(state.padKey.root===0&&state.pluck.notes[0].pitch%12===0&&state.drums.kick[0]===1,"All tracks must switch together");
assertRuntime(JSON.stringify([state.drumsLevel,state.pluck.level,state.pad.level,state.fx])===savedLevels,"Generate All must preserve levels and FX");
state.playing=false;
// FLOW scheduling preserves loop snapshots and waits at a bar boundary on underrun.
const savedFlowMusic=musicSnapshot(),savedFlowFx=JSON.stringify(state.fx);
const played=[];
const playSaved=playInstrumentNote,hitSaved=hitDrum;
playInstrumentNote=(note,time,instrument)=>played.push({note,time,instrument});
hitDrum=()=>{};
audio.ctx={currentTime:0};state.playing=true;state.bpm=120;
flow.enabled=true;flow.ready=true;flow.refill=()=>{};
flow.queue=[[{step:.25,pitch:64,len:4,velocity:.5,instrument:"pluck"}]];
flowBar=null;
assertRuntime(scheduleStep(0,1)===true&&played.length===1&&played[0].time===1.03125,"FLOW must preserve 1/64 timing");
assertRuntime(await generateAll()===false&&await generateDrums()===false,"FLOW must reject competing Magenta jobs");
for(let step=1;step<16;step++)scheduleStep(step,1+step/8);
assertRuntime(flowBar===null&&flow.queue.length===0,"FLOW must consume each bar only once");
assertRuntime(scheduleStep(0,3)===false&&played.length===1,"Empty FLOW queue must buffer, not replay old notes");
audio.step=0;audio.nextTime=-100;audio.ctx.currentTime=100;scheduler();
assertRuntime(audio.step===0&&audio.nextTime>100,"Underrun/background resume must not fast-forward through missing bars");
stopTransport();
assertRuntime(!flow.enabled&&flow.queue.length===0&&!state.playing,"STOP must cancel FLOW");
assertRuntime(musicSnapshot()===savedFlowMusic&&JSON.stringify(state.fx)===savedFlowFx,"FLOW must preserve saved loops and FX");
playInstrumentNote=playSaved;hitDrum=hitSaved;audio.ctx=null;
`+"\n})()", { document, window: {}, Worker: class { postMessage(message) {this.lastRequest=message;} terminate() {} }, setTimeout() {}, clearTimeout() {}, clearInterval() {} });

// Real adapters are tested with polyphonic output at the model's 24 steps/beat.
const sandbox={};
runInNewContext(readFileSync(new URL("./neural-common.js",import.meta.url),"utf8"),sandbox);
const {instrumentNotes,drumNotes,polyphony}=sandbox.NeuralPatterns;
const sequence={quantizationInfo:{stepsPerQuarter:24},notes:[
 {pitch:60,quantizedStartStep:0,quantizedEndStep:48,instrument:0,velocity:100},
 {pitch:64,quantizedStartStep:0,quantizedEndStep:48,instrument:0,velocity:90},
 {pitch:67,quantizedStartStep:0,quantizedEndStep:48,instrument:0,velocity:80},
 {pitch:72,quantizedStartStep:48,quantizedEndStep:96,instrument:0,velocity:100},
 {pitch:75,quantizedStartStep:48,quantizedEndStep:96,instrument:0,velocity:90},
 {pitch:36,quantizedStartStep:0,quantizedEndStep:1,instrument:1,isDrum:true}
]};
const pad=instrumentNotes(sequence,"pad");
assert.ok(polyphony(pad).maximum>=3);
assert.ok(pad.some(note=>note.step===8&&note.len===8),"PAD timing must convert 96 model steps to 16 sequencer steps");
assert.ok(pad.every(note=>[0,2,3,5,7,8,10].includes(note.pitch%12)));
assert.throws(()=>instrumentNotes({quantizationInfo:{stepsPerQuarter:4},notes:[]},"pluck"),/silence/);
assert.throws(()=>instrumentNotes({quantizationInfo:{stepsPerQuarter:4},notes:[
 {pitch:60,quantizedStartStep:0,quantizedEndStep:8}
]},"pad"),/polyphonic/);
assert.deepEqual(JSON.parse(JSON.stringify(drumNotes({quantizationInfo:{stepsPerQuarter:4},notes:[
 {pitch:36,quantizedStartStep:0},{pitch:38,quantizedStartStep:4},{pitch:42,quantizedStartStep:8}
]}))),{kick:[0],snare:[4],hat:[8]});
for(let root=0;root<12;root++)for(const mode of ["minor","major"]){
  const key={root,mode},scale=sandbox.NeuralPatterns.scaleFor(key);
  const chord=[0,mode==="minor"?3:4,7].map(offset=>({step:0,pitch:60+(root+offset)%12,len:16,velocity:.7}));
  const melody=sandbox.NeuralPatterns.harmonizePluck([{step:0,pitch:61,len:16,velocity:.8}],chord,key);
  assert.ok(melody.every(note=>scale.includes(note.pitch%12)));
  assert.ok(chord.some(note=>note.pitch%12===melody[0].pitch%12));
  const next=sandbox.NeuralPatterns.randomKey(key);
  assert.ok(next.root!==root||next.mode!==mode,"PAD key must change on regeneration");
}
new Script(readFileSync(new URL("./neural-worker.js",import.meta.url),"utf8"));
console.log("OK: display, wheel, transport, neural failure/queue isolation and model note adapters.");
