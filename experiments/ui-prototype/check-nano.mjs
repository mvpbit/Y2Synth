import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext,Script} from "node:vm";
const sandbox={setTimeout,clearTimeout};
for(const file of ["nano-common.js","nano-flow.js"])
  runInNewContext(readFileSync(new URL(file,import.meta.url),"utf8"),sandbox);
new Script(readFileSync(new URL("nano-worker.js",import.meta.url),"utf8"));
const {Decoder,validBar,sample}=sandbox.NanoMusic;
const decode=new Decoder();
for(const token of ["NOTE_60","DUR_16","VEL_5","NOTE_67","DUR_8","VEL_7","POS_1","NOTE_72","DUR_256","VEL_0"])
  assert.equal(decode.push(token),null);
const first=decode.push("BAR");
assert.ok(validBar(first.bar));
assert.deepEqual(JSON.parse(JSON.stringify(first.bar)),[
  {step:0,pitch:67,len:2,velocity:1,instrument:"pluck"},
  {step:0,pitch:60,len:4,velocity:.75,instrument:"pad"},
  {step:.25,pitch:72,len:32,velocity:.125,instrument:"pluck"}]);
assert.equal(decode.phase,"position");
assert.equal(decode.allows("NOTE_60"),false);
decode.push("POS_63");decode.push("NOTE_21");decode.push("DUR_1");decode.push("VEL_7");
assert.equal(decode.allows("POS_62"),false);
assert.equal(decode.push("EOS").end,true);
assert.throws(()=>new Decoder().push("DUR_4"),/Invalid/);
assert.equal(validBar([{step:NaN}]),false);
assert.equal(validBar(Array(65).fill(first.bar[0])),false);
assert.equal(sample([0,1000,10],[0,2],()=>.9),2,"Sampling must respect the grammar mask");
assert.throws(()=>sample([NaN],[0]),/probabilities/);
let latest;
class WorkerStub{
  constructor(){latest=this;this.sent=[];this.terminated=false;}
  postMessage(data){this.sent.push(data);}
  terminate(){this.terminated=true;}
  answer(extra={}){
    this.onmessage({data:{id:this.sent.at(-1).id,bars:[first.bar,first.bar],totalTokens:100,...extra}});
  }
}
const errors=[];
const flow=new sandbox.NanoFlow({createWorker:()=>new WorkerStub(),onError:e=>errors.push(e.message)});
let started=flow.start(120);
latest.answer();
assert.equal(await started,true);
assert.equal(flow.queue.length,2);
assert.equal(latest.sent.at(-1).command,"more");
latest.answer();await Promise.resolve();
assert.equal(flow.queue.length,4);
flow.refill();
assert.equal(latest.sent.length,2,"Lookahead cannot grow without consumption");
assert.ok(flow.takeBar());assert.equal(latest.sent.length,3);
latest.answer();await Promise.resolve();
assert.equal(flow.queue.length,5);
assert.ok(flow.takeBar());assert.ok(flow.takeBar());
const cancelled=latest;
flow.stop();cancelled.answer();await Promise.resolve();
assert.equal(flow.queue.length,0);assert.equal(flow.enabled,false);
assert.equal(cancelled.terminated,true);
started=flow.start(100);
const loading=latest;
flow.stop();loading.answer();
assert.equal(await started,false,"Cancelled load must not reactivate FLOW");
started=flow.start(120);latest.answer({bars:[[{step:99}],[]]});
await assert.rejects(started,/Invalid FLOW/);
assert.equal(flow.enabled,false);
started=flow.start(120);latest.answer();await started;
latest.answer({error:"offline"});await Promise.resolve();await Promise.resolve();
assert.equal(flow.enabled,false);assert.deepEqual(errors,["offline"]);
started=flow.start(120);latest.onerror();await assert.rejects(started,/worker failed/);
console.log("PASS: NanoMaestro decoding, timing, masking, bounded queue, cancellation, stale responses and errors.");
