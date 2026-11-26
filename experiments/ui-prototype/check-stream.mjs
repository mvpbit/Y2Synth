import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
const context={};
for(const name of ["neural-common.js","ambient.js","ambient-validator.js"])
  runInNewContext(readFileSync(new URL(name,import.meta.url),"utf8"),context);
runInNewContext(`
  const stream=new AmbientStream({root:1,mode:"minor"});
  const pad=[61,64,68].map(pitch=>({step:0,len:16,pitch}));
  const seed=[{step:15,len:1,pitch:68,velocity:.7}];
  const recent=[];
  for(let i=0;i<2048;i++){
    if(!stream.harmonyFor)stream.setHarmony(pad);
    const bar=stream.next(seed);
    assert.ok(validAmbientBar(bar));
    const melody=bar.filter(n=>n.instrument==="pluck");
    const key=melody.map(n=>n.step+"/"+n.pitch).join(",");
    assert.ok(!recent.includes(key),"recent melody repeated");
    recent.push(key);if(recent.length>64)recent.shift();
    assert.equal(stream.primer().notes.length,melody.length);
    assert.ok(stream.history.length<=64);
  }
`,Object.assign(context,{assert}));
console.log("PASS: 2048 bars with a constant model fixture, including harmony changes.");
