// Remote CI only: use the official browser bundles as the numerical oracle.
import {createServer} from "node:http";
import {readFile,writeFile} from "node:fs/promises";
import {resolve,sep} from "node:path";
import {createRequire} from "node:module";
const require=createRequire(resolve(process.env.NEURAL_CHECK_TOOLS,"package.json"));
const {chromium}=require("playwright");
const root=resolve("experiments/drumsrnn/models");
const server=createServer(async(req,res)=>{
  try {
    if(req.url==="/"){res.setHeader("Content-Type","text/html");res.end("<!doctype html><title>Neural reference</title>");return;}
    const path=resolve(root,"."+decodeURIComponent(req.url.split("?")[0]));
    if(!path.startsWith(root+sep)){res.writeHead(403).end();return;}
    res.end(await readFile(path));
  } catch {res.writeHead(404).end();}
});
await new Promise(r=>server.listen(0,"127.0.0.1",r));
const browser=await chromium.launch({headless:true});
try {
 const page=await browser.newPage();
 page.on("console",msg=>console.log(msg.text()));
 await page.goto("http://127.0.0.1:"+server.address().port);
 for(const url of [
  "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.7.0/dist/tf.min.js",
  "https://cdn.jsdelivr.net/npm/@magenta/music@1.23.1/es6/core.js",
  "https://cdn.jsdelivr.net/npm/@magenta/music@1.23.1/es6/music_rnn.js",
  "https://cdn.jsdelivr.net/npm/@magenta/music@1.23.1/es6/music_vae.js",
 ])await page.addScriptTag({url});
 const reference=await page.evaluate(async()=>{
  await tf.setBackend("cpu");await tf.ready();
  const melody=new music_rnn.MusicRNN(location.origin+"/melody_rnn_tfjs");
  const result=await melody.continueSequenceAndReturnProbabilities({notes:[],quantizationInfo:{stepsPerQuarter:4},totalQuantizedSteps:1},16);
  const tokens=result.probs.map(p=>p.indexOf(Math.max(...p)));
  await result.sequence;melody.dispose();
  const pad=new music_vae.MusicVAE(location.origin+"/multitrack_chords_tfjs");
  await pad.initialize();
  const z=Array.from({length:256},(_,i)=>Math.fround(Math.sin(i*.37)*.75));
  const latent=tf.tensor2d([z]);
  const decoded=await pad.decodeTensors(latent,undefined,{chordProgression:["Cm"]});
  const labels=decoded.argMax(2);
  const output={melody:tokens,z,pad:Array.from(await labels.data())};
  labels.dispose();decoded.dispose();latent.dispose();pad.dispose();
  return output;
 });
 await writeFile(process.env.NEURAL_REFERENCE,JSON.stringify(reference));
 console.log("Official Magenta.js reference saved: 16 melody + 512 multitrack tokens");
}finally{await browser.close();await new Promise(r=>server.close(r));}
