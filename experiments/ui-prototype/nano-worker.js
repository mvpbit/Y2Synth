"use strict";
importScripts("nano-common.js");
const REVISION="9a64dd24b3a550a6101f868af71bdc9960376e26";
const BASE="https://huggingface.co/utkucoban/NanoMaestro-Realtime/resolve/"+REVISION+"/NanoMaestro-Light/";
const RUNTIME="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
let session=null,vocab=null,h=null,c=null,current=null,decoder=null,busy=false,tempo=120;
let totalTokens=0,pieces=0;
async function checkedFile(name,hash){
  const response=await fetch(BASE+name,{signal:AbortSignal.timeout(90000)});
  if(!response.ok)throw Error("Model download failed ("+response.status+")");
  const bytes=await response.arrayBuffer();
  const digest=Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256",bytes)),x=>x.toString(16).padStart(2,"0")).join("");
  if(digest!==hash)throw Error("Model checksum mismatch");
  return bytes;
}
async function initialize(id){
  if(session)return;
  postMessage({id,status:"FLOW · loading model (14 MB)…"});
  importScripts(RUNTIME+"ort.wasm.min.js");
  ort.env.wasm.wasmPaths=RUNTIME;
  ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;
  const [model,words]=await Promise.all([
    checkedFile("model_int8.onnx","1d8f054bd04bf63d2d0514e6773d5a293c8034b7488684daeb44311efa96a87c"),
    checkedFile("vocab.json","51a3142aa391e9d588e866a87657a792e72f03a0cbc63ad5057760d38e7c67b3")
  ]);
  vocab=JSON.parse(new TextDecoder().decode(words));
  for(const token of ["BOS","BPM_120","GRID_64","BAR","POS_0"])
    if(!Number.isInteger(vocab.stoi[token]))throw Error("Unexpected model vocabulary");
  session=await ort.InferenceSession.create(model,{executionProviders:["wasm"],graphOptimizationLevel:"disabled"});
}
async function infer(token){
  const input=new ort.Tensor("int64",BigInt64Array.of(BigInt(token)),[1,1]);
  let result;
  try{
    result=await session.run({input,h,c});
    for(const name of ["logits","h_out","c_out"])
      if(!result[name]||!Array.from(result[name].data).every(Number.isFinite))throw Error("Non-finite model output");
    const logits=Float32Array.from(result.logits.data);
    const nh=result.h_out,nc=result.c_out;
    if(nh.data.length!==2048||nc.data.length!==2048)throw Error("Unexpected recurrent state");
    h.dispose();c.dispose();h=nh;c=nc;
    result.logits.dispose();result=null;
    totalTokens++;
    return logits;
  }finally{
    input.dispose();
    if(result)for(const tensor of Object.values(result))tensor.dispose();
  }
}
async function restart(){
  h?.dispose();c?.dispose();
  h=new ort.Tensor("float32",new Float32Array(2048),[2,1,1024]);
  c=new ort.Tensor("float32",new Float32Array(2048),[2,1,1024]);
  decoder=new NanoMusic.Decoder();pieces++;
  const tempos=Object.keys(vocab.stoi).filter(x=>/^BPM_\d+$/.test(x));
  tempos.sort((a,b)=>Math.abs(+a.slice(4)-tempo)-Math.abs(+b.slice(4)-tempo));
  const prompt=["BOS",tempos[0]||"BPM_120","GRID_64","BAR","POS_0"];
  for(const token of prompt.slice(0,-1))await infer(vocab.stoi[token]);
  current=vocab.stoi[prompt.at(-1)];
}
async function generate(request){
  const {id,command}=request;
  if(command!=="start"&&command!=="more")throw Error("Unknown FLOW command");
  if(command==="start"){
    tempo=Math.max(40,Math.min(240,Number(request.bpm)||120));
    await initialize(id);totalTokens=0;pieces=0;await restart();
  }else if(!session||!decoder)throw Error("FLOW is not initialized");
  postMessage({id,status:"FLOW · generating ahead…"});
  const bars=[];
  for(let count=0;count<2048&&bars.length<2;count++){
    const logits=await infer(current);
    const allowed=Object.entries(vocab.stoi).filter(([token])=>decoder.allows(token)).map(([,index])=>index);
    current=NanoMusic.sample(logits,allowed);
    const token=vocab.itos[current]??vocab.itos[String(current)];
    const complete=decoder.push(token);
    if(complete){bars.push(complete.bar);if(complete.end)await restart();}
  }
  if(bars.length!==2||!bars.every(NanoMusic.validBar))throw Error("Model could not complete a valid phrase");
  postMessage({id,bars,totalTokens,pieces});
}
self.onmessage=async({data})=>{
  if(busy){postMessage({id:data.id,error:"FLOW request already running"});return;}
  busy=true;
  try{await generate(data);}
  catch(error){postMessage({id:data.id,error:error?.message||String(error)});}
  finally{busy=false;}
};
