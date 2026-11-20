/* Bounded lookahead and cancellable worker lifecycle. No audio in this module. */
(function(root){
"use strict";
class NanoFlow{
  constructor({createWorker=()=>new Worker("nano-worker.js"),onStatus=()=>{},onError=()=>{}}={}){
    Object.assign(this,{createWorker,onStatus,onError,enabled:false,ready:false,queue:[],worker:null,job:null,epoch:0,serial:0,totalTokens:0});
  }
  stop(){
    this.enabled=false;this.ready=false;this.epoch++;this.queue=[];this.totalTokens=0;
    const job=this.job;this.job=null;
    if(job){clearTimeout(job.timer);job.reject(Error("FLOW cancelled"));}
    this.worker?.terminate();this.worker=null;
  }
  request(command,bpm){
    if(this.job)return Promise.reject(Error("FLOW request already running"));
    return new Promise((resolve,reject)=>{
      const id=++this.serial;
      const timer=setTimeout(()=>{
        if(this.job?.id!==id)return;
        this.job=null;reject(Error("Generation timed out; turn FLOW on to retry"));
      },180000);
      this.job={id,resolve,reject,timer};
      try{this.worker.postMessage({id,command,bpm});}
      catch(error){clearTimeout(timer);this.job=null;reject(error);}
    });
  }
  async start(bpm){
    this.stop();this.enabled=true;
    const epoch=this.epoch;
    try{
      this.worker=this.createWorker();
      this.worker.onmessage=({data})=>{
        if(epoch!==this.epoch||!this.job||data.id!==this.job.id)return;
        if(data.status){this.onStatus(data.status);return;}
        const job=this.job;this.job=null;clearTimeout(job.timer);
        if(data.error)job.reject(Error(data.error));
        else if(!Array.isArray(data.bars)||data.bars.length!==2||!data.bars.every(NanoMusic.validBar)||
                !data.bars.some(bar=>bar.length))job.reject(Error("Invalid FLOW phrase"));
        else{this.totalTokens=data.totalTokens;job.resolve(data.bars);}
      };
      this.worker.onerror=()=>{
        if(epoch!==this.epoch)return;
        const job=this.job;this.job=null;
        const error=Error("Model worker failed; turn FLOW on to retry");
        if(job){clearTimeout(job.timer);job.reject(error);}
        else this.fail(error);
      };
      const bars=await this.request("start",bpm);
      if(epoch!==this.epoch)return false;
      this.queue.push(...bars);this.ready=true;this.refill();return true;
    }catch(error){
      if(epoch!==this.epoch)return false;
      this.stop();throw error;
    }
  }
  fail(error){this.stop();this.onError(error);}
  refill(){
    if(!this.enabled||!this.ready||this.job||this.queue.length>3)return;
    const epoch=this.epoch;
    this.request("more").then(bars=>{
      if(epoch!==this.epoch)return;
      this.queue.push(...bars);
    }).catch(error=>{if(epoch===this.epoch)this.fail(error);});
  }
  takeBar(){const bar=this.queue.shift()??null;this.refill();return bar;}
}
root.NanoFlow=NanoFlow;
})(globalThis);
