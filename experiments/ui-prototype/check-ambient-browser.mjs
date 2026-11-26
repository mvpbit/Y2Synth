// Exercise Magenta inference and Web Audio in Chromium.
import assert from "node:assert/strict";
import http from "node:http";
import {readFile,mkdir} from "node:fs/promises";
import {chromium} from "playwright";
const root=new URL("./",import.meta.url);
const server=http.createServer(async(req,res)=>{
  try{
    const name=new URL(req.url,"http://localhost").pathname.slice(1)||"index.html";
    if(!/^[\w.-]+$/.test(name))throw Error("Bad path");
    const content=await readFile(new URL(name,root));
    res.setHeader("Content-Type",name.endsWith(".html")?"text/html":"text/javascript");
    res.end(content);
  }catch{res.writeHead(404);res.end();}
});
await new Promise(resolve=>server.listen(0,"127.0.0.1",resolve));
const browser=await chromium.launch({args:["--autoplay-policy=no-user-gesture-required"]});
const page=await browser.newPage({viewport:{width:900,height:1100}});
const messages=[];
page.on("pageerror",error=>messages.push(error.message));
page.on("console",message=>{if(message.type()==="error")console.log("BROWSER:",message.text());});
try{
  await page.goto("http://127.0.0.1:"+server.address().port);
  await page.locator("#center").click();
  const layout=await page.evaluate(()=>{
    const main=document.querySelector("#screen-main");
    return {height:main.clientHeight,scroll:main.scrollHeight,
      last:main.querySelector(".item:last-child").getBoundingClientRect().bottom,
      bottom:main.getBoundingClientRect().bottom};
  });
  assert.ok(layout.scroll<=layout.height&&layout.last<=layout.bottom,"All settings must fit without scrolling");
  const flowButton=page.getByRole("button",{name:/^FLOW /});
  await flowButton.click();
  await page.waitForFunction(()=>flowStarting||flow.enabled);
  await flowButton.click();
  assert.equal(await page.evaluate(()=>flowStarting||flow.enabled||state.playing),false,"Cancel startup");
  await flowButton.click();
  await page.waitForFunction(()=>flow.ready&&state.playing||(!flowStarting&&!flow.enabled&&state.message.startsWith("FLOW:")),{},{timeout:240000});
  const status=await page.evaluate(()=>({ready:flow.ready,playing:state.playing,message:state.message,barsGenerated:flow.barsGenerated}));
  console.log("START:",JSON.stringify(status));
  assert.ok(status.ready&&status.playing,status.message);
  await page.waitForFunction(()=>flow.barsGenerated>=4&&flow.queue.length>=2,{},{timeout:180000});
  const firstBars=await page.evaluate(()=>flow.barsGenerated);
  await page.waitForFunction(()=>{
    const samples=new Float32Array(audio.analyser.fftSize);
    audio.analyser.getFloatTimeDomainData(samples);
    return samples.some(x=>Math.abs(x)>0.00001);
  },{},{timeout:30000});
  await page.waitForFunction(previous=>flow.barsGenerated>previous,firstBars,{timeout:60000});
  // A fixed-duration sample detects lifecycle errors and tests ongoing replenishment.
  await page.waitForFunction(()=>flow.barsGenerated>=10,{}, {timeout:180000});
  const running=await page.evaluate(()=>({queue:flow.queue.length,playing:state.playing,enabled:flow.enabled,
    barsGenerated:flow.barsGenerated,voices:audio.voices.size,message:state.message}));
  console.log("CONTINUOUS:",JSON.stringify(running));
  assert.ok(running.enabled&&running.playing&&running.queue<=5&&running.voices<=256);
  await page.locator("#return").click();
  await page.locator("[data-transport]").click();
  assert.equal(await page.evaluate(()=>state.playing),false);
  assert.equal(await page.evaluate(()=>flow.enabled),true,"Pause keeps recurrent session");
  await page.locator("[data-transport]").click();
  assert.equal(await page.evaluate(()=>state.playing),true);
  await mkdir("nano-browser-results",{recursive:true});
  await page.screenshot({path:"nano-browser-results/flow-home.png"});
  await page.locator("#play").dispatchEvent("pointerdown",{pointerId:1});
  await page.waitForTimeout(850);
  await page.locator("#play").dispatchEvent("pointerup",{pointerId:1});
  // Check the actual STOP function as well: pointer behavior is covered by UI checks.
  await page.evaluate(()=>stopTransport());
  assert.deepEqual(await page.evaluate(()=>[flow.enabled,flow.queue.length,!!flow.job,state.playing,audio.voices.size]),[false,0,false,false,0]);
  const snapshot=await page.evaluate(()=>JSON.stringify([state.pluck,state.pad,state.drums,state.fx]));
  await page.route("https://storage.googleapis.com/magentadata/**",route=>route.abort());
  await page.locator("#center").click();
  await page.getByRole("button",{name:/^FLOW /}).click();
  await page.waitForFunction(()=>!flowStarting&&!flow.enabled,{},{timeout:180000});
  assert.match(await page.evaluate(()=>state.message),/^FLOW:/);
  assert.equal(await page.evaluate(()=>JSON.stringify([state.pluck,state.pad,state.drums,state.fx])),snapshot);
  assert.equal(await page.evaluate(()=>state.playing),false);
  assert.deepEqual(messages,[]);
  console.log("PASS: ambient section renewal with three real Magenta models, continuous refill, audio signal, layout, pause/resume, STOP, network failure.");
}finally{await browser.close();await new Promise(resolve=>server.close(resolve));}
