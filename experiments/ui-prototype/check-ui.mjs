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
assert.match(html, /GENERATE TRACK/);
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
assert.match(html, /state\.pluck\.notes=demoNotes\("pluck"\)/);
assert.match(html, /state\.pad\.notes=demoNotes\("pad"\)/);
assert.match(html, /for\(const track of \["pluck","pad"\]\)/);
assert.match(html, /function playInstrumentNote\(note,time,instrument\)/);
assert.match(html, /function generateInstrument\(instrument\)/);
assert.match(html, /function restoreDemo\(instrument\)/);
assert.match(html, /state\[instrument\]\.notes=demoNotes\(instrument\)/);
assert.match(html, /go\("pluck"\)/);
assert.match(html, /go\("pad"\)/);
assert.match(html, /stepGrid\("PLUCK"[^;]*,"pluck"\)/);
assert.match(html, /stepGrid\("PAD"[^;]*,"pad"\)/);
assert.match(html, /\{label:"PLUCK",sub:"Melody/);
assert.match(html, /\{label:"PAD",sub:"Chords/);
assert.match(html, /pluck:"PLUCK",pad:"PAD"/);
assert.match(html, /<span class="chip">PLUCK<\/span>/);
assert.match(html, /<span class="chip">PAD<\/span>/);
assert.doesNotMatch(html, /synth1|synth2|hitSynth|generateSynth|PRESET: PLUCK|PRESET: PAD|state\.preset|state\.synthLevel|label:"SYNTH ?[12]"/i);
const script = html.match(/<script>([\s\S]*?)<\/script>/);
assert.ok(script, "Inline Web Audio program not found");
new Script(script[1], { filename: "ui-prototype-inline.js" });
assert.equal((html.match(/<script>/g) || []).length, 1);
// A DOM-light runtime smoke test catches errors static matching cannot:
// both instruments must be independently generated and Pad demo must be chords.
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
runInNewContext(script[1] + String.raw`
if (!state.pluck || !state.pad) throw new Error("Missing named instruments");
if (state.pluck.notes.some(note => note.len === 8)) throw new Error("Wrong Pluck demo");
if (!state.pad.notes.every(note => note.len === 8)) throw new Error("Wrong Pad demo");
if (!choices.user().some(option => option.label === "PLUCK")) throw new Error("Pluck menu missing");
if (!choices.user().some(option => option.label === "PAD")) throw new Error("Pad menu missing");
const beforePluck = JSON.stringify(state.pluck.notes);
generateInstrument("pad");
if (JSON.stringify(state.pluck.notes) !== beforePluck) throw new Error("Pad generator changed Pluck");
restoreDemo("pad");
if (state.pad.notes.length !== 6 || !state.pad.notes.every(note => note.len === 8)) {
  throw new Error("Pad demo restore did not restore chords");
}
const beforePad = JSON.stringify(state.pad.notes);
generateInstrument("pluck");
if (JSON.stringify(state.pad.notes) !== beforePad) throw new Error("Pluck generator changed Pad");
restoreDemo("pluck");
if (state.pluck.notes.length !== 6 || state.pluck.notes.some(note => note.len === 8)) {
  throw new Error("Pluck demo restore did not restore melody");
}
// Menu movement, value edit mode and round-the-ring gestures are independent.
home();
wheelTurn(1);
if (state.page !== "settings" || state.focus !== 0) throw new Error("Home wheel navigation");
wheelTurn(2);
if (state.focus !== 2) throw new Error("Clockwise scroll should select FX");
wheelTurn(-1);
if (state.focus !== 1) throw new Error("Anticlockwise scroll should select AI");
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
if (state.playhead !== (stepBefore + 1) % 16) throw new Error("NEXT must advance step");
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
`, { document, window: {}, setTimeout() {}, clearInterval() {} });
console.log("OK: Y2 480×360, native click-wheel buttons/rotations, BPM edit mode, transport and independent Pluck/Pad.");
