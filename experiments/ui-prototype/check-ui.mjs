// Static checks for the exact Innioasis Y2 framebuffer coordinate system.
// No npm dependencies, no browser automation and no audio playback needed.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Script } from "node:vm";

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
// Permanent instrument roles: Synth 1 = Pluck, Synth 2 = Pad.
// Each synth must have its own notes and level and be visible in HOME / USER / AI.
assert.match(html, /synth1:\{level:[^}]*notes:\[\]\}, synth2:\{level:[^}]*notes:\[\]\}/);
assert.match(html, /state\.synth1\.notes=demoNotes\("Pluck"\)/);
assert.match(html, /state\.synth2\.notes=demoNotes\("Pad"\)/);
assert.match(html, /for\(const track of \["synth1","synth2"\]\)/);
assert.match(html, /function hitSynth\(note,time,track\)/);
assert.match(html, /function generateSynth\(track\)/);
assert.match(html, /go\("synth1"\)/);
assert.match(html, /go\("synth2"\)/);
assert.match(html, /stepGrid\("PLUCK"/);
assert.match(html, /stepGrid\("PAD"/);
assert.match(html, /\{label:"PLUCK",sub:"Мелодия/);
assert.match(html, /\{label:"PAD",sub:"Аккорды/);
assert.match(html, /synth1:"PLUCK",synth2:"PAD"/);
assert.match(html, /<span class="chip">PLUCK<\/span>/);
assert.match(html, /<span class="chip">PAD<\/span>/);
assert.doesNotMatch(html, /PRESET: PLUCK|PRESET: PAD|state\.preset|state\.synthLevel/);
assert.doesNotMatch(html, /label:"SYNTH ?[12]"|stepGrid\("SYNTH[12]"|synth1:"SYNTH|synth2:"SYNTH|<span class="chip">S[12] \//);
const script = html.match(/<script>([\s\S]*?)<\/script>/);
assert.ok(script, "Inline Web Audio program not found");
new Script(script[1], { filename: "ui-prototype-inline.js" });
assert.equal((html.match(/<script>/g) || []).length, 1);
console.log("OK: Y2 480×360 layout, 36+292+32 bands, fixed Pluck/Pad tracks and JavaScript syntax.");
