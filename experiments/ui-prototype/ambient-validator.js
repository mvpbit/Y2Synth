"use strict";
function validAmbientBar(bar) {
  if (!Array.isArray(bar) || !bar.length || bar.length > 64) return false;
  return bar.every(note => {
    if (!note || !Number.isFinite(note.step) || note.step < 0 || note.step >= 16 ||
        !Number.isFinite(note.velocity) || note.velocity <= 0 || note.velocity > 1) return false;
    if (note.instrument === "pulse" || note.instrument === "tick") return true;
    return ["pad","pluck"].includes(note.instrument) && Number.isInteger(note.pitch) &&
      note.pitch >= 21 && note.pitch <= 108 && Number.isFinite(note.len) && note.len > 0 && note.len <= 32;
  });
}
