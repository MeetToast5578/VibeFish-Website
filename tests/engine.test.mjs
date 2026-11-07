// Checks engine.js against positions recorded from the Python bot (tools/export.py writes the fixture).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Chess } from 'chess.js';
import * as ort from 'onnxruntime-node';
import { encodeV1, encodeV2, think, VERSIONS } from '../engine.js';

const fixture = JSON.parse(readFileSync(new URL('fixture.json', import.meta.url)));
const sessions = {};
for (const v in VERSIONS) sessions[v] = await ort.InferenceSession.create(new URL('../' + VERSIONS[v].file, import.meta.url).pathname);

for (const pos of fixture) {
  const game = new Chess();
  for (const m of pos.moves) game.move({ from: m.slice(0, 2), to: m.slice(2, 4), promotion: m[4] });
  const at = pos.moves.join(' ');
  assert.deepEqual(encodeV1(game), pos.enc1, 'v1 encoding after: ' + at);
  assert.deepEqual(encodeV2(game, Chess), pos.enc2, 'v2 encoding after: ' + at);
  for (const v in VERSIONS) {
    const m = await think(ort, sessions[v], v, game, Chess);
    assert.equal(m.from + m.to + (m.promotion ?? ''), pos.picks[v], `${v} move after: ${at}`);
  }
}
console.log(`ok: ${fixture.length} positions, encodings and moves match Python for ${Object.keys(VERSIONS).join(', ')}`);
