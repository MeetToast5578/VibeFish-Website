// VibeFish board encoding and move choice, ported from the chessbot repo (model.py, models/v1/model.py).
// Works on a chess.js game. Squares use python-chess numbering: a1 = 0, b1 = 1, ..., h8 = 63.

const TYPES = 'pnbrqk';                       // piece type index + 1 = python-chess type (pawn 1 .. king 6)
const sqName = s => 'abcdefgh'[s & 7] + ((s >> 3) + 1);
const sqIndex = n => n.charCodeAt(0) - 97 + (n[1] - 1) * 8;
const on = (f, r) => f >= 0 && f < 8 && r >= 0 && r < 8;

// 64-array of {t: 1-6, c: 'w'|'b'} or null
function squares(game) {
  const b = Array(64).fill(null);
  game.board().forEach((row, i) => row.forEach((p, f) => {
    if (p) b[(7 - i) * 8 + f] = { t: TYPES.indexOf(p.type) + 1, c: p.color };
  }));
  return b;
}

const ROOK_DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1]];
const BISHOP_DIRS = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
const KNIGHT = [[1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2]];
const KING = [...ROOK_DIRS, ...BISHOP_DIRS];

// piece types of color c that attack sq (python-chess attackers_mask, as a list of types)
function attackers(b, c, sq) {
  const f = sq & 7, r = sq >> 3, out = [];
  const at = (ff, rr) => b[rr * 8 + ff];
  const pr = c === 'w' ? r - 1 : r + 1;       // a white pawn attacks upwards, so it sits one rank below
  for (const df of [-1, 1]) {
    const p = on(f + df, pr) && at(f + df, pr);
    if (p && p.c === c && p.t === 1) out.push(1);
  }
  for (const [df, dr] of KNIGHT) {
    const p = on(f + df, r + dr) && at(f + df, r + dr);
    if (p && p.c === c && p.t === 2) out.push(2);
  }
  for (const [dirs, slider] of [[BISHOP_DIRS, 3], [ROOK_DIRS, 4]]) {
    for (const [df, dr] of dirs) {
      for (let ff = f + df, rr = r + dr; on(ff, rr); ff += df, rr += dr) {
        const p = at(ff, rr);
        if (!p) continue;
        if (p.c === c && (p.t === slider || p.t === 5)) out.push(p.t);
        break;
      }
    }
  }
  for (const [df, dr] of KING) {
    const p = on(f + df, r + dr) && at(f + df, r + dr);
    if (p && p.c === c && p.t === 6) out.push(6);
  }
  return out;
}

// python-chess is_pinned: the piece on sq is pinned to its own king by an enemy slider
function pinned(b, sq) {
  const c = b[sq].c, k = b.findIndex(p => p && p.c === c && p.t === 6);
  if (k < 0 || k === sq) return false;
  const df = (sq & 7) - (k & 7), dr = (sq >> 3) - (k >> 3);
  if (df && dr && Math.abs(df) !== Math.abs(dr)) return false;
  const sf = Math.sign(df), sr = Math.sign(dr), slider = sf && sr ? 3 : 4;
  let ff = (k & 7) + sf, rr = (k >> 3) + sr;
  for (; rr * 8 + ff !== sq; ff += sf, rr += sr) if (b[rr * 8 + ff]) return false;
  for (ff += sf, rr += sr; on(ff, rr); ff += sf, rr += sr) {
    const p = b[rr * 8 + ff];
    if (p) return p.c !== c && (p.t === slider || p.t === 5);
  }
  return false;
}

const legalEpFile = game => {
  const m = game.moves({ verbose: true }).find(m => m.flags.includes('e'));
  return m ? sqIndex(m.to) & 7 : 8;
};

function castling(game) {
  const rights = game.fen().split(' ')[2], us = game.turn();
  const has = (c, side) => rights.includes(c === 'w' ? side.toUpperCase() : side);
  const them = us === 'w' ? 'b' : 'w';
  return has(us, 'k') + 2 * has(us, 'q') + 4 * has(them, 'k') + 8 * has(them, 'q');
}

// how often the current position occurred (python-chess transposition key: pieces, turn, castling, legal ep)
function repetitions(game, Chess) {
  const key = g => g.fen().split(' ').slice(0, 3).join(' ') + ' ' + legalEpFile(g);
  const replay = new Chess(), target = key(game);
  let n = key(replay) === target ? 1 : 0;
  for (const m of game.history({ verbose: true })) {
    replay.move(m);
    if (key(replay) === target) n++;
  }
  return n;
}

// v2-v4 input: [68, 7] features (model.py encode)
export function encodeV2(game, Chess) {
  const us = game.turn(), them = us === 'w' ? 'b' : 'w', flip = us === 'w' ? 0 : 56;
  const b = squares(game), x = new Array(68 * 7).fill(0);
  for (let sq = 0; sq < 64; sq++) {
    const row = (sq ^ flip) * 7, p = b[sq];
    if (p) {
      x[row] = p.t + (p.c === us ? 0 : 6);
      x[row + 5] = +pinned(b, sq);
    }
    [us, them].forEach((c, j) => {
      const a = attackers(b, c, sq);
      if (a.length) {
        x[row + 1 + j] = Math.min(a.length, 3);
        x[row + 3 + j] = Math.min(...a);
      }
    });
  }
  const hist = game.history({ verbose: true });
  const mark = (m, from, to) => { x[(sqIndex(m.from) ^ flip) * 7 + 6] = from; x[(sqIndex(m.to) ^ flip) * 7 + 6] = to; };
  if (hist.length >= 2) mark(hist[hist.length - 2], 3, 4);
  if (hist.length) mark(hist[hist.length - 1], 1, 2);
  const rep = repetitions(game, Chess);
  x[64 * 7] = castling(game);
  x[65 * 7] = legalEpFile(game);
  x[66 * 7] = Math.floor(Math.min(+game.fen().split(' ')[4], 99) / 10);
  x[67 * 7] = rep >= 3 ? 2 : rep >= 2 ? 1 : 0;
  return x;
}

// v1 input: 66 tokens (models/v1/model.py encode)
export function encodeV1(game) {
  const us = game.turn(), them = us === 'w' ? 'b' : 'w', flip = us === 'w' ? 0 : 56;
  const b = squares(game), toks = new Array(66).fill(0);
  for (let sq = 0; sq < 64; sq++) {
    const p = b[sq];
    const piece = p ? p.t + (p.c === us ? 0 : 6) : 0;
    const mine = Math.min(attackers(b, us, sq).length, 3), theirs = Math.min(attackers(b, them, sq).length, 3);
    toks[sq ^ flip] = piece * 16 + mine * 4 + theirs;
  }
  // python-chess keeps an ep square after every double pawn push, legal capture or not
  const last = game.history({ verbose: true }).at(-1);
  const ep = last && last.flags.includes('b') ? sqIndex(last.to) & 7 : 8;
  toks[64] = 208 + castling(game);
  toks[65] = 208 + 16 + ep;
  return toks;
}

// best legal move given the network's outputs. mix > 0 blends in the action-value head (v4, v4.1).
export function choose(game, policy, action, mix) {
  const flip = game.turn() === 'w' ? 0 : 56;
  const legal = [...new Set(game.moves({ verbose: true })
    .map(m => (sqIndex(m.from) ^ flip) * 64 + (sqIndex(m.to) ^ flip)))].sort((a, b) => a - b);
  const max = Math.max(...legal.map(i => policy[i]));
  const lse = max + Math.log(legal.reduce((s, i) => s + Math.exp(policy[i] - max), 0));
  const logsig = z => -Math.log1p(Math.exp(-z));
  let best = legal[0], bestScore = -Infinity;
  for (const i of legal) {
    const s = policy[i] - lse + (mix ? mix * logsig(action[i]) : 0);
    if (s > bestScore) { best = i; bestScore = s; }
  }
  const from = sqName(Math.floor(best / 64) ^ flip), to = sqName((best % 64) ^ flip);
  const piece = game.get(from);
  const promo = piece.type === 'p' && (to[1] === '1' || to[1] === '8');
  return { from, to, ...(promo && { promotion: 'q' }) };
}

// the published versions; numbers from the chessbot README
export const VERSIONS = {
  v1: { file: 'models/v1.onnx', mb: 1.8, mix: 0, about: '0.88M params · ~975 Elo' },
  'v1.1': { file: 'models/v1.1.onnx', mb: 1.8, mix: 0, about: '0.89M params · v1 retrained, beats v2 89%' },
  v2: { file: 'models/v2.onnx', mb: 5.7, mix: 0, about: '2.9M params · ~1,350–1,650 Elo' },
  v3: { file: 'models/v3.onnx', mb: 13.3, mix: 0, about: '6.7M params · ~1,980 Elo' },
  v4: { file: 'models/v4.onnx', mb: 27.2, mix: 2, about: '13.6M params · ~2,040–2,280 Elo' },
  'v4.1': { file: 'models/v4.1.onnx', mb: 27.2, mix: 2, about: '13.6M params · ~2,100 Elo, beats v4 57%' },
};

// one forward pass with an onnxruntime session -> move for chess.js
export async function think(ort, session, version, game, Chess) {
  const v2 = version !== 'v1';
  const data = v2 ? encodeV2(game, Chess) : encodeV1(game);
  const x = new ort.Tensor('int64', BigInt64Array.from(data, BigInt), v2 ? [1, 68, 7] : [1, 66]);
  const out = await session.run({ x });
  return choose(game, out.policy.data, out.action?.data, VERSIONS[version].mix);
}
