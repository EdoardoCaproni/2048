/* Rules of 2048 on an R x C board and the policies of the viewer.
 *
 * Same rules as game2048.py: a move slides every row (left/right) or
 * column (up/down) towards the side of the move; equal neighbours merge
 * once, starting from that side; a move is legal only if it changes the
 * board; the new tile is a 2 (probability 1 - p4) or a 4 in a uniformly
 * chosen empty cell. A board is an array of exponents in row-major order
 * (0 = empty, k = tile 2^k), and the action order down, left, right, up is
 * also the tie-breaking order, as in the Python code.
 * Works in the browser (window.Engine) and in Node (module.exports).
 */
(function (root) {
  'use strict';
  const ACTIONS = ['down', 'left', 'right', 'up'];
  const CHAR_TO_ACTION = { d: 'down', l: 'left', r: 'right', u: 'up' };

  function slideLine(vals) {
    const tiles = vals.filter((e) => e !== 0);
    const out = [];
    const mergedAt = [];
    let points = 0;
    for (let i = 0; i < tiles.length; i++) {
      if (i + 1 < tiles.length && tiles[i] === tiles[i + 1]) {
        out.push(tiles[i] + 1);
        points += 2 ** (tiles[i] + 1);
        mergedAt.push(out.length - 1);
        i++;
      } else {
        out.push(tiles[i]);
      }
    }
    while (out.length < vals.length) out.push(0);
    return { out, points, mergedAt };
  }

  function makeGame(rows, cols, target, p4) {
    const n = rows * cols;
    const targetExp = Math.round(Math.log2(target));
    const lines = { down: [], left: [], right: [], up: [] };
    for (let r = 0; r < rows; r++) {
      const row = [];
      for (let c = 0; c < cols; c++) row.push(r * cols + c);
      lines.left.push(row);
      lines.right.push(row.slice().reverse());
    }
    for (let c = 0; c < cols; c++) {
      const col = [];
      for (let r = 0; r < rows; r++) col.push(r * cols + c);
      lines.up.push(col);
      lines.down.push(col.slice().reverse());
    }

    /* Deterministic part of a move: {after, points, merged (cells), paths}.
       paths lists, for every tile, the cell it comes from and the cell it
       ends in (two tiles share the destination of a merge). */
    function slide(board, action) {
      const after = board.slice();
      let points = 0;
      const merged = [];
      const paths = [];
      for (const line of lines[action]) {
        const res = slideLine(line.map((i) => board[i]));
        points += res.points;
        line.forEach((cell, j) => { after[cell] = res.out[j]; });
        res.mergedAt.forEach((j) => merged.push(line[j]));
        const src = line.filter((cell) => board[cell] !== 0);
        for (let i = 0, k = 0; i < src.length; i++, k++) {
          if (i + 1 < src.length && board[src[i]] === board[src[i + 1]]) {
            paths.push({ from: src[i], to: line[k], merge: true });
            paths.push({ from: src[i + 1], to: line[k], merge: true });
            i++;
          } else {
            paths.push({ from: src[i], to: line[k], merge: false });
          }
        }
      }
      return { action, after, points, merged, paths };
    }

    /* Legal moves in ACTIONS order (empty at game over). */
    function moves(board) {
      const out = [];
      for (const a of ACTIONS) {
        const m = slide(board, a);
        if (m.after.some((e, i) => e !== board[i])) out.push(m);
      }
      return out;
    }

    function spawn(after, rng) {
      const empty = [];
      after.forEach((e, i) => { if (e === 0) empty.push(i); });
      const cell = empty[Math.floor(rng() * empty.length)];
      const board = after.slice();
      board[cell] = rng() < p4 ? 2 : 1;
      return { board, cell };
    }

    function initial(rng) {
      const i = Math.floor(rng() * n);
      let j = Math.floor(rng() * (n - 1));
      if (j >= i) j++;
      const board = new Array(n).fill(0);
      for (const cell of [i, j]) board[cell] = rng() < p4 ? 2 : 1;
      return board;
    }

    const maxExp = (board) => Math.max(...board);
    const isWin = (board) => maxExp(board) >= targetExp;
    const tileSum = (board) => board.reduce((s, e) => s + (e ? 2 ** e : 0), 0);
    const key = (board) => board.map((e) => e.toString(16)).join('');
    return { rows, cols, n, target, targetExp, p4, slide, moves, spawn, initial,
             maxExp, isWin, tileSum, key };
  }

  /* Seeded generator (mulberry32): the same seed gives the same game. */
  function rngFrom(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function decodeBytes(b64) {
    if (typeof atob === 'function') {
      const s = atob(b64);
      const bytes = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
      return bytes;
    }
    return new Uint8Array(Buffer.from(b64, 'base64'));  // a copy, aligned
  }

  function decodeF32(b64) {
    const bytes = decodeBytes(b64);
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  }

  /* IEEE half precision (numpy float16, little endian) to Float32Array. */
  function decodeF16(b64) {
    const bytes = decodeBytes(b64);
    const h = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    const scale = [];
    for (let e = 0; e < 32; e++) scale.push(2 ** (e - 25));
    const out = new Float32Array(h.length);
    for (let i = 0; i < h.length; i++) {
      const x = h[i];
      const e = (x >> 10) & 31;
      const f = x & 1023;
      let v;
      if (e === 0) v = f * 2 ** -24;                 // zero and subnormals
      else if (e === 31) v = f ? NaN : Infinity;
      else v = (1024 + f) * scale[e];
      out[i] = x & 0x8000 ? -v : v;
    }
    return out;
  }

  /* First move with the largest value (ties: ACTIONS order, as numpy argmax). */
  function argmax(moves, value) {
    let best = null;
    let bestV = -Infinity;
    for (const m of moves) {
      const v = value(m);
      if (v > bestV) { best = m; bestV = v; }
    }
    return { move: best, value: bestV };
  }

  /* Policies: (board, moves, rng) -> {move, value?}; moves is non-empty. */
  const rules = {
    corner: (board, moves) => ({ move: moves[0] }),
    greedy: (board, moves) => argmax(moves, (m) => m.points),
    random: (board, moves, rng) => ({ move: moves[Math.floor(rng() * moves.length)] }),
  };

  /* Table over the decision states of a small board (exported from Python). */
  function tablePolicy(game, keyIndex, actions) {
    return function (board, moves) {
      const i = keyIndex.get(game.key(board));
      const ch = i === undefined ? undefined : actions[i];
      const name = CHAR_TO_ACTION[ch];
      const move = moves.find((m) => m.action === name);
      return { move: move || moves[0], unseen: !move };
    };
  }

  /* Semi-gradient Sarsa with n-tuple features: q(x,u) = r(x,u) + V(M(x,u)). */
  function ntuplePolicy(nt) {
    const W = decodeF32(nt.W);
    const k = nt.k;
    const size = 16 ** k;
    const cells = nt.cells;
    const owner = nt.owner;
    function V(board) {
      let v = 0;
      for (let m = 0; m < cells.length; m++) {
        const cm = cells[m];
        let idx = 0;
        for (let j = 0; j < k; j++) idx += Math.min(board[cm[j]], 15) * 16 ** j;
        v += W[owner[m] * size + idx];
      }
      return v;
    }
    const policy = (board, moves) => argmax(moves, (m) => m.points + V(m.after));
    policy.V = V;
    return policy;
  }

  /* DQN with the perceptron of dqn2048.py (the tutorial's network): one-hot
     cells (16 values each) -> 128 -> 128 -> 4 Q-values. */
  function dqnPolicy(d) {
    const L = d.layers.map((l) => ({ w: decodeF32(l.w), b: decodeF32(l.b), in: l.in, out: l.out }));
    function q(board) {
      const [l1, l2, l3] = L;
      const h1 = new Float64Array(l1.out);
      for (let o = 0; o < l1.out; o++) {
        let s = l1.b[o];
        const base = o * l1.in;
        for (let c = 0; c < board.length; c++) s += l1.w[base + c * 16 + Math.min(board[c], 15)];
        h1[o] = s > 0 ? s : 0;
      }
      const h2 = new Float64Array(l2.out);
      for (let o = 0; o < l2.out; o++) {
        let s = l2.b[o];
        const base = o * l2.in;
        for (let i = 0; i < l2.in; i++) s += l2.w[base + i] * h1[i];
        h2[o] = s > 0 ? s : 0;
      }
      const out = new Float64Array(l3.out);
      for (let o = 0; o < l3.out; o++) {
        let s = l3.b[o];
        const base = o * l3.in;
        for (let i = 0; i < l3.in; i++) s += l3.w[base + i] * h2[i];
        out[o] = s;
      }
      return out;
    }
    return greedyOnQ(q);
  }

  /* Greedy policy on the legal moves of a Q-network; the choice keeps the
     four Q-values (field q), so the page does not compute them twice. */
  function greedyOnQ(q) {
    const policy = function (board, moves) {
      const qv = q(board);
      const choice = argmax(moves, (m) => qv[ACTIONS.indexOf(m.action)]);
      choice.q = qv;
      return choice;
    };
    policy.q = q;
    return policy;
  }

  /* DQN with the convolutional network PairCNN of dqn_gpu2048.py. The input
     is the one-hot board, 16 channels on the R x C grid; a first layer of
     1x2 (h) and 2x1 (v) convolutions reads the pairs of neighbouring cells,
     a second layer applies 1x2 and 2x1 convolutions to each of these maps
     (hh, hv, vh, vv: pairs of pairs); all the maps, after ReLU, are joined
     in a hidden layer (ReLU) that gives the four Q-values. The weights come
     in half precision with the shapes of PyTorch; the page computes in
     double precision. The maps are kept cell by cell (all the channels of a
     cell together, where PyTorch flattens channel by channel), and the
     hidden-layer weights are permuted to match; the zero inputs after a
     ReLU are skipped. */
  function cnnPolicy(d) {
    const T = {};
    for (const k of Object.keys(d.tensors)) {
      T[k] = { shape: d.tensors[k].shape, w: decodeF16(d.tensors[k].data) };
    }
    const R = d.rows;
    const C = d.cols;
    const [CH, NV] = T['first.h.weight'].shape;  // channels, values per cell
    const [HID, NIN] = T['hidden.weight'].shape;
    const TAPS = { h: [[0, 0], [0, 1]], v: [[0, 0], [1, 0]] };  // (dy, dx) of the two cells
    const dims = { h: [R, C - 1], v: [R - 1, C] };
    // weights [tap][input][output]: one contiguous vector of outputs per input
    const transpose = (w, nin) => {
      const t = new Float32Array(2 * nin * CH);
      for (let o = 0; o < CH; o++) {
        for (let i = 0; i < nin; i++) {
          for (let k = 0; k < 2; k++) t[(k * nin + i) * CH + o] = w[(o * nin + i) * 2 + k];
        }
      }
      return t;
    };
    const first = ['h', 'v'].map((name) => ({
      name, taps: TAPS[name], H: dims[name][0], W: dims[name][1],
      w: transpose(T[`first.${name}.weight`].w, NV), b: T[`first.${name}.bias`].w,
    }));
    const second = d.second.map((name) => {
      const [Hs, Ws] = dims[name[0]];
      const taps = TAPS[name[1]];
      const H = Hs - taps[1][0];
      const W = Ws - taps[1][1];
      dims[name] = [H, W];
      return { name, src: name[0], Ws, taps, H, W,
               w: transpose(T[`second.${name}.weight`].w, CH), b: T[`second.${name}.bias`].w };
    });
    // offsets of the maps in the feature vector, and the permuted hidden weights
    const offset = {};
    const Wh = T['hidden.weight'].w;
    const WH = new Float32Array(NIN * HID);
    let off = 0;
    for (const name of ['h', 'v', ...d.second]) {
      const npos = dims[name][0] * dims[name][1];
      offset[name] = off;
      for (let p = 0; p < npos; p++) {
        for (let o = 0; o < CH; o++) {
          const src = off + o * npos + p;   // PyTorch: channel by channel
          const dst = (off + p * CH + o) * HID;
          for (let j = 0; j < HID; j++) WH[dst + j] = Wh[j * NIN + src];
        }
      }
      off += npos * CH;
    }
    if (off !== NIN) throw new Error(`PairCNN: ${off} features, the hidden layer expects ${NIN}`);
    const bh = T['hidden.bias'].w;
    const Wo = T['out.weight'].w;
    const bo = T['out.bias'].w;
    const feats = new Float64Array(NIN);
    const acc = new Float64Array(CH);
    const hid = new Float64Array(HID);

    function q(board) {
      for (const L of first) {  // one-hot input: two weight vectors per output cell
        const base = offset[L.name];
        for (let r = 0; r < L.H; r++) {
          for (let c = 0; c < L.W; c++) {
            const e0 = Math.min(board[(r + L.taps[0][0]) * C + c + L.taps[0][1]], NV - 1);
            const e1 = Math.min(board[(r + L.taps[1][0]) * C + c + L.taps[1][1]], NV - 1);
            const w0 = e0 * CH;
            const w1 = (NV + e1) * CH;
            const out = base + (r * L.W + c) * CH;
            for (let o = 0; o < CH; o++) {
              const s = L.b[o] + L.w[w0 + o] + L.w[w1 + o];
              feats[out + o] = s > 0 ? s : 0;
            }
          }
        }
      }
      for (const L of second) {
        const src = offset[L.src];
        const base = offset[L.name];
        for (let r = 0; r < L.H; r++) {
          for (let c = 0; c < L.W; c++) {
            for (let o = 0; o < CH; o++) acc[o] = L.b[o];
            for (let k = 0; k < 2; k++) {
              const pin = src + ((r + L.taps[k][0]) * L.Ws + c + L.taps[k][1]) * CH;
              for (let i = 0; i < CH; i++) {
                const v = feats[pin + i];
                if (v === 0) continue;
                const wb = (k * CH + i) * CH;
                for (let o = 0; o < CH; o++) acc[o] += v * L.w[wb + o];
              }
            }
            const out = base + (r * L.W + c) * CH;
            for (let o = 0; o < CH; o++) feats[out + o] = acc[o] > 0 ? acc[o] : 0;
          }
        }
      }
      for (let j = 0; j < HID; j++) hid[j] = bh[j];
      for (let f = 0; f < NIN; f++) {
        const v = feats[f];
        if (v === 0) continue;
        const wb = f * HID;
        for (let j = 0; j < HID; j++) hid[j] += v * WH[wb + j];
      }
      const out = new Float64Array(4);
      for (let a = 0; a < 4; a++) {
        let s = bo[a];
        for (let j = 0; j < HID; j++) if (hid[j] > 0) s += hid[j] * Wo[a * HID + j];
        out[a] = s;
      }
      return out;
    }
    return greedyOnQ(q);
  }

  const Engine = { ACTIONS, CHAR_TO_ACTION, slideLine, makeGame, rngFrom, decodeF32, decodeF16,
                   rules, tablePolicy, ntuplePolicy, dqnPolicy, cnnPolicy };
  if (typeof module !== 'undefined' && module.exports) module.exports = Engine;
  else root.Engine = Engine;
})(typeof window !== 'undefined' ? window : globalThis);
