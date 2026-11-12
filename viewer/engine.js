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

  function decodeF32(b64) {
    let bytes;
    if (typeof atob === 'function') {
      const s = atob(b64);
      bytes = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
    } else {
      bytes = new Uint8Array(Buffer.from(b64, 'base64'));
    }
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
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

  /* DQN: one-hot cells (16 values each) -> 128 -> 128 -> 4 Q-values. */
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
    const policy = function (board, moves) {
      const qv = q(board);
      return argmax(moves, (m) => qv[ACTIONS.indexOf(m.action)]);
    };
    policy.q = q;
    return policy;
  }

  const Engine = { ACTIONS, CHAR_TO_ACTION, slideLine, makeGame, rngFrom, decodeF32,
                   rules, tablePolicy, ntuplePolicy, dqnPolicy };
  if (typeof module !== 'undefined' && module.exports) module.exports = Engine;
  else root.Engine = Engine;
})(typeof window !== 'undefined' ? window : globalThis);
