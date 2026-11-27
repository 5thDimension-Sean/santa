// Run with: node benchmark.cjs [practice_puzzles.csv]
'use strict';
const fs = require('node:fs');
const { performance } = require('node:perf_hooks');

function fields(line) {
  const columns = [];
  let value = '', quoted = false;
  for (let i = 0; i < line.length; ++i) {
    const c = line[i];
    if (c === '"' && quoted && line[i + 1] === '"') { value += '"'; ++i; }
    else if (c === '"') quoted = !quoted;
    else if (c === ',' && !quoted) { columns.push(value); value = ''; }
    else value += c;
  }
  if (quoted) throw Error('Unterminated CSV field');
  return [...columns, value];
}

function readPuzzles(path) {
  const [header, ...rows] = fs.readFileSync(path, 'utf8').trim().split(/\r?\n/);
  const keys = fields(header);
  const integers = s => s.slice(1, -1).split(',').filter(v => v.trim()).map(BigInt);
  return rows.map(line => {
    const cells = fields(line);
    const row = Object.fromEntries(keys.map((k, i) => [k, cells[i]]));
    return { id: row.id, family: row.family, seeds: integers(row.starting_values),
      targets: integers(row.targets), limit: BigInt(row.value_limit), size: Number(row.registers) };
  });
}

function calculate(op, x, y) {
  switch (op) {
    case '+': return x + y;
    case '-': return x - y;
    case '*': return x * y;
    case '/': return x % y === 0n ? x / y : 0n;
    default: throw Error('Unknown operator');
  }
}

// Trace dependencies using instruction identities so overwrites are harmless.
function prune(p, code) {
  const writer = Array(p.size).fill(-1), inputs = [], values = [...p.seeds];
  const todo = new Set(p.targets), roots = [];
  for (const v of p.seeds) todo.delete(v);
  for (const [index, [op, a, b, d]] of code.entries()) {
    inputs.push([writer[a], writer[b]]);
    values[d] = calculate(op, values[a], values[b]);
    writer[d] = index;
    if (todo.delete(values[d])) roots.push(index);
  }
  const live = new Set();
  while (roots.length) {
    const i = roots.pop();
    if (i < 0 || live.has(i)) continue;
    live.add(i);
    roots.push(...inputs[i]);
  }
  return code.filter((_, i) => live.has(i));
}

function validate(p, code) {
  const regs = [...p.seeds], seen = new Set(regs);
  if (!Array.isArray(code) || !code.every(i => Array.isArray(i) && i.length === 4)) throw Error('Invalid instructions');
  if (code.length > 4096 || Buffer.byteLength(JSON.stringify(code)) > 131072) throw Error('Size limit');
  for (const [op, a, b, d] of code) {
    if (![a,b,d].every(i => Number.isInteger(i) && i >= 0 && i < p.size)
        || regs[a] === undefined || regs[b] === undefined) throw Error('Bad register in ' + p.id);
    const v = calculate(op, regs[a], regs[b]);
    if (v <= 0n || v > p.limit) throw Error('Illegal arithmetic in ' + p.id);
    regs[d] = v;
    seen.add(v);
  }
  if (!p.targets.every(t => seen.has(t))) throw Error('Missing target in ' + p.id);
}

// Each index contains recipes with both operands already in registers.
function oneStep(values, limit) {
  const index = new Map();
  function add(v, op, a, b) {
    if (v > 0n && v <= limit && !values.has(v) && !index.has(v)) index.set(v, [op, a, b]);
  }
  const sorted = [...values.keys()].sort((a,b) => a < b ? -1 : a > b ? 1 : 0);
  for (let i = 0; i < sorted.length; ++i) {
    const a = sorted[i];
    for (let j = 0; j <= i; ++j) {
      const b = sorted[j];
      add(a+b, '+', a, b);
      add(a*b, '*', a, b);
      add(a-b, '-', a, b);
      if (a%b === 0n) add(a/b, '/', a, b);
    }
  }
  return index;
}

// Walk down a signed-radix expansion. At each level try a shorter prefix
// using one indexed operation followed by an optional small adjustment.
function recipe(target, base, fixed, available, index, limit) {
  let n = target, best = null;
  const tail = [];
  function tryPrefix(v, last) {
    const first = available.has(v) ? [] : index.has(v) ? [index.get(v)] : null;
    if (!first) return;
    const length = first.length + Number(Boolean(last)) + tail.length;
    if (!best || length < best.length)
      best = [...first, ...(last ? [last] : []), ...tail.slice().reverse()];
  }
  while (!best || tail.length < best.length) {
    tryPrefix(n, null);
    if (available.has(n)) break;
    if (!best || tail.length + 1 < best.length) {
      for (const k of fixed.keys()) {
        if (k < n) tryPrefix(n-k, ['+', n-k, k]);
        if (n+k <= limit) tryPrefix(n+k, ['-', n+k, k]);
        if (k > 1n && n%k === 0n) tryPrefix(n/k, ['*', n/k, k]);
      }
    }
    let remainder = n%base;
    if (remainder > base/2n && n+base-remainder <= limit) remainder -= base;
    const multiple = n-remainder;
    if (remainder > 0n) tail.push(['+', multiple, remainder]);
    if (remainder < 0n) tail.push(['-', multiple, -remainder]);
    tail.push(['*', multiple/base, base]);
    n = multiple/base;
  }
  return best;
}

function solve(p, base, reuse) {
  const regs = [...p.seeds], fixed = new Map(), code = [];
  p.seeds.forEach((v,i) => fixed.set(v,i));
  for (let v = 1n; v <= base; ++v) {
    if (fixed.has(v)) continue;
    const d = regs.length;
    code.push(v === 1n ? ['/',0,0,d] : ['+',fixed.get(v-1n),fixed.get(1n),d]);
    regs.push(v);
    fixed.set(v,d);
  }
  if (regs.length >= p.size) throw Error('Insufficient scratch space');
  const fixedCount = regs.length, targets = new Set(p.targets), covered = new Set(regs);
  const fixedIndex = oneStep(fixed, p.limit);
  const written = Array(p.size).fill(-1);
  for (const target of p.targets) {
    if (covered.has(target)) continue;
    const available = new Map(fixed);
    if (reuse) for (let i = fixedCount; i < regs.length; ++i) {
      if (!fixed.has(regs[i])) available.set(regs[i],i);
    }
    const index = reuse && available.size > fixed.size ? oneStep(available,p.limit) : fixedIndex;
    const steps = recipe(target,base,fixed,available,index,p.limit);
    for (const [op,x,y] of steps) {
      const a = available.get(x), b = available.get(y);
      const v = calculate(op,x,y);
      let d = regs.length;
      if (d === p.size) {
        // Preserve completed targets for later use; break ties by write age.
        d = fixedCount;
        for (let i = fixedCount+1; i < p.size; ++i) {
          const priority = Number(targets.has(regs[i])) - Number(targets.has(regs[d]));
          if (priority < 0 || (priority === 0 && written[i] < written[d])) d = i;
        }
        if (available.get(regs[d]) === d) available.delete(regs[d]);
      }
      code.push([op,a,b,d]);
      regs[d] = v;
      written[d] = code.length;
      if (!fixed.has(v)) available.set(v,d);
      covered.add(v);
    }
  }
  const result = prune(p,code);
  validate(p,result);
  return result;
}

function portfolio(p,reuse) {
  let best;
  const bases = new Set([2n,4n,8n,16n,24n,32n].map(v => v < p.limit ? v : p.limit));
  for (const base of bases) {
    const code = solve(p,base,reuse);
    if (!best || code.length < best.length) best = code;
  }
  return best;
}

function benchmark(file) {
  const puzzles = readPuzzles(file);
  const results = [], programs = ['id,program'];
  for (const [i,p] of puzzles.entries()) {
    const codes = [], times = [];
    for (const mode of [i%2, 1-i%2]) {
      const start = performance.now();
      codes[mode] = portfolio(p,Boolean(mode));
      times[mode] = performance.now()-start;
    }
    results.push([p.id,p.family,codes[0].length,codes[1].length,...times.map(v=>v.toFixed(2))]);
    const best = codes[0].length <= codes[1].length ? codes[0] : codes[1];
    programs.push(p.id + ',"' + JSON.stringify(best).replaceAll('"','""') + '"');
    if ((i+1)%25 === 0) console.log('Solved',i+1,'of',puzzles.length);
  }
  fs.writeFileSync('practice_scores.csv',['id,family,independent,reuse,independent_ms,reuse_ms',...results.map(r=>r.join(','))].join('\n')+'\n');
  fs.writeFileSync('practice_programs.csv',programs.join('\n')+'\n');
}

function verifyScored(puzzleFile, submission) {
  const puzzles = new Map(readPuzzles(puzzleFile).map(p => [p.id,p]));
  const data = new TextDecoder('utf-8',{fatal:true}).decode(fs.readFileSync(submission));
  const [header,...rows] = data.trimEnd().split(/\r?\n/);
  if (header !== 'id,program' || rows.length !== 1000 || puzzles.size !== 1000) throw Error('Invalid header or row count');
  const seen = new Set();
  let total = 0, longest = 0;
  for (const row of rows) {
    const [id,cell,...extra] = fields(row);
    if (extra.length || !puzzles.has(id) || seen.has(id) || Buffer.byteLength(cell)>131072) throw Error('Invalid row ' + id);
    const code = JSON.parse(cell);
    validate(puzzles.get(id),code);
    total += code.length;
    longest = Math.max(longest,code.length);
    seen.add(id);
  }
  for (let i=1; i<=1000; ++i) {
    if (!seen.has('scored_'+String(i).padStart(4,'0'))) throw Error('Missing scored ID '+i);
  }
  console.log(JSON.stringify({verified:seen.size, instructions:total, longest, savingsNeededToBeat124178:Math.max(0,total-124177)}));
}

if (process.argv[2] === '--verify') verifyScored(process.argv[3],process.argv[4]);
else benchmark(process.argv[2] || 'practice_puzzles.csv');
