// Based on the suffix target rebuilder in Nilanjan_Saha123, "Best-of-Public Elf Optimizer" (Kaggle, Apache-2.0).
// Change: optional dense cut positions (argv[3] = step) and JOINT PAIRS (argv[4] = max pairs): remove two targets
// together (so helpers they share are dropped too) and rebuild both at the end, in both orders.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <set>
using namespace std;
using V = int64_t;

struct Op { char op; V a, b, out; };
struct Jump { V quotient, multiplier, remainder, product, number; };
struct State { V value; int parent; Jump jump; };
struct Catalog {
  vector<V> values, basis;
  unordered_map<V, Op> direct;
  Catalog(const array<V, 48>& reg, V limit) {
    for (V x : reg) if (x && !direct.count(x)) { values.push_back(x); direct[x] = {0, 0, 0, x}; }
    sort(values.begin(), values.end());
    for (V a : values)
      for (V b : values) {
        if (a < b) break;
        if (a + b <= limit) direct.emplace(a + b, Op{'+', a, b, a + b});
        if (a > b) direct.emplace(a - b, Op{'-', a, b, a - b});
        if (a <= limit / b) direct.emplace(a * b, Op{'*', a, b, a * b});
        if (a % b == 0) direct.emplace(a / b, Op{'/', a, b, a / b});
      }
    for (const auto& entry : direct) basis.push_back(entry.first);
    sort(basis.begin(), basis.end());
  }
};

vector<Op> reverse_search(V target, V limit, const Catalog& bank, int width, int budget) {
  vector<State> nodes{{target, -1, {}}}; vector<vector<int>> layers(80);
  int best = min(budget, 80), finish = -1; layers[0].push_back(0);
  unordered_map<V, int> visited{{target, 0}}; vector<Op> base;
  auto direct_op = [&](vector<Op>& ops, V x) {
    Op s = bank.direct.at(x); if (s.op) ops.push_back(s);
  };
  for (int cost = 0; cost < best && cost < 74; ++cost) {
    auto& layer = layers[cost];
    auto less = [&](int i, int j) { return nodes[i].value < nodes[j].value; };
    if ((int)layer.size() > width) {
      nth_element(layer.begin(), layer.begin() + width, layer.end(), less); layer.resize(width);
    }
    for (int id : layer) {
      V n = nodes[id].value; if (visited[n] < cost) continue;
      auto hit = bank.direct.find(n);
      if (hit != bank.direct.end()) {
        int total = cost + bool(hit->second.op);
        if (total < best) { best = total; finish = id; base.clear(); direct_op(base, n); }
        continue;
      }
      auto join = [&](char op, V a, V b) {
        auto ib = bank.direct.find(b); if (ib == bank.direct.end()) return;
        int total = cost + 1 + bool(bank.direct.at(a).op) + (a != b && bool(ib->second.op));
        if (total >= best) return;
        best = total; finish = id; base.clear();
        direct_op(base, a); if (a != b) direct_op(base, b);
        base.push_back({op, a, b, n});
      };
      auto push = [&](V q, V m, V r, int fee) {
        if (q <= 1 || q >= n || cost + fee >= best) return;
        int depth = cost + fee; auto [it, ok] = visited.emplace(q, depth);
        if (!ok && it->second <= depth) return;
        it->second = depth; auto end = bank.direct.find(q);
        if (end != bank.direct.end()) {
          if (depth + bool(end->second.op) >= best) return;
          best = depth + bool(end->second.op); finish = nodes.size();
          base.clear(); direct_op(base, q);
        } else layers[depth].push_back(nodes.size());
        nodes.push_back({q, id, {q, m, r, q * m, n}});
      };
      for (V m : bank.basis) {
        if (n > m) join('+', m, n - m);
        if (n + m <= limit) {
          auto it = bank.direct.find(n + m); if (it != bank.direct.end()) join('-', n + m, m);
        }
        if (m > n) join('-', m, m - n);
        if (m > n && m % n == 0) join('/', m, m / n);
        if (m <= 1 || m >= n) continue;
        V q = n / m, r = n % m; int fee = 1 + bool(bank.direct.at(m).op);
        if (!r) { join('*', m, q); push(q, m, 0, fee); } else {
          auto dig = [&](V quotient, V remainder) {
            auto it = bank.direct.find(abs(remainder));
            if (it != bank.direct.end()) push(quotient, m, remainder, fee + 1 + bool(it->second.op));
          };
          dig(q, r); if (m - r <= limit - n) dig(q + 1, r - m);
          if (q > 2) dig(q - 1, r + m);
        }
      }
    }
  }
  if (finish < 0) return {};
  for (int i = finish; nodes[i].parent != -1; i = nodes[i].parent) {
    Jump j = nodes[i].jump; direct_op(base, j.multiplier);
    base.push_back({'*', j.quotient, j.multiplier, j.product});
    if (j.remainder) {
      direct_op(base, abs(j.remainder));
      base.push_back({j.remainder > 0 ? '+' : '-', j.product, abs(j.remainder), j.number});
    }
  }
  return base;
}

struct Machine {
  V limit; array<V, 48> reg{}; array<int, 48> used{}; vector<array<int, 4>> program;
  unordered_set<V> covered, pinned;
  int find(V v) const { for (int i = 0; i < 48; ++i) if (reg[i] == v) return i; return -1; }
  bool run(const vector<Op>& ops) {
    for (int k = 0; k < (int)ops.size(); ++k) {
      Op s = ops[k]; if (find(s.out) >= 0) continue;
      int a = find(s.a), b = find(s.b); if (a < 0 || b < 0) return false;
      int dest = find(0);
      if (dest < 0) {
        unordered_set<V> keep = pinned;
        for (int j = k + 1; j < (int)ops.size(); ++j) {
          if (ops[j].a != s.out) keep.insert(ops[j].a);
          if (ops[j].b != s.out) keep.insert(ops[j].b);
        }
        int oldest = numeric_limits<int>::max();
        for (int i = 0; i < 48; ++i)
          if (!keep.count(reg[i]) && used[i] < oldest) { oldest = used[i]; dest = i; }
      }
      if (dest < 0) return false;
      program.push_back({s.op, a, b, dest}); covered.insert(s.out);
      reg[dest] = s.out; used[a] = used[b] = used[dest] = program.size();
    }
    return true;
  }
};

using Instruction = array<int, 4>;
V apply(int op, V a, V b) {
  if (op == '+') return a + b;
  if (op == '-') return a - b;
  if (op == '*') return a * b;
  return a / b;
}

vector<Instruction> retain(const vector<Instruction>& original, const vector<V>& output, const unordered_set<V>& keep, int old_position = -1, int* new_position = nullptr) {
  vector<char> first(original.size()); auto unseen = keep;
  for (int i = 0; i < (int)original.size(); ++i) first[i] = unseen.erase(output[i]);
  uint64_t live = 0; vector<Instruction> result;
  for (int i = (int)original.size() - 1; i >= 0; --i) {
    auto s = original[i]; uint64_t dest = uint64_t(1) << s[3];
    if (first[i] || (live & dest)) {
      live = (live & ~dest) | (uint64_t(1) << s[1]) | (uint64_t(1) << s[2]);
      result.push_back(s);
      if (new_position && i < old_position) ++*new_position;
    }
  }
  reverse(result.begin(), result.end()); return result;
}

Machine replay(const vector<V>& starts, const vector<Instruction>& program, V limit) {
  Machine m; m.limit = limit; copy(starts.begin(), starts.end(), m.reg.begin());
  m.covered.insert(starts.begin(), starts.end()); m.program = program;
  for (auto s : program) {
    V v = apply(s[0], m.reg[s[1]], m.reg[s[2]]); m.reg[s[3]] = v; m.covered.insert(v);
  }
  return m;
}

int STEP = 0, PAIRS = 0, GROUPS = 0;
// Insert target t into a valid program at the cheapest cut (every `step` positions), keeping every value the rest of
// the program still reads (pinned), as in the single-target rebuilder. Returns empty if nothing fits under `cap`.
vector<Instruction> insert_best(const vector<V>& starts, V limit, const vector<Instruction>& prog, V t, int width,
                                size_t cap, int step) {
  vector<Instruction> bestp; size_t bestsz = cap;
  vector<int> cuts; for (int c = 0; c <= (int)prog.size(); c += max(1, step)) cuts.push_back(c);
  if (cuts.back() != (int)prog.size()) cuts.push_back(prog.size());
  for (int cut : cuts) {
    if (prog.size() + 1 >= bestsz) break;
    vector<Instruction> head(prog.begin(), prog.begin() + cut);
    auto m = replay(starts, head, limit);
    if (m.covered.count(t)) { return prog; }
    uint64_t live = 0;
    for (int j = (int)prog.size() - 1; j >= cut; --j) {
      auto s = prog[j];
      live = (live & ~(uint64_t(1) << s[3])) | (uint64_t(1) << s[1]) | (uint64_t(1) << s[2]);
    }
    for (int i = 0; i < 48; ++i) if (live & (uint64_t(1) << i)) m.pinned.insert(m.reg[i]);
    int budget = (int)bestsz - (int)prog.size();
    if (budget <= 1) continue;
    Catalog bank(m.reg, limit);
    auto ops = reverse_search(t, limit, bank, width, budget);
    auto ops80 = getenv("DBG") ? reverse_search(t, limit, bank, width, 80) : vector<Op>{};
    if (getenv("DBG")) cerr << "    cut " << cut << " budget " << budget << " found " << ops.size() << " unbounded " << ops80.size() << "\n";
    if (ops.empty() || !m.run(ops)) continue;
    m.program.insert(m.program.end(), prog.begin() + cut, prog.end());
    if (m.program.size() < bestsz) { bestsz = m.program.size(); bestp = std::move(m.program); }
  }
  return bestp;
}
vector<Instruction> optimize(const vector<V>& starts, const vector<V>& targets, V limit, const vector<Instruction>& seed, int width, int attempts) {
  auto original = replay(starts, {}, limit); vector<V> output;
  for (auto s : seed) {
    V x = apply(s[0], original.reg[s[1]], original.reg[s[2]]);
    output.push_back(x); original.reg[s[3]] = x;
  }
  unordered_set<V> required(targets.begin(), targets.end());
  for (V s : starts) required.erase(s);
  auto best = retain(seed, output, required);
  struct Trial { V target; vector<Instruction> prefix; int boundary; };
  vector<Trial> trials;
  for (V t : targets) {
    if (!required.erase(t)) continue;
    int boundary = 0, pos = find(output.begin(), output.end(), t) - output.begin();
    auto prefix = retain(seed, output, required, pos, &boundary);
    required.insert(t);
    if (prefix.size() + 1 < best.size()) trials.push_back({t, std::move(prefix), boundary});
  }
  sort(trials.begin(), trials.end(), [](const Trial& a, const Trial& b) {
    return a.prefix.size() != b.prefix.size() ? a.prefix.size() < b.prefix.size() : a.target < b.target;
  });
  int tested = 0;
  for (auto& trial : trials) {
    int budget = best.size() - trial.prefix.size();
    if (budget <= 1 || tested++ >= attempts) break;
    auto complete = replay(starts, trial.prefix, limit);
    if (complete.covered.count(trial.target)) { best = trial.prefix; continue; }
    vector<int> positions{trial.boundary, (int)trial.prefix.size()};
    if (trial.boundary > 20) positions.push_back(trial.boundary - 12);
    if (trial.boundary + 16 < (int)trial.prefix.size()) positions.push_back(trial.boundary + 16);
    if (STEP > 0) for (int c = 0; c <= (int)trial.prefix.size(); c += STEP) positions.push_back(c);
    sort(positions.begin(), positions.end()); positions.erase(unique(positions.begin(), positions.end()), positions.end());
    for (int cut : positions) {
      budget = best.size() - trial.prefix.size();
      if (budget <= 1) break;
      vector<Instruction> head(trial.prefix.begin(), trial.prefix.begin() + cut);
      auto m = replay(starts, head, limit); uint64_t live = 0;
      for (int j = (int)trial.prefix.size() - 1; j >= cut; --j) {
        auto s = trial.prefix[j];
        live = (live & ~(uint64_t(1) << s[3])) | (uint64_t(1) << s[1]) | (uint64_t(1) << s[2]);
      }
      for (int i = 0; i < 48; ++i) if (live & (uint64_t(1) << i)) m.pinned.insert(m.reg[i]);
      Catalog bank(m.reg, limit);
      auto ops = reverse_search(trial.target, limit, bank, width, budget);
      if (ops.empty() || !m.run(ops)) continue;
      m.program.insert(m.program.end(), trial.prefix.begin() + cut, trial.prefix.end());
      if (m.program.size() < best.size()) best = std::move(m.program);
    }
  }
  // ---- joint pairs: drop two targets (and everything only they need), rebuild both at the end
  if (PAIRS > 0) {
    vector<V> tg;
    for (V t : targets) if (required.count(t)) tg.push_back(t);
    int tried = 0;
    for (size_t x = 0; x < tg.size() && tried < PAIRS; ++x)
      for (size_t y = x + 1; y < tg.size() && tried < PAIRS; ++y) {
        // recompute outputs of the current best
        auto base = replay(starts, {}, limit); vector<V> out2;
        for (auto s2 : best) { V v = apply(s2[0], base.reg[s2[1]], base.reg[s2[2]]); out2.push_back(v); base.reg[s2[3]] = v; }
        unordered_set<V> req2 = required; req2.erase(tg[x]); req2.erase(tg[y]);
        auto prefix = retain(best, out2, req2);
        if (prefix.size() + 2 >= best.size()) continue;
        ++tried;
        for (int order = 0; order < 2; ++order) {
          V first = order ? tg[y] : tg[x], second = order ? tg[x] : tg[y];
          auto m = replay(starts, prefix, limit);
          bool ok = true;
          for (V t : {first, second}) {
            if (m.covered.count(t)) continue;
            int budget = (int)best.size() - (int)m.program.size();
            if (budget <= 1) { ok = false; break; }
            Catalog bank(m.reg, limit);
            auto ops = reverse_search(t, limit, bank, width, budget);
            if (ops.empty() || !m.run(ops)) { ok = false; break; }
          }
          if (ok && m.program.size() < best.size()) best = std::move(m.program);
        }
      }
  }
  // ---- helper groups: for every computed value, the set of targets whose recipe uses it (2..5 targets);
  // drop the whole set (helper and everything only they need) and rebuild it cheapest-first with the reverse beam.
  if (GROUPS > 0) {
    int tried = 0; bool improved = true;
    while (improved && tried < GROUPS) {
      improved = false;
      int n = best.size();
      auto base = replay(starts, {}, limit); vector<V> out2(n); vector<array<int,2>> par(n);
      array<int,48> prod; prod.fill(-1);
      for (int i = 0; i < n; ++i) {
        auto s2 = best[i]; par[i] = {prod[s2[1]], prod[s2[2]]};
        V v = apply(s2[0], base.reg[s2[1]], base.reg[s2[2]]); out2[i] = v; base.reg[s2[3]] = v; prod[s2[3]] = i;
      }
      // first producer of each required target
      unordered_map<V,int> firstp; unordered_set<V> seen0(starts.begin(), starts.end());
      for (int i = 0; i < n; ++i) if (required.count(out2[i]) && !firstp.count(out2[i]) && !seen0.count(out2[i])) firstp[out2[i]] = i;
      // ancestors of each target producer
      vector<pair<V,vector<char>>> anc;
      for (auto& [t, i0] : firstp) {
        vector<char> a(n, 0); vector<int> st{i0}; a[i0] = 1;
        while (!st.empty()) { int u = st.back(); st.pop_back(); for (int p : par[u]) if (p >= 0 && !a[p]) { a[p] = 1; st.push_back(p); } }
        anc.push_back({t, std::move(a)});
      }
      set<vector<V>> done;
      for (int h = 0; h < n && tried < GROUPS && !improved; ++h) {
        if (firstp.count(out2[h]) && firstp[out2[h]] == h) continue;
        vector<V> grp;
        for (auto& [t, a] : anc) if (a[h]) grp.push_back(t);
        if (grp.size() < 2 || grp.size() > 5) continue;
        sort(grp.begin(), grp.end());
        if (!done.insert(grp).second) continue;
        unordered_set<V> req2 = required; for (V t : grp) req2.erase(t);
        auto prefix = retain(best, out2, req2);
        if (prefix.size() + grp.size() >= best.size()) continue;
        ++tried;
        vector<Instruction> cur = prefix; bool ok = true;
        // insert the group's targets in the order of their original first production
        vector<pair<int,V>> order; for (V t : grp) order.push_back({firstp[t], t});
        sort(order.begin(), order.end());
        for (auto& [pos, t] : order) {
          auto chk = replay(starts, cur, limit);
          if (chk.covered.count(t)) continue;
          auto nxt = insert_best(starts, limit, cur, t, width, best.size(), max(1, STEP));
          if (nxt.empty()) { ok = false; break; }
          cur = std::move(nxt);
        }
        struct { vector<Instruction> program; } m{cur};
        if (getenv("DBG")) cerr << "group " << grp.size() << " prefix " << prefix.size() << " rebuilt " << (ok ? (int)m.program.size() : -1) << " best " << best.size() << "\n";
        if (ok && m.program.size() < best.size()) { best = std::move(m.program); improved = true; }
      }
    }
  }
  return best;
}

int main(int argc, char** argv) {
  int width = argc > 1 ? stoi(argv[1]) : 3, attempts = argc > 2 ? stoi(argv[2]) : 8; STEP = argc > 3 ? stoi(argv[3]) : 0; PAIRS = argc > 4 ? stoi(argv[4]) : 0; GROUPS = argc > 5 ? stoi(argv[5]) : 0;
  int ns, nt, np; V limit;
  while (cin >> ns >> nt >> limit >> np) {
    vector<V> starts(ns), targets(nt);
    for (V& v : starts) cin >> v;
    for (V& v : targets) cin >> v;
    vector<Instruction> p(np);
    for (auto& s : p) {
      char op; cin >> op >> s[1] >> s[2] >> s[3]; s[0] = op;
    }
    auto best = optimize(starts, targets, limit, p, width, attempts);
    cout << best.size();
    for (auto s : best) cout << ' ' << char(s[0]) << ' ' << s[1] << ' ' << s[2] << ' ' << s[3];
    cout << '\n';
  }
}
