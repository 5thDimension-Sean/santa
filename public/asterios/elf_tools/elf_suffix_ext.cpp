// Based on the suffix target rebuilder in Nilanjan_Saha123, "Best-of-Public Elf Optimizer" (Kaggle, Apache-2.0).
// Change: optional dense cut positions (argv[3] = step; 0 keeps the original 4 positions).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
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

int STEP = 0;
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
  return best;
}

int main(int argc, char** argv) {
  int width = argc > 1 ? stoi(argv[1]) : 3, attempts = argc > 2 ? stoi(argv[2]) : 8; STEP = argc > 3 ? stoi(argv[3]) : 0;
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
