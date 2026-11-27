// Complete two-instruction operand table and the existing four-step leaf search.
#include "arithmetic_search.hpp"
#include "four_step_leaf.hpp"
#include <stdexcept>
#include <string>

struct Entry {
    N value;
    uint32_t first, encoded;
    bool operator<(const Entry& other) const {
        return std::tie(value, first, encoded) < std::tie(other.value, other.first, other.encoded);
    }
};
struct Node { N value; Op rule; };
using Chain = std::vector<Node>;

void print_chain(const Chain& chain) {
    for (const auto& n : chain)
        std::cout << n.value << ' ' << n.rule.op << ' ' << n.rule.a << ' ' << n.rule.b << ' ';
    std::cout << std::endl;
}

void mitm(const std::vector<N>& kept, const std::vector<N>& targets, N limit, int cap) {
    std::vector<Node> one;
    std::unordered_set<N> seen(kept.begin(), kept.end());
    for (size_t i=0; i<kept.size(); ++i)
        for (size_t j=0; j<=i; ++j)
            operations(kept[i], kept[j], limit, [&](N v, Op op) {
                if (seen.insert(v).second) one.push_back({v,op});
            });
    const uint32_t plain = UINT32_MAX;
    std::vector<Entry> table;
    // Store a second operand index and the operation/direction in one word.
    // Reserve once to avoid a second multi-GB allocation during vector growth.
    size_t upper = kept.size()+one.size()+4*one.size()*(kept.size()+1);
    if (upper > 400000000) throw std::runtime_error("MITM table exceeds worker memory budget");
    table.reserve(upper);
    for (uint32_t j=0; j<kept.size(); ++j) table.push_back({kept[j],plain,j});
    for (uint32_t i=0; i<one.size(); ++i) {
        N a=one[i].value;
        table.push_back({a,i,plain});
        for (uint32_t j=0; j<=kept.size(); ++j) {
            N b=j==kept.size()?a:kept[j];
            operations(a,b,limit,[&](N v,Op op) {
                if (seen.count(v)) return;
                unsigned code=std::string("+-*/").find(op.op);
                table.push_back({v,i,j*8+2*code+(op.a!=a)});
            });
        }
    }
    std::sort(table.begin(), table.end());
    table.erase(std::unique(table.begin(),table.end(),[](auto x,auto y){return x.value==y.value;}),table.end());
    auto append=[&](Chain& chain, Entry e) {
        if (e.first==plain) return;
        Node n=one[e.first];
        chain.push_back(n);
        if (e.encoded==plain) return;
        auto j=e.encoded/8;
        N a=n.value, b=j==kept.size()?a:kept[j];
        if (e.encoded%2) std::swap(a,b);
        chain.push_back({e.value,{"+-*/"[e.encoded/2%4],a,b}});
    };
    for (N target:targets) {
        int count=0;
        for (auto x:table) {
            if (count>=cap) break;
            if (x.value==target) continue;
            inverse(x.value,target,limit,[&](N y,Op op){
                if (y<x.value || y==target || count>=cap) return;
                auto it=std::lower_bound(table.begin(),table.end(),y,
                                         [](Entry e,N v){return e.value<v;});
                if (it==table.end() || it->value!=y) return;
                Chain chain;
                append(chain,x);
                append(chain,*it);
                chain.push_back({target,op});
                print_chain(chain);
                ++count;
            });
        }
    }
}

int main() {
    std::string mode;
    N limit;
    int nk, nt, cap;
    if (!(std::cin>>mode>>limit>>nk>>nt>>cap)) return 1;
    std::vector<N> kept(nk), targets(nt);
    for(auto& v:kept) std::cin>>v;
    for(auto& v:targets) std::cin>>v;
    if (!std::cin) return 1;
    if(mode=="mitm") mitm(kept,targets,limit,cap);
    else if(mode=="finisher") {
        for(N t:targets) {
            int count=0;
            four_step_leaf(kept,limit,t,[&](N a,N b,N c,N v,Op x,Op y,Op z,Op w){
                if(count<cap) print_chain({{a,x},{b,y},{c,z},{v,w}});
                ++count;
            },[&]{return count>=cap;});
        }
    } else return 1;
}
