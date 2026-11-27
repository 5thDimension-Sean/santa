#include "arithmetic_search.hpp"
#include <array>
#include <unordered_set>

using Recipes = std::unordered_map<N,Op>;
struct Tail { Op mid,last; };

template<class F> void backwards_self(N t,F emit) {
    if(t%2==0) emit(t/2, Op{'+',t/2,t/2});
    N r=std::sqrt((long double)t);
    while((r+1)<=t/(r+1)) ++r;
    while(r>t/r) --r;
    if(r*r==t) emit(r,Op{'*',r,r});
    // Self-division always gives 1, already available in one instruction.
}

void search(int id,N limit,N t,const std::vector<N>& live) {
    const std::unordered_set<N> old(live.begin(),live.end());
    Recipes one,finish;
    for(size_t i=0;i<live.size();++i) {
        for(size_t j=0;j<=i;++j)
            operations(live[i],live[j],limit,[&](N v,Op op){
                if(!old.count(v)) one.emplace(v,op);
            });
        inverse(live[i],t,limit,[&](N v,Op op){finish.emplace(v,op);});
    }
    backwards_self(t,[&](N v,Op op){finish.emplace(v,op);});
    std::unordered_map<N,Tail> tail;
    for(auto [c,last]:finish) {
        auto add=[&](N b,Op mid){tail.emplace(b,Tail{mid,last});};
        for(N x:live) inverse(x,c,limit,add);
        backwards_self(c,add);
    }
    auto hit=[&](Op a,Op b,Op c,Op d) {
        std::cout<<id;
        for(Op op:{a,b,c,d}) std::cout<<' '<<op.op<<' '<<op.a<<' '<<op.b;
        std::cout<<'\n';
    };
    for(auto [a,first]:one) {
        Recipes chain;
        auto add=[&](N b,Op op){chain.emplace(b,op);};
        for(N x:live) operations(a,x,limit,add);
        operations(a,a,limit,add);
        auto second=[&](N b,auto found) {
            auto i=one.find(b);
            if(i!=one.end()) found(i->second);
            auto j=chain.find(b);
            if(j!=chain.end()) found(j->second);
        };
        // Final uses C with a snapshot value or C again. Middle uses A and B.
        for(auto [c,last]:finish) inverse(a,c,limit,[&](N b,Op mid){
            second(b,[&](Op op){hit(first,op,mid,last);});
        });
        // Final uses A and C. Reverse the third instruction, checking all
        // choices for its operand other than B: snapshot, A, or B itself.
        inverse(a,t,limit,[&](N c,Op last){
            auto check=[&](N b,Op mid){second(b,[&](Op op){hit(first,op,mid,last);});};
            for(N x:live) inverse(x,c,limit,check);
            inverse(a,c,limit,check);
            backwards_self(c,check);
        });
        for(auto [b,op]:chain) {
            // Final uses C with an old operand or itself; B begins the two-op suffix.
            auto suffix=tail.find(b);
            if(suffix!=tail.end()) hit(first,op,suffix->second.mid,suffix->second.last);
            // Final uses B and C. An independent B can be scheduled first
            // and is covered by the A-and-C case above.
            inverse(b,t,limit,[&](N c,Op last){
                second(c,[&](Op mid){hit(first,op,mid,last);});
                inverse(b,c,limit,[&](N x,Op mid){
                    if(x==a || x==b || old.count(x)) hit(first,op,mid,last);
                });
            });
        }
    }
}

int main() {
    int id,n; N limit,t;
    while(std::cin>>id>>limit>>t>>n) {
        std::vector<N> live(n);
        for(auto& v:live) std::cin>>v;
        search(id,limit,t,live);
    }
    return std::cin.eof()?0:1;
}
