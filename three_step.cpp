#include "arithmetic_search.hpp"
void solve(int id,N limit,N t,const std::vector<N>& live) {
    std::unordered_map<N,Op> one,finish;
    for(size_t i=0;i<live.size();++i) {
        for(size_t j=0;j<=i;++j)
            operations(live[i],live[j],limit,[&](N v,Op op){one.emplace(v,op);});
        inverse(live[i],t,limit,[&](N v,Op op){finish.emplace(v,op);});
    }
    if(t%2==0) finish.emplace(t/2,Op{'+',t/2,t/2});
    N r=std::sqrt((long double)t);
    while((r+1)<=t/(r+1)) ++r;
    while(r>t/r) --r;
    if(r*r==t) finish.emplace(r,Op{'*',r,r});
    auto report=[&](Op a,Op b,Op c) {
        std::cout<<id;
        for(Op op:{a,b,c}) std::cout<<' '<<op.op<<' '<<op.a<<' '<<op.b;
        std::cout<<'\n';
    };
    // Last operation consumes the second result and a snapshot value, or itself.
    for(auto [b,last]:finish) {
        for(N x:live) inverse(x,b,limit,[&](N a,Op mid){
            auto it=one.find(a);
            if(it!=one.end()) report(it->second,mid,last);
        });
        // Second instruction can reuse its first result twice.
        if(b%2==0 && one.count(b/2)) report(one.at(b/2),Op{'+',b/2,b/2},last);
        N s=std::sqrt((long double)b);
        for(N k=std::max<N>(1,s-1);k<=s+1;++k)
            if(k*k==b && one.count(k)) report(one.at(k),Op{'*',k,k},last);
        // Self-subtraction is illegal, self-division makes 1 directly from the snapshot.
    }
    // Last operation consumes both new results. They may be independent, or
    // the second may use the first once or twice.
    std::unordered_map<N,bool> old;
    for(N x:live) old[x]=true;
    for(auto [a,first]:one) inverse(a,t,limit,[&](N b,Op last){
        auto it=one.find(b);
        if(it!=one.end()) report(first,it->second,last);
        inverse(a,b,limit,[&](N x,Op mid){
            if(x==a || old.count(x)) report(first,mid,last);
        });
    });
}

int main() {
    int id,n; N limit,t;
    while(std::cin>>id>>limit>>t>>n) {
        std::vector<N> live(n);
        for(auto& v:live) std::cin>>v;
        solve(id,limit,t,live);
    }
    return std::cin.eof()?0:1;
}

