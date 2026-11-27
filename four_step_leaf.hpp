#pragma once
#include <algorithm>
#include <unordered_set>
// Both the historical checker and the reroute move use this same leaf generator.
template<class Emit,class Stop>
void four_step_leaf(std::vector<N> v,N limit,N t,Emit accept,Stop stop){
    int n=v.size();
        std::sort(v.begin(),v.end());
        std::unordered_map<N,std::pair<Op,Op>> back;
        std::unordered_set<N> free(v.begin(),v.end());
        // Reverse the two final operations with either operand order.
        for(N x:v)inverse(x,t,limit,[&](N b,Op last){
            if(b==t || free.count(b))return;
            for(N y:v)inverse(y,b,limit,[&](N a,Op third){
                if(a!=t && a!=b && !free.count(a))back.emplace(a,std::pair<Op,Op>{third,last});
            });
        });
        std::unordered_map<N,Op> one;
        for(int i=0;i<n;++i)for(int j=0;j<=i;++j)
            operations(v[i],v[j],limit,[&](N a,Op op){
                if(a!=t && !free.count(a))one.emplace(a,op);
            });
        // Output complete value-level witnesses; the caller charges the full DAG and replays.
        int emitted=0;
        for(auto [a,first]:one){
            if(stop() || emitted>=25000)break;
            for(N x:v){
            operations(a,x,limit,[&](N b,Op second){
                if(b==a || !back.count(b))return;
                auto [third,last]=back.at(b);
                N c=third.op=='+'?third.a+third.b:third.op=='-'?third.a-third.b:third.op=='*'?third.a*third.b:third.a/third.b;
                if(c==a || emitted>=25000)return;
                accept(a,b,c,t,first,second,third,last);++emitted;
            });
            }
        }
}
