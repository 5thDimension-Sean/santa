#include "arithmetic_search.hpp"
#include <algorithm>
#include <unordered_set>
#include <vector>
struct Rule { N v; Op r; };
struct Patch { int n; Rule a[4]; };
extern "C" __declspec(dllexport) int patches(const N* vs,int n,N t,N limit,Patch* dest,int cap){
#ifdef CACHE_POOL
    static thread_local std::vector<N> prefix,values;
    static thread_local std::unordered_set<N> free;
    static thread_local std::unordered_map<N,Op> ones;
    static thread_local N lastlimit=0;
    if(limit!=lastlimit || int(prefix.size())!=n || !std::equal(prefix.begin(),prefix.end(),vs)){
    lastlimit=limit;prefix.assign(vs,vs+n);values=prefix;free.clear();ones.clear();
    free.insert(vs,vs+n);
    for(int i=0;i<n;++i)for(int j=0;j<=i;++j)operations(vs[i],vs[j],limit,[&](N v,Op op){
        if(!free.count(v) && ones.emplace(v,op).second)values.push_back(v);
    });
    }
#else
    std::unordered_set<N> free(vs,vs+n);
    std::unordered_map<N,Op> ones;
    std::vector<N> values(vs,vs+n);
    for(int i=0;i<n;++i)for(int j=0;j<=i;++j)operations(vs[i],vs[j],limit,[&](N v,Op op){
        if(v!=t && !free.count(v) && ones.emplace(v,op).second)values.push_back(v);
    });
#endif
    int count=0;
    auto match=[&](N v,bool outer,Op end){
        if(!v || (outer && (v==t || free.count(v))))return;
        for(N h:values){
            if(count>=cap)return;
            if(h==t || h==v)continue;
            inverse(h,v,limit,[&](N x,Op r){
                if(count>=cap || x==v || x==t || h>x || (!free.count(x) && !ones.count(x)))return;
                Patch p{};auto add=[&](N z,Op recipe){p.a[p.n++]={z,recipe};};
                if(ones.count(h))add(h,ones.at(h));
                if(x!=h && ones.count(x))add(x,ones.at(x));
                add(v,r);if(outer)add(t,end);
                dest[count++]=p;
            });
        }
    };
    match(t,false,Op{});
    for(int i=0;i<n && count<cap;++i){
        N k=vs[i];
        if(t<=limit-k)match(t+k,true,{'-',t+k,k});
        if(t>k)match(t-k,true,{'+',t-k,k});
        if(k>t)match(k-t,true,{'-',k,k-t});
    }
    return count;
}
