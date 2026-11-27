#include "arithmetic_search.hpp"
#include <unordered_set>
#include <limits>
#include <algorithm>

// Cost of a concrete suffix using the candidate at least once. Keep the
// cheapest per target so many equivalent decompositions cannot inflate a vote.
extern "C" __declspec(dllexport) void vote(const N* free,int n,const N* candidates,int m,
    const N* targets,int nt,N limit,unsigned char* costs,N* shared){
    std::unordered_set<N> live(free,free+n),one;
    for(int i=0;i<n;++i)for(int j=0;j<=i;++j)
        operations(free[i],free[j],limit,[&](N v,Op){one.insert(v);});
    auto cost=[&](N v){return live.count(v)?0:one.count(v)?1:20;};
    for(int i=0;i<m;++i)for(int j=0;j<nt;++j){
        N b=candidates[i],t=targets[j];int best=20;N residual=0;
        auto add=[&](int score,N r){if(score<best){best=score;residual=r;}};
        if(b==t)add(0,0);
        inverse(b,t,limit,[&](N x,Op){add(1+(x==b?0:cost(x)),x);});
        N floor=t/b;
        for(N q:{floor,floor+1}){
            if(q<2 || q>limit/b || cost(q)>1)continue;
            N r=std::abs(t-q*b);
            if(r && cost(r)<=1)add(2+cost(q)+cost(r),r);
        }
        // T +/- a retained offset produces a multiple of b or b +/- c.
        // Precomputed one-step values also cover exact two-instruction paths.
        if(best>1){
            for(int k=0;k<n;++k){
                N x=free[k];
                for(N v:{t>x?t-x:0,t<=limit-x?t+x:0,x>t?x-t:0}){
                    if(!v || v==b)continue;
                    if(v%b==0 && cost(v/b)<=1)add(2+cost(v/b),v/b);
                    N diff=std::abs(v-b);
                    if(diff && cost(diff)<=1)add(2+cost(diff),diff);
                }
            }
        }
        costs[i*nt+j]=best;shared[i*nt+j]=residual;
    }
}

extern "C" __declspec(dllexport) void distance2(const N* vs,int n,const N* ts,int m,N limit,unsigned char* out){
    std::unordered_set<N> free(vs,vs+n),one;
    for(int i=0;i<n;++i)for(int j=0;j<=i;++j)
        operations(vs[i],vs[j],limit,[&](N v,Op){one.insert(v);});
    for(int i=0;i<m;++i){
        N t=ts[i];
        if(free.count(t)){out[i]=0;continue;}
        if(one.count(t)){out[i]=1;continue;}
        bool found=false;
        for(int j=0;j<n && !found;++j)
            inverse(vs[j],t,limit,[&](N x,Op){if(one.count(x))found=true;});
        if(t%2==0 && one.count(t/2))found=true;
        N root=std::sqrt((long double)t);
        for(N x=std::max<N>(1,root-1);x<=root+1;++x)
            if(x*x==t && one.count(x))found=true;
        out[i]=found?2:20;
    }
}
