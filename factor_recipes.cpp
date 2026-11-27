// Factor offsets around each requested output. Report candidate intermediate
// values; the regional integer program chooses recipes and checks sharing.
#include "public/goodpjw2008/atakan_optimizers/santa_factor64.h"
#include <iostream>
#include <random>
#include <unordered_map>
#include <unordered_set>

using U=std::uint64_t;
struct Option { int cost; std::uint64_t tie; std::vector<U> values; };
int main(int argc,char** argv) {
    if(argc==2 && std::string(argv[1])=="--divisors") {
        U v;
        while(std::cin>>v) {
            if(!v) return 1;
            auto divisors=small_factor::divisors(v,2500);
            for(auto q:divisors) std::cout<<q<<' ';
            std::cout<<'\n';
        }
        return 0;
    }
    U limit,seed; int nf,nt;
    if(!(std::cin>>limit>>seed>>nf>>nt) || nf<1 || nt<1) return 1;
    std::mt19937_64 random(seed);
    std::vector<U> free(nf),targets(nt);
    for(U &x:free) std::cin>>x;
    for(U &x:targets) std::cin>>x;
    std::unordered_set<U> available(free.begin(),free.end()),one;
    for(int i=0;i<nf;++i) for(int j=0;j<=i;++j) {
        U a=std::max(free[i],free[j]), b=std::min(free[i],free[j]);
        if(a<=limit-b) one.insert(a+b);
        if(a>b) one.insert(a-b);
        if(a<=limit/b) one.insert(a*b);
        if(a%b==0) one.insert(a/b);
    }
    // For a two-operation factor store its one-operation intermediate.
    std::unordered_map<U,U> middle;
    auto construct=[&](U v,std::vector<U>& out) {
        if(available.count(v)) return true;
        if(one.count(v)) {out.push_back(v); return true;}
        auto [it,unseen]=middle.emplace(v,0);
        if(unseen) for(U f:free) {
            U qs[]={v>f?v-f:0,v<=limit-f?v+f:0,f>v?f-v:0,
                    v%f==0?v/f:0,v<=limit/f?v*f:0,f%v==0?f/v:0};
            for(U q:qs) if(q!=v && one.count(q)) {it->second=q;break;}
            if(it->second) break;
        }
        if(!it->second) return false;
        out.push_back(it->second);out.push_back(v);return true;
    };
    std::vector<U> offsets=free;
    std::shuffle(offsets.begin(),offsets.end(),random);
    if(offsets.size()>48) offsets.resize(48);
    for(U x=1;x<=64;++x) if(available.count(x)||one.count(x)) offsets.push_back(x);
    offsets.push_back(0);
    std::sort(offsets.begin(),offsets.end());
    offsets.erase(std::unique(offsets.begin(),offsets.end()),offsets.end());
    // Index two-operation products once per region, allowing t = a*n +/- b*m
    // to be tested by a lookup even when neither product was in a donor tape.
    std::vector<U> coefficients(one.begin(),one.end());
    std::shuffle(coefficients.begin(),coefficients.end(),random);
    if(coefficients.size()>8192) coefficients.resize(8192);
    coefficients.insert(coefficients.begin(),free.begin(),free.end());
    std::unordered_map<U,U> products;
    for(U n:coefficients) for(U a:free) {
        if(n<=1 || a<=1 || n>limit/a) continue;
        products.emplace(n*a,n);
    }
    std::vector<U> productList;
    for(auto [v,n]:products) productList.push_back(v);
    std::shuffle(productList.begin(),productList.end(),random);
    std::unordered_set<U> output;
    for(U t:targets) {
        std::vector<Option> options;
        int matches=0;
        for(U a:productList) {
            if(a==t) continue;
            U candidates[]={t>a?t-a:0,t<=limit-a?t+a:0,a>t?a-t:0};
            for(U b:candidates) {
                auto it=products.find(b);
                if(it==products.end() || b==t) continue;
                std::vector<U> vals;
                for(U v:{a,b,products.at(a),it->second}) if(!available.count(v)) vals.push_back(v);
                std::sort(vals.begin(),vals.end());vals.erase(std::unique(vals.begin(),vals.end()),vals.end());
                if(std::find(vals.begin(),vals.end(),t)==vals.end()) {
                    options.push_back({int(vals.size())+1,random(),std::move(vals)});
                    ++matches;
                }
            }
            if(matches>=100) break;
        }
        for(U delta:offsets) {
            U adjusted[]={t>delta?t-delta:0,t<=limit-delta?t+delta:0,delta>t?delta-t:0};
            for(int branch=0;branch<3;++branch) {
                U n=adjusted[branch];
                if(n<4 || (delta==0 && branch!=0)) continue;
                auto divisors=small_factor::divisors(n,700);
                for(U a:divisors) {
                    U b=n/a;
                    if(a>b) break;
                    if(a<=1 || a==t || b==t) continue;
                    std::vector<U> vals;
                    if(!construct(a,vals)||!construct(b,vals)) continue;
                    if(delta && !construct(delta,vals)) continue;
                    if(n!=t && !available.count(n)) vals.push_back(n);
                    if(std::find(vals.begin(),vals.end(),t)!=vals.end()) continue;
                    std::sort(vals.begin(),vals.end());vals.erase(std::unique(vals.begin(),vals.end()),vals.end());
                    int cost=int(vals.size())+1;
                    if(cost<=6) options.push_back({cost,random(),std::move(vals)});
                }
            }
        }
        std::sort(options.begin(),options.end(),[](const auto& a,const auto& b){return a.cost!=b.cost?a.cost<b.cost:a.tie<b.tie;});
        if(!options.empty()){
            int accepted=0;
            for(auto& opt:options) {
                if(opt.cost>options[0].cost+1 || accepted==8) break;
                auto old=output.size();output.insert(opt.values.begin(),opt.values.end());
                if(old!=output.size()) ++accepted;
            }
        }
    }
    for(U v:output) std::cout<<v<<' ';
    std::cout<<'\n';
}
