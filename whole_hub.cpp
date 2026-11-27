#include "arithmetic_search.hpp"
#include <algorithm>
#include <array>

#include <windows.h>
#include <chrono>
#include <cmath>
#include <ctime>
#include <functional>
#include <map>
#include <set>
#include <unordered_set>
#include <tuple>
#include "four_step_leaf.hpp"

struct Rule { N v; Op op; };
using Route=std::vector<Rule>;
using Rules=std::unordered_map<N,Op>;
struct Option { int root; Route route; int credit; };
struct Roots { std::map<int,std::vector<Option>> by; double score=0; };
struct Search {
    N limit; std::vector<N> live,targets;
    std::unordered_set<N> start,known;
    Rules base,one;
    std::unordered_map<N,int> index;
    std::vector<int> support;
    std::unordered_map<N,Roots> hubs;
    std::map<int,std::vector<Option>> direct;
    std::vector<N> values,selected;
    int originals,ranked_count=0,paired=0,leaves=0,reported=0;
    double seconds;
    static double cpu(){
        FILETIME c,e,k,u; GetProcessTimes(GetCurrentProcess(),&c,&e,&k,&u);
        auto ticks=[](FILETIME t){return (double)t.dwHighDateTime*4294967296.0+t.dwLowDateTime;};
        return (ticks(k)+ticks(u))/1e7;
    }
    bool expired(){return cpu()>=seconds;}
    bool cheap(N v)const{return known.count(v)||one.count(v);}
    bool in1(N v)const{return one.count(v);}
    void source(Route& route,N v){
        auto i=one.find(v);
        if(i==one.end())return;
        for(auto &r:route)if(r.v==v)return;
        route.push_back({v,i->second});
    }
    int reach(const Rules& rules,std::vector<N>* order=nullptr)const {
        std::unordered_map<N,int> color;int n=0;
        std::function<bool(N)> walk=[&](N v){
            if(start.count(v))return true;
            int &c=color[v];if(c)return c==2;
            auto it=rules.find(v);if(it==rules.end())return false;
            c=1;auto op=it->second;
            if(!walk(op.a)||!walk(op.b))return false;
            color[v]=2;++n;if(order)order->push_back(v);return true;
        };
        for(N t:targets)if(!walk(t))return 1000000;
        return n;
    }
    std::unordered_map<N,int> ids;
    std::vector<std::pair<int,int>> edges;
    std::vector<int> stamps,target_ids;
    int epoch=0;
    int id(N value){
        auto [it,fresh]=ids.emplace(value,ids.size());
        if(fresh){edges.emplace_back(-2,-2);stamps.push_back(0);}
        return it->second;
    }
    void reset(){
        std::fill(edges.begin(),edges.end(),std::make_pair(-2,-2));
        for(N v:start){int z=id(v);edges[z]={-1,-1};}
        for(auto [v,r]:base){int z=id(v),x=id(r.a),y=id(r.b);edges[z]={x,y};}
    }
    int count(int z){
        if(stamps[z]==epoch || edges[z].first==-1)return 0;
        if(stamps[z]==-epoch || edges[z].first<0)return 1000000;
        stamps[z]=-epoch;
        auto [a,b]=edges[z];
        int n=count(a);if(n<1000000)n+=count(b)+1;
        stamps[z]=epoch;return n;
    }
    std::vector<std::vector<bool>> depends,exclusive;
    void setup(){
        for(N v:targets)target_ids.push_back(id(v));
        for(int i=0;i<(int)live.size();++i)index[live[i]]=i;
        known.insert(live.begin(),live.end());
        for(size_t i=0;i<live.size();++i)for(size_t j=0;j<=i;++j)
            operations(live[i],live[j],limit,[&](N v,Op r){if(!known.count(v))one.emplace(v,r);});
        values=live;for(auto [v,r]:one)values.push_back(v);
        // Remove the original output edge; propagate zero reference counts.
        std::unordered_map<N,int> uses;
        for(auto [v,r]:base){++uses[r.a];++uses[r.b];}
        for(N v:targets)++uses[v];
        depends.assign(live.size(),std::vector<bool>(live.size()));
        for(int i=0;i<(int)live.size();++i){
            N v=live[i];depends[i][i]=true;
            if(!base.count(v))continue;
            auto r=base.at(v);
            for(int j=0;j<i;++j)depends[i][j]=depends[index[r.a]][j] || depends[index[r.b]][j];
        }
        exclusive=depends;
        for(N v:live){
            auto &removed=exclusive[index[v]];
            std::fill(removed.begin(),removed.end(),false);
            if(start.count(v)){support.push_back(0);continue;}
            auto remaining=uses;int n=0;
            std::function<void(N)> erase=[&](N z){
                if(!base.count(z))return;
                ++n;removed[index[z]]=true;auto r=base.at(z);
                if(--remaining[r.a]==0)erase(r.a);
                if(--remaining[r.b]==0)erase(r.b);
            };
            erase(v);support.push_back(n);
        }
    }
    bool identical(const Op& a,const Op& b)const{
        return a.op==b.op && ((a.a==b.a && a.b==b.b) ||
                ((a.op=='+'||a.op=='*') && a.a==b.b && a.b==b.a));
    }
    void record(int i,Route route){
        N t=live[i];if(route.empty() || route.back().v!=t)return;
        // Reject local identities and immediate circular constructions.
        for(auto [v,r]:route)
            if(v==r.a || v==r.b || r.a==t || r.b==t)return;
        if(identical(base.at(t),route.back().op))return;
        int credit=support[i]-(int)route.size()+1;
        if(credit < -1)return;
        std::set<int> reused;
        for(auto &r:route)for(N x:{r.op.a,r.op.b}){
            auto it=index.find(x);if(it==index.end())continue;
            int j=it->second;if(depends[j][i])return;
            if(!exclusive[i][j])continue;
            for(int k=0;k<=j;++k)if(exclusive[i][k]&&depends[j][k])reused.insert(k);
        }
        credit-=reused.size();
        if(credit < -1)return;
        Option opt{i,route,credit};
        auto add=[&](std::vector<Option>& list){
            for(auto &old:list)if(old.route.size()==route.size()){
                bool same=true;for(size_t j=0;j<route.size();++j)
                    if(old.route[j].v!=route[j].v || !identical(old.route[j].op,route[j].op)){same=false;break;}
                if(same)return;
            }
            if(list.size()<4)list.push_back(opt);
            else{
                auto worst=std::min_element(list.begin(),list.end(),[](auto& a,auto& b){return a.credit<b.credit;});
                if(worst->credit<credit)*worst=opt;
            }
        };
        if(route.size()==1){add(direct[i]);return;}
        for(size_t j=0;j+1<route.size();++j){
            N h=route[j].v;if(known.count(h))continue;
            add(hubs[h].by[i]);
        }
    }
    void generate(int i){
        N t=live[i];if(!support[i])return;
        // One- and two-operation alternatives, including neutral substitutions.
        for(N x:live)inverse(x,t,limit,[&](N y,Op r){
            if(!cheap(y))return;
            Route s;source(s,y);s.push_back({t,r});record(i,s);
        });
        if(support[i]<2)return;
        // Vote for proportional outputs. Quotient or residual may itself cost a step.
        for(N q:values){
            if(q<2)continue;
            for(N h:{t/q,t/q+1}){
                if(h<2 || h>limit/q || !cheap(h))continue;
                N prod=h*q,delta=std::abs(t-prod);
                if(!delta || !cheap(delta) || known.count(prod))continue;
                Route s;source(s,q);source(s,h);source(s,delta);s.push_back({prod,{'*',h,q}});
                s.push_back({t,prod<t?Op{'+',prod,delta}:Op{'-',prod,delta}});record(i,s);
            }
        }
        // Split backward at an offset and a multiplier. Intersect the residual
        // with one-step forward values to obtain two-step hubs and shared chains.
        for(N c:live)for(int sign:{1,-1}){
            if((sign==1 && t>limit-c)||(sign==-1 && t<=c))continue;
            N prod=sign==1?t+c:t-c;if(known.count(prod))continue;
            Op last=sign==1?Op{'-',prod,c}:Op{'+',prod,c};
            for(N q:live){
                if(q<2 || prod%q)continue;
                N mid=prod/q;if(mid==t || mid==q || known.count(mid))continue;
                if(cheap(mid)){
                    Route s;source(s,mid);s.push_back({prod,{'*',q,mid}});s.push_back({t,last});record(i,s);
                }
                if(support[i]<3)continue;
                // Split an additive factor into two independently cheap operands.
                // Require an existing vote on a different root for at least one operand.
                std::vector<N> voted;
                for(auto &[h,entries]:hubs)if(in1(h) && !entries.by.count(i))voted.push_back(h);
                for(N h:voted)inverse(h,mid,limit,[&](N e,Op r){
                    if(!cheap(e))return;
                    Route s;source(s,h);source(s,e);s.push_back({mid,r});s.push_back({prod,{'*',q,mid}});s.push_back({t,last});record(i,s);
                });
                for(N x:live)inverse(x,mid,limit,[&](N h,Op r){
                    if(!cheap(h))return;
                    Route s;source(s,h);s.push_back({mid,r});s.push_back({prod,{'*',q,mid}});s.push_back({t,last});record(i,s);
                });
            }
        }
    }
    double merit(const Roots& entries)const{
        double total=0;int relevant=0;
        for(auto &[i,opts]:entries.by){
            int c=-100;for(auto &o:opts)c=std::max(c,o.credit);
            if(c>=0){total+=std::min(3,c)+0.25;if(support[i]>=2)++relevant;}
        }
        return total+.45*relevant;
    }
    // Evaluate the entire live DAG for each candidate substitution. This
    // recomputes exact dead support and charges any previously shared operands.
    int apply(Rules& g,const Option& o,bool commit){
        std::vector<std::pair<int,std::pair<int,int>>> undo;
        for(auto [v,r]:o.route){
            int z=id(v);
            if(v!=live[o.root] && edges[z].first!=-2)continue;
            int x=id(r.a),y=id(r.b);
            undo.push_back({z,edges[z]});edges[z]={x,y};
            if(commit)g[v]=r;
        }
        ++epoch;int n=0;
        for(int z:target_ids){n+=count(z);if(n>=1000000)break;}
        if(!commit)for(auto it=undo.rbegin();it!=undo.rend();++it)edges[it->first]=it->second;
        return n;
    }
    std::set<std::vector<std::tuple<N,char,N,N>>> seen;
    void emit(const Rules& g,int n,N a,N b){
        if(n>=originals || reported>=100)return;
        std::vector<N> order;if(reach(g,&order)!=n)return;
        std::vector<std::tuple<N,char,N,N>> sig;
        for(N v:order){auto r=g.at(v);sig.emplace_back(v,r.op,r.a,r.b);}
        if(!seen.insert(sig).second)return;
        std::cout<<"C "<<n<<" "<<a<<" "<<b<<"\n";
        for(N v:order){Op r=g.at(v);std::cout<<v<<" "<<r.op<<" "<<r.a<<" "<<r.b<<"\n";}
        ++reported;
    }
    // Four-instruction leaf: two steps forward from the proposed hub and two
    // backward from each output. A new hub is paid inside the final bundle.
    void extend(N h,const Route& hsrc,const std::vector<int>& roots){
        std::unordered_map<N,Op> fwd;
        for(N x:live)operations(h,x,limit,[&](N z,Op op){if(!known.count(z)&&z!=h)fwd.emplace(z,op);});
        for(int i:roots){
            N t=live[i];
            for(N c:live)inverse(c,t,limit,[&](N v,Op last){
                if(known.count(v)||v==h)return;
                auto link=[&](N u,Op third){
                    if(u==h || u==t || known.count(u))return;
                    auto z=fwd.find(u);
                    if(z!=fwd.end()){
                        Route route=hsrc;route.push_back({u,z->second});route.push_back({v,third});route.push_back({t,last});record(i,route);
                    }
                    // The second forward operation reuses another live value.
                    // A factor on one outside leg keeps this split bounded.
                    if(third.op!='*' && third.op!='/')return;
                    for(N x:live)inverse(x,u,limit,[&](N g,Op second){
                        auto it=fwd.find(g);if(it==fwd.end()||g==t||g==v||g==u)return;
                        Route route=hsrc;route.push_back({g,it->second});route.push_back({u,second});
                        route.push_back({v,third});route.push_back({t,last});record(i,route);
                    });
                };
                for(N q:live)if(q>=2 && v%q==0)link(v/q,{'*',v/q,q});
                // Signed residual: T = q*factor +/- (h +/- offset).
                if(last.op=='+' || last.op=='-'){
                    auto it=fwd.find(v);if(it!=fwd.end()){Route route=hsrc;route.push_back({v,it->second});route.push_back({t,last});record(i,route);}
                }
            });
        }
    }
    Route construction(N h)const{
        auto it=one.find(h);if(it!=one.end())return {{h,it->second}};
        auto info=hubs.find(h);
        if(info!=hubs.end())for(auto &[i,list]:info->second.by)for(auto &o:list){
            Route prefix;for(auto r:o.route){prefix.push_back(r);if(r.v==h && prefix.size()<=2)return prefix;}
        }
        return {};
    }
    void pair_links(N a,N b){
        Route ca=construction(a),cb=construction(b);
        if(ca.empty()||cb.empty())return;
        for(size_t i=0;i<live.size();++i){
            if(support[i]<2)continue;
            N t=live[i];
            for(N x:live){
                if(x<2 || a>limit/x)continue;
                N prod=a*x;if(known.count(prod)||prod==t)continue;
                for(N v:{std::abs(t-prod),t<=limit-prod?t+prod:0}){
                    if(v<=0||known.count(v)||v==b||v==t)continue;
                    inverse(b,v,limit,[&](N k,Op second){
                        if(!known.count(k))return;
                        Route s=ca;s.insert(s.end(),cb.begin(),cb.end());
                        s.push_back({prod,{'*',a,x}});s.push_back({v,second});
                        Op last=prod>t?Op{'-',prod,v}:Op{'+',prod,v};
                        if(v==t+prod)last={'-',v,prod};
                        s.push_back({t,last});record(i,s);
                    });
                }
            }
        }
    }
    struct Survivor {Rules rules;int n;N a,b;};
    std::vector<Survivor> survivors;
    std::set<std::pair<N,N>> saved_pairs;
    void retain(const Rules& g,int n,N a,N b){
        if(n>originals || saved_pairs.count({a,b}))return;
        if(survivors.size()>=4 && n>=survivors.back().n)return;
        std::vector<N> order;reach(g,&order);
        if(std::find(order.begin(),order.end(),a)==order.end())return;
        saved_pairs.emplace(a,b);
        survivors.push_back({g,n,a,b});
        std::stable_sort(survivors.begin(),survivors.end(),[](auto &x,auto &y){return x.n<y.n;});
        if(survivors.size()>4)survivors.pop_back();
    }
    int finishes=0;
    void finish(){
        for(auto &state:survivors){
            for(N t:targets){
                if(expired()||finishes>=8)return;
                auto keep=state.rules;
                keep.erase(t);
                std::unordered_set<N> seen=start;
                std::vector<N> pool(start.begin(),start.end());
                int retained=0;
                std::function<bool(N)> visit=[&](N z){
                    if(seen.count(z))return true;
                    if(!keep.count(z))return false;
                    auto r=keep.at(z);
                    bool x=visit(r.a),y=visit(r.b);
                    if(x&&y){seen.insert(z);pool.push_back(z);return true;}
                    return false;
                };
                // Count surviving dependencies even when a target depends on the cut.
                std::unordered_set<N> touched=start;
                std::function<void(N)> mark=[&](N z){
                    if(!keep.count(z)||!touched.insert(z).second)return;
                    ++retained;auto r=keep.at(z);mark(r.a);mark(r.b);
                };
                for(N v:targets){mark(v);visit(v);}
                if(retained+4>=originals || !seen.count(state.a))continue;
                ++finishes;
                four_step_leaf(pool,limit,t,[&](N u,N v,N w,N t,Op r,Op s,Op z,Op last){
                    if(touched.count(u)||touched.count(v)||touched.count(w))return;
                    Rules candidate=state.rules;
                    candidate[u]=r;candidate[v]=s;candidate[w]=z;candidate[t]=last;
                    int n=reach(candidate);if(n<originals)emit(candidate,n,state.a,state.b);
                },[&]{return expired();});
            }
        }
    }
    void bundle(N a,N b){
        std::map<int,std::vector<Option>> choices=hubs.at(a).by;
        if(b)for(auto &[i,list]:hubs.at(b).by)choices[i].insert(choices[i].end(),list.begin(),list.end());
        std::vector<int> roots;
        for(auto &[i,list]:choices)roots.push_back(i);
        if(roots.size()<2)return;
        for(auto &[i,alts]:direct)if(choices.count(i))choices[i].insert(choices[i].end(),alts.begin(),alts.end());
        // Start at each promising reroute, allowing a neutral or loss-making
        // seed. Keep direct alternate recipes available for newly dead chains.
        std::stable_sort(roots.begin(),roots.end(),[&](int i,int j){return support[i]>support[j];});
        for(int seed=0;seed<std::min<int>(roots.size(),b?8:4);++seed){
            reset();Rules g=base;std::set<int> done;int n=originals;
            for(size_t step=0;step<std::min<size_t>(roots.size(),12);++step){
                int best=1000000,chosen=-1;const Option* route=nullptr;
                for(int i:roots){
                    if(done.count(i) || (step==0&&i!=roots[seed]))continue;
                    for(auto &o:choices.at(i)){
                        int cost=apply(g,o,false);
                        if(cost<best){best=cost;chosen=i;route=&o;}
                    }
                }
                if(!route || best>originals+5)break;
                n=apply(g,*route,true);done.insert(chosen);
                if(step && n<=originals)retain(g,n,a,b);
                if(n<originals){emit(g,n,a,b);break;}
            }
        }
    }
    void solve(){
        setup();
        for(int i=0;i<(int)live.size();++i){generate(i);if(expired())break;}
        for(int i=0;i<(int)live.size()&&!expired();++i)if(support[i]>=3)generate(i);
        std::vector<std::pair<double,N>> rank;
        for(auto &[h,rs]:hubs)if(rs.by.size()>=2)rank.emplace_back(merit(rs),h);
        std::sort(rank.begin(),rank.end(),[](auto x,auto y){return x.first!=y.first?x.first>y.first:x.second<y.second;});
        ranked_count=rank.size();
        for(size_t j=0;j<rank.size() && j<48;++j)selected.push_back(rank[j].second);
        std::vector<int> costly;
        for(size_t i=0;i<live.size();++i)if(support[i]>=3)costly.push_back(i);
        std::stable_sort(costly.begin(),costly.end(),[&](int i,int j){return support[i]>support[j];});
        if(costly.size()>12)costly.resize(12);
        for(N h:selected){
            if(expired())break;Route prefix=construction(h);if(prefix.empty())continue;
            extend(h,prefix,costly);++leaves;bundle(h,0);
        }
        for(int i=0;i<std::min<int>(32,selected.size())&&!expired();++i)
            for(int j=0;j<i&&!expired();++j){
                N a=selected[i],b=selected[j];pair_links(a,b);pair_links(b,a);bundle(a,b);++paired;
            }
        finish();std::cout<<"F "<<finishes<<"\n";
        std::cout<<"T "<<cpu()<<"\n";
        std::cout<<"S "<<one.size()<<" "<<hubs.size()<<" "<<ranked_count<<" "<<leaves<<" "<<paired<<" "<<reported<<"\n";
        for(size_t i=0;i<rank.size();++i)std::cout<<"H "<<i+1<<" "<<rank[i].second<<"\n";
    }
};
int main(){
    Search s;int ns,nt,len;
    if(!(std::cin>>s.seconds>>s.limit>>ns>>nt>>len))return 1;
    for(int i=0;i<ns;++i){N v;std::cin>>v;s.start.insert(v);s.live.push_back(v);}
    for(int i=0;i<nt;++i){N v;std::cin>>v;s.targets.push_back(v);}
    for(int i=0;i<len;++i){N v;Op r;std::cin>>v>>r.op>>r.a>>r.b;s.base[v]=r;s.live.push_back(v);}
    s.originals=len;s.solve();
}
