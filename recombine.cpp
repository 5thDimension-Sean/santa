// Search over value sets from several complete programs for the same puzzle.
// stdin: one puzzle (limit, seed count, target count, program count; seeds;
// targets; then each length and its instructions). stdout: an improved program.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using Value = std::uint64_t;
using Mask = std::vector<unsigned char>;
struct Rule { char op; int a, b, out; };
struct Instruction { char op; int a, b, d; };
struct Bank {
    std::vector<int> order;
    std::unordered_map<int, Rule> producer;
};

struct Search {
    Value limit;
    int expected;
    std::vector<Value> values;
    std::unordered_map<Value,int> ids;
    std::vector<int> seeds, targets;
    std::vector<Bank> banks;
    std::vector<int> bridges;
    std::vector<Rule> rules;
    std::vector<std::vector<int>> users;
    Mask required, bestSet;
    std::vector<Instruction> result;
    std::mt19937 rng;

    int intern(Value v) {
        auto [i, fresh] = ids.emplace(v, int(values.size()));
        if (fresh) values.push_back(v);
        return i->second;
    }
    void read() {
        int ns, nt, nb;
        std::cin >> limit >> ns >> nt >> nb;
        for (int i=0;i<ns;++i) { Value v; std::cin>>v; seeds.push_back(intern(v)); }
        for (int i=0;i<nt;++i) { Value v; std::cin>>v; targets.push_back(intern(v)); }
        for (int i=0;i<nb;++i) {
            int n; std::cin>>n;
            if (i==0) expected=n;
            Bank bank;
            std::array<int,48> reg{};
            std::copy(seeds.begin(),seeds.end(),reg.begin());
            for (int k=0;k<n;++k) {
                char op; int a,b,d; std::cin>>op>>a>>b>>d;
                int x=reg[a], y=reg[b];
                Value u=values[x], v=values[y];
                int z=intern(op=='+' ? u+v : op=='-' ? u-v : op=='*' ? u*v : u/v);
                reg[d]=z;
                if (bank.producer.emplace(z,Rule{op,x,y,z}).second) bank.order.push_back(z);
            }
            banks.push_back(std::move(bank));
        }
        required.assign(values.size(),0);
        for (int x:seeds) required[x]=1;
        for (int x:targets) required[x]=1;
        bestSet=required;
        for (int x:banks[0].order) bestSet[x]=1;
    }
    void buildRules() {
        users.resize(values.size());
        auto add=[&](char op,int a,int b,Value v) {
            auto it=ids.find(v);
            if (it==ids.end() || it->second==a || it->second==b) return;
            int r=int(rules.size());
            rules.push_back({op,a,b,it->second});
            users[a].push_back(r);
            if (a!=b) users[b].push_back(r);
        };
        for (int i=0;i<int(values.size());++i) for(int j=0;j<=i;++j) {
            int a=i,b=j;
            if(values[a]<values[b]) std::swap(a,b);
            Value x=values[a],y=values[b];
            if (x<=limit-y) add('+',a,b,x+y);
            if (x>y) add('-',a,b,x-y);
            if (x<=limit/y) add('*',a,b,x*y);
            if (x%y==0) add('/',a,b,x/y);
        }
    }
    // A new value with several arithmetic links can replace work in more
    // than one branch. Count links before selecting a bounded helper pool.
    void addBridges() {
        std::unordered_map<Value,int> links;
        auto record=[&](Value v) { if(!ids.count(v)) ++links[v]; };
        const auto &base=banks.front().order;
        for(int i=0;i<int(base.size());++i) for(int j=0;j<=i;++j) {
            Value a=values[base[i]], b=values[base[j]];
            if(a<b) std::swap(a,b);
            if(a<=limit-b) record(a+b);
            if(a>b) record(a-b);
            if(a<=limit/b) record(a*b);
            if(a%b==0) record(a/b);
        }
        std::vector<std::pair<int,Value>> ranked;
        for(auto [v,count]:links) if(count>=2) ranked.emplace_back(count,v);
        std::shuffle(ranked.begin(),ranked.end(),rng);
        std::stable_sort(ranked.begin(),ranked.end(),[](auto a,auto b){return a.first>b.first;});
        if(ranked.size()>128) ranked.resize(128);
        for(auto [count,v]:ranked) bridges.push_back(intern(v));
        required.resize(values.size()); bestSet.resize(values.size());
    }
    // Derive an acyclic program using only allowed values. Discard operations
    // whose results never feed a target, then allocate registers at last use.
    bool close(Mask &allowed, Mask *reachable=nullptr) {
        Mask reached(values.size());
        std::vector<int> queue, chosen(values.size(),-1);
        for(int x:seeds) if(!reached[x]) {reached[x]=1; queue.push_back(x);}
        for(std::size_t i=0;i<queue.size();++i) {
            for(int ri:users[queue[i]]) {
                const auto &r=rules[ri];
                if(!allowed[r.out] || reached[r.out] || !reached[r.a] || !reached[r.b]) continue;
                chosen[r.out]=ri;
                reached[r.out]=1;
                queue.push_back(r.out);
            }
        }
        if(reachable) *reachable=reached;
        for(int x:targets) if(!reached[x]) return false;
        Mask live=required;
        std::vector<int> uses(values.size());
        int length=0;
        for(auto it=queue.rbegin();it!=queue.rend();++it) {
            if(!live[*it] || chosen[*it]<0) continue;
            const auto &r=rules[chosen[*it]];
            live[r.a]=live[r.b]=1;
            ++uses[r.a]; ++uses[r.b]; ++length;
        }
        allowed=live;
        if(length>=expected) return true;
        std::vector<int> location(values.size(),-1),free;
        Mask occupied(48,0);
        for(int i=0;i<int(seeds.size());++i) if(uses[seeds[i]] && location[seeds[i]]<0) {
            location[seeds[i]]=i; occupied[i]=1;
        }
        for(int i=0;i<48;++i) if(!occupied[i]) free.push_back(i);
        std::vector<Instruction> prog;
        std::vector<int> rank(values.size(),int(banks.front().order.size()));
        for(int i=0;i<int(banks.front().order.size());++i) rank[banks.front().order[i]]=i;
        Mask done(values.size());
        while(int(prog.size())<length) {
            int next=-1,priority=-999999;
            for(int x:queue) {
                if(done[x] || !live[x] || chosen[x]<0) continue;
                const auto &r=rules[chosen[x]];
                if(location[r.a]<0 || location[r.b]<0) continue;
                int releases=int(uses[r.a]==(r.a==r.b?2:1))+int(r.a!=r.b && uses[r.b]==1);
                if(free.empty() && !releases) continue;
                int score=-rank[x]+(free.size()<6 ? 10000*(releases+int(!uses[x])) : 0);
                if(score>priority) {next=x; priority=score;}
            }
            if(next<0) return true;
            done[next]=1;
            const auto &r=rules[chosen[next]];
            int a=location[r.a],b=location[r.b];
            if(--uses[r.a]==0) {free.push_back(a);location[r.a]=-1;}
            if(--uses[r.b]==0) {free.push_back(b);location[r.b]=-1;}
            int d=free.back(); free.pop_back();
            location[next]=d;
            prog.push_back({r.op,a,b,d});
            if(!uses[next]) {free.push_back(d);location[next]=-1;}
        }
        expected=length; bestSet=live; result=std::move(prog);
        return true;
    }
    void addBranch(const Bank& bank,int v,Mask& mask) {
        if(mask[v]) return;
        mask[v]=1;
        auto it=bank.producer.find(v);
        if(it!=bank.producer.end()) {
            addBranch(bank,it->second.a,mask);
            addBranch(bank,it->second.b,mask);
        }
    }
    // Keep the set of computed values fixed, but change their recipes and
    // order to reduce the number of live registers at the worst cuts.
    void pack(double seconds) {
        auto deadline=std::chrono::steady_clock::now()+std::chrono::duration<double>(seconds);
        std::vector<int> original;
        Mask seed(values.size());
        for(int x:seeds) seed[x]=1;
        for(int x:banks[0].order) if(!seed[x]) original.push_back(x);
        int n=int(original.size());
        if(n>=expected) return;
        std::vector<std::vector<int>> recipes(values.size());
        for(int j=0;j<int(rules.size());++j) recipes[rules[j].out].push_back(j);
        std::vector<int> order=original,pos(values.size(),-1),choice(values.size(),-1);
        auto positions=[&](){for(int i=0;i<n;++i) pos[order[i]]=i;};
        positions();
        for(int x:order) {
            auto r=banks[0].producer.at(x);
            for(int j:recipes[x]) {
                auto s=rules[j];
                if(s.op==r.op && ((s.a==r.a && s.b==r.b) ||
                    ((s.op=='+' || s.op=='*') && s.a==r.b && s.b==r.a))) {choice[x]=j;break;}
            }
            if(choice[x]<0) return;
        }
        std::vector<int> last(values.size()),diff(n+1);
        bool fits=false;
        auto score=[&]() {
            std::fill(last.begin(),last.end(),-1);
            for(int i=0;i<n;++i) {
                const auto &r=rules[choice[order[i]]];
                last[r.a]=last[r.b]=i;
            }
            std::fill(diff.begin(),diff.end(),0);
            for(int x=0;x<int(values.size());++x) if(pos[x]+1<last[x]) {
                ++diff[pos[x]+1]; --diff[last[x]];
            }
            int live=1,cost=0;
            fits=true;
            for(int i=0;i<n;++i) {
                live+=diff[i];
                int excess=std::max(0,live-48);
                if(excess) fits=false;
                cost+=192*excess*excess+std::max(0,live-43);
            }
            return cost;
        };
        auto emit=[&](){
            Mask needed=required;
            std::vector<int> uses(values.size());
            for(auto it=order.rbegin();it!=order.rend();++it) if(needed[*it]) {
                auto r=rules[choice[*it]]; needed[r.a]=needed[r.b]=1; ++uses[r.a]; ++uses[r.b];
            }
            std::vector<int> location(values.size(),-1),free;
            Mask busy(48);
            for(int i=0;i<int(seeds.size());++i) if(uses[seeds[i]] && location[seeds[i]]<0) {
                location[seeds[i]]=i;busy[i]=1;
            }
            for(int i=0;i<48;++i) if(!busy[i]) free.push_back(i);
            std::vector<Instruction> code;
            for(int x:order) if(needed[x]) {
                auto r=rules[choice[x]];
                int a=location[r.a],b=location[r.b];
                if(--uses[r.a]==0) free.push_back(a);
                if(--uses[r.b]==0) free.push_back(b);
                if(free.empty()) return false;
                int d=free.back();free.pop_back();location[x]=d;
                code.push_back({r.op,a,b,d});if(!uses[x]) free.push_back(d);
            }
            result=std::move(code); return true;
        };
        auto bestOrder=order,bestChoice=choice;
        int bestCost=score(),current=bestCost;
        std::uniform_real_distribution<double> unit(0,1);
        auto window=[&](int first,int count) {
            struct Step {std::uint64_t mask; int cost,live,parent,added;};
            std::vector<std::uint64_t> consumers(values.size()),deps(count);
            Mask later(values.size());
            for(int i=first;i<n;++i) {
                auto r=rules[choice[order[i]]];
                if(i>=first+count) later[r.a]=later[r.b]=1;
                else {
                    auto bit=std::uint64_t(1)<<(i-first);
                    for(int a:{r.a,r.b}) {
                        consumers[a]|=bit;
                        if(pos[a]>=first) deps[i-first]|=std::uint64_t(1)<<(pos[a]-first);
                    }
                }
            }
            int initial=0;
            for(int v=0;v<int(values.size());++v) if(pos[v]<first && (later[v]||consumers[v])) ++initial;
            std::vector<std::vector<Step>> layers(1,{{0,0,initial,-1,-1}});
            for(int depth=0;depth<count;++depth) {
                std::unordered_map<std::uint64_t,int> indices;
                std::vector<Step> next;
                auto &previous=layers.back();
                for(int p=0;p<int(previous.size());++p) {
                    auto s=previous[p];
                    for(int k=0;k<count;++k) {
                        auto bit=std::uint64_t(1)<<k;
                        if((s.mask&bit) || (deps[k]&~s.mask)) continue;
                        auto mask=s.mask|bit;
                        int v=order[first+k];
                        auto r=rules[choice[v]];
                        int peak=s.live+1;
                        if(!later[r.a] && !(consumers[r.a]&~mask)) --peak;
                        if(r.b!=r.a && !later[r.b] && !(consumers[r.b]&~mask)) --peak;
                        int excess=std::max(0,peak-48);
                        int cost=s.cost+192*excess*excess+std::max(0,peak-43);
                        int live=peak-int(!later[v] && !(consumers[v]&~mask));
                        auto [it,fresh]=indices.emplace(mask,int(next.size()));
                        if(fresh) next.push_back({mask,cost,live,p,k});
                        else if(cost<next[it->second].cost) next[it->second]={mask,cost,live,p,k};
                    }
                }
                int width=192;
                if(int(next.size())>width) {
                    auto key=[](const Step &v){return v.cost+8*v.live;};
                    std::nth_element(next.begin(),next.begin()+width,next.end(),[&](auto a,auto b){return key(a)<key(b);});
                    next.resize(width);
                }
                layers.push_back(std::move(next));
            }
            auto orig=order;
            int parent=0;
            for(int d=count;d>0;--d) {
                auto s=layers[d][parent];
                order[first+d-1]=orig[first+s.added]; parent=s.parent;
            }
            positions();
        };
        auto relocate=[&](int from,int to){
            if(from<to) std::rotate(order.begin()+from,order.begin()+from+1,order.begin()+to+1);
            if(to<from) std::rotate(order.begin()+to,order.begin()+from,order.begin()+from+1);
            positions();
        };
        if(emit()) return;
        for(int rep=0;rep<100 && std::chrono::steady_clock::now()<deadline;++rep) {
            score();
            std::vector<int> hot;
            int live=1;
            for(int i=0;i<n;++i) {live+=diff[i];if(live>48) hot.push_back(i);}
            if(hot.empty()) break;
            int count=std::min(n,32+int(rng()%32));
            int first=std::clamp(hot[rng()%hot.size()]-int(rng()%count),0,n-count);
            window(first,count);
            int cost=score();
            if(emit()) return;
            if(cost<=bestCost) {bestCost=cost;bestOrder=order;}
            else {order=bestOrder;positions();}
        }
        for(int it=0;std::chrono::steady_clock::now()<deadline;++it) {
            if(it%25000==0) {order=bestOrder;choice=bestChoice;positions();current=score();}
            int i=rng()%n,x=order[i],old=choice[x],j=i;
            std::vector<int> oldOrder;
            int mode=rng()%12;
            if(mode==0 && i>0) {
                // Pull an entire dependency chain earlier in one move.
                int lo=rng()%i;
                Mask chain(values.size());
                chain[x]=1;
                for(int k=i;k>=lo;--k) if(chain[order[k]]) {
                    auto r=rules[choice[order[k]]]; chain[r.a]=chain[r.b]=1;
                }
                oldOrder=order;
                std::stable_partition(order.begin()+lo,order.begin()+i+1,[&](int y){return chain[y];});
                positions();
            } else if(mode==1 && i+1<n) {
                // Delay a helper and its dependent instructions together.
                int hi=i+1+rng()%(n-i-1);
                Mask chain(values.size());chain[x]=1;
                for(int k=i+1;k<=hi;++k) {
                    auto r=rules[choice[order[k]]];
                    chain[order[k]]=chain[r.a]||chain[r.b];
                }
                oldOrder=order;
                std::stable_partition(order.begin()+i,order.begin()+hi+1,[&](int y){return !chain[y];});
                positions();
            } else if(mode<=4) {
                int alt=recipes[x][rng()%recipes[x].size()];
                auto s=rules[alt];
                if(alt==old || (!seed[s.a] && pos[s.a]<0) || (!seed[s.b] && pos[s.b]<0)
                   || pos[s.a]>=i || pos[s.b]>=i) continue;
                choice[x]=alt;
            } else {
                auto r=rules[old];
                int lo=std::max(pos[r.a],pos[r.b])+1,hi=n-1;
                for(int k=i+1;k<n;++k) {
                    auto s=rules[choice[order[k]]];
                    if(s.a==x || s.b==x) {hi=k-1;break;}
                }
                j=lo+rng()%(hi-lo+1);
                if(i==j) continue;
                relocate(i,j);
            }
            int cost=score();
            if(fits && emit()) return;
            if(cost<bestCost) {
                bestCost=cost;bestOrder=order;bestChoice=choice;
                if(emit()) return;
            }
            double temp=0.8+19.*(1.-double(it%25000)/25000.);
            if(cost<=current || unit(rng)<std::exp(double(current-cost)/temp)) current=cost;
            else {
                choice[x]=old;
                if(!oldOrder.empty()) {order.swap(oldOrder);positions();}
                else relocate(j,i);
            }
        }
    }
    void run(double seconds) {
        auto start=std::chrono::steady_clock::now();
        auto until=start+std::chrono::duration<double>(seconds);
        auto cutDeadline=start+std::chrono::duration<double>(seconds*0.6);
        auto cuts=banks.front().order;
        std::shuffle(cuts.begin(),cuts.end(),rng);
        // Remove one old helper and measure the part of the graph it blocks.
        // Only test new helpers that can unlock at least one blocked value.
        for(int x:cuts) {
            if(std::chrono::steady_clock::now()>=cutDeadline) break;
            if(required[x] || !bestSet[x]) continue;
            Mask cut=bestSet,reached;
            cut[x]=0;
            if(close(cut,&reached)) continue;
            for(int h:bridges) {
                bool useful=false;
                for(int ri:users[h]) {
                    const auto &r=rules[ri];
                    if(cut[r.out] && !reached[r.out] &&
                       (r.a==h || reached[r.a]) && (r.b==h || reached[r.b])) {useful=true;break;}
                }
                if(!useful) continue;
                Mask trial=cut; trial[h]=1;
                close(trial);
            }
        }
        Mask current=bestSet;
        for(int round=0;std::chrono::steady_clock::now()<until;++round) {
            if(round%9==0) current=bestSet;
            const Bank &donor=banks[rng()%banks.size()];
            Mask trial=current;
            if(round%7==0) for(int x:donor.order) trial[x]=1;
            else if(!donor.order.empty()) {
                int count=1+int(rng()%12);
                for(int i=0;i<count;++i) {
                    int x=donor.order[rng()%donor.order.size()];
                    const auto &r=donor.producer.at(x);
                    addBranch(donor,r.a,trial); addBranch(donor,r.b,trial);
                    trial[x]=1;
                }
            }
            if(!bridges.empty()) {
                int additions=1+rng()%16;
                for(int i=0;i<additions;++i) trial[bridges[rng()%bridges.size()]]=1;
            }
            std::vector<int> removable;
            for(int x=0;x<int(trial.size());++x) if(trial[x] && !required[x]) removable.push_back(x);
            std::shuffle(removable.begin(),removable.end(),rng);
            for(int x:removable) {
                if(!trial[x]) continue;
                Mask reduced=trial; reduced[x]=0;
                if(close(reduced)) trial.swap(reduced);
            }
            close(trial);
            if(std::count(trial.begin(),trial.end(),1)<=std::count(bestSet.begin(),bestSet.end(),1)+2) current=trial;
        }
    }
};
int main(int argc,char **argv) {
    if(argc!=3 && argc!=4) return 2;
    Search search;
    search.rng.seed(std::stoul(argv[2]));
    search.read();
    if(argc==4 && std::string(argv[3])=="--pack") {
        search.buildRules(); search.pack(std::stod(argv[1]));
    } else {
        search.addBridges(); search.buildRules(); search.run(std::stod(argv[1]));
    }
    for(auto r:search.result) std::cout<<r.op<<" "<<r.a<<" "<<r.b<<" "<<r.d<<"\n";
}
