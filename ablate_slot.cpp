#include "arithmetic_search.hpp"
#include <algorithm>
#include <array>
#include <set>
#include <unordered_set>

struct Ins { char op; int a,b,d; };
struct Slot {
    std::vector<N> reg;
    std::vector<int> writable;
    std::vector<Ins> path,answer;
    int dest; N limit,target; Op banned;
    int work=0; bool capped=false;
    bool allowed(Op r) const {
        if(r.op!=banned.op)return true;
        if(r.a==banned.a && r.b==banned.b)return false;
        return !(r.op=='+' || r.op=='*') || r.a!=banned.b || r.b!=banned.a;
    }
    using Map=std::unordered_map<N,int>;
    Map index() const {
        Map result;
        for(int i=0;i<int(reg.size());++i)if(reg[i])result.emplace(reg[i],i);
        return result;
    }
    bool make(N v,const Map &idx,int d,Ins &out) const {
        for(auto [x,i]:idx){
            bool ok=false;
            inverse(x,v,limit,[&](N other,Op r){
                auto it=idx.find(other);
                if(!ok && it!=idx.end() && allowed(r)){
                    out={r.op,idx.at(r.a),idx.at(r.b),d};ok=true;
                }
            });
            if(ok)return true;
        }
        return false;
    }
    bool last2(const Map &idx){
        // The last operation uses a snapshot operand, or the new value twice.
        auto test=[&](N mid,Op last){
            if(!allowed(last))return false;
            Ins first;
            if(!make(mid,idx,0,first))return false;
            for(int d:writable){
                N old=reg[d];reg[d]=mid;
                auto loc=[&](N v){
                    if(v==mid)return d;
                    auto it=idx.find(v);
                    if(it!=idx.end() && it->second!=d)return it->second;
                    for(int j=0;j<int(reg.size());++j)if(reg[j]==v)return j;
                    return -1;
                };
                int a=loc(last.a),b=loc(last.b);
                reg[d]=old;
                if(a<0 || b<0)continue;
                first.d=d;
                answer=path;answer.push_back(first);answer.push_back({last.op,a,b,dest});
                return true;
            }
            return false;
        };
        for(auto [x,i]:idx){
            bool ok=false;
            inverse(x,target,limit,[&](N mid,Op r){if(!ok)ok=test(mid,r);});
            if(ok)return true;
        }
        if(target%2==0 && test(target/2,{'+',target/2,target/2}))return true;
        N r=std::sqrt((long double)target);
        for(N i=std::max<N>(1,r-1);i<=r+1;++i)
            if(i*i==target && test(i,{'*',i,i}))return true;
        // For 1 include mid/mid even when mid was absent initially.
        if(target==1){
            for(auto [x,i]:idx)for(auto [y,j]:idx){
                bool ok=false;operations(x,y,limit,[&](N v,Op){if(!ok)ok=test(v,{'/',v,v});});
                if(ok)return true;
            }
        }
        return false;
    }
    bool last3(const Map &idx){
        std::unordered_map<N,Op> one;
        for(auto [x,i]:idx)for(auto [y,j]:idx)if(x>=y)
            operations(x,y,limit,[&](N z,Op r){if(allowed(r))one.emplace(z,r);});
        std::unordered_map<N,std::vector<Op>> finish;
        auto add=[&](N z,Op r){if(allowed(r))finish[z].push_back(r);};
        for(auto [v,i]:idx)inverse(v,target,limit,add);
        if(target%2==0)add(target/2,{'+',target/2,target/2});
        N root=std::sqrt((long double)target);
        for(N r=std::max<N>(1,root-1);r<=root+1;++r)
            if(r*r==target)add(r,{'*',r,r});
        auto schedule=[&](N a,Op first,N b,Op second,Op last){
            if(!allowed(second) || !allowed(last))return false;
            auto loc=[&](N v)->int{
                for(int i=0;i<int(reg.size());++i)if(reg[i]==v)return i;
                return -1;
            };
            for(int d:writable){
                N before=reg[d];reg[d]=a;
                int i=loc(second.a),j=loc(second.b);
                if(i>=0 && j>=0)for(int e:writable){
                    N was=reg[e];reg[e]=b;
                    int x=loc(last.a),y=loc(last.b);
                    reg[e]=was;
                    if(x<0 || y<0)continue;
                    reg[d]=before;
                    answer=path;answer.push_back({first.op,idx.at(first.a),idx.at(first.b),d});
                    answer.push_back({second.op,i,j,e});answer.push_back({last.op,x,y,dest});return true;
                }
                reg[d]=before;
            }
            return false;
        };
        for(auto [a,first]:one){
            bool found=false;
            inverse(a,target,limit,[&](N b,Op last){
                auto it=one.find(b);
                if(!found && it!=one.end())found=schedule(a,first,b,it->second,last);
                if(found)return;
                inverse(a,b,limit,[&](N x,Op mid){
                    if(!found && (idx.count(x)||x==a))found=schedule(a,first,b,mid,last);
                });
            });
            if(found)return true;
            auto middle=[&](N b,Op mid){
                if(found || !allowed(mid))return;
                if(target==1 && schedule(a,first,b,mid,{'/',b,b})){found=true;return;}
                auto it=finish.find(b);
                if(it!=finish.end())for(Op last:it->second)
                    if(schedule(a,first,b,mid,last)){found=true;break;}
            };
            for(auto [x,i]:idx){operations(a,x,limit,middle);if(found)return true;}
            operations(a,a,limit,middle);if(found)return true;
        }
        return false;
    }
    bool multiply_detour(){
        if(banned.op!='*')return false;
        Map idx=index();
        for(auto a:{banned.a,banned.b}){
            N b=target/a;
            if(a<2 || !idx.count(a) || !idx.count(b))continue;
            Op expr[4]={{'/',a,a},{'-',a,1},{'*',a-1,b},{'+',(a-1)*b,b}};
            N value[4]={1,a-1,(a-1)*b,target};
            auto emit=[&](auto&& self,int k)->bool {
                auto available=index();
                auto r=expr[k];
                if(!allowed(r) || !available.count(r.a) || !available.count(r.b))return false;
                int x=available[r.a],y=available[r.b];
                if(k==3){
                    answer=path;answer.push_back({r.op,x,y,dest});return true;
                }
                for(int d:writable){
                    N was=reg[d];reg[d]=value[k];path.push_back({r.op,x,y,d});
                    bool hit=self(self,k+1);path.pop_back();reg[d]=was;
                    if(hit)return true;
                }
                return false;
            };
            if(emit(emit,0))return true;
        }
        return false;
    }
    bool search(int left){
        if(++work>1500){capped=true;return false;}
        Map idx=index();
        if(left==1){
            Ins last;
            if(!make(target,idx,dest,last))return false;
            answer=path;answer.push_back(last);return true;
        }
        if(left==2)return last2(idx);
        if(left==3)return last3(idx);
        if(left==4 && multiply_detour())return true;
        std::unordered_set<N> tried;
        for(auto [a,i]:idx)for(auto [b,j]:idx){
            if(a<b)continue;
            bool found=false;
            operations(a,b,limit,[&](N v,Op r){
                if(found || capped || !allowed(r) || !tried.insert(v).second)return;
                std::unordered_set<N> overwritten;
                for(int d:writable){
                    N old=reg[d];
                    if(!overwritten.insert(old).second)continue;
                    path.push_back({r.op,idx.at(r.a),idx.at(r.b),d});
                    reg[d]=v;found=search(left-1);reg[d]=old;path.pop_back();
                    if(found || capped)return;
                }
            });
            if(found)return true;
            if(capped)return false;
        }
        return false;
    }
};
int main(){
    int id,n,m,depth;
    while(std::cin>>id>>n>>m>>depth){
        Slot s;std::cin>>s.limit>>s.target>>s.dest>>s.banned.op>>s.banned.a>>s.banned.b;
        s.reg.resize(n);for(auto &x:s.reg)std::cin>>x;
        s.writable.resize(m);for(int &x:s.writable)std::cin>>x;
        int excluded=0;
        for(int d=1;d<=depth;++d){
            s.work=0;
            if(s.search(d) || s.capped)break;
            excluded=d;
        }
        std::cout<<id<<' '<<excluded<<' '<<s.answer.size();
        for(auto i:s.answer)std::cout<<' '<<i.op<<' '<<i.a<<' '<<i.b<<' '<<i.d;
        std::cout<<'\n';
    }
}
