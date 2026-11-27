#include "arithmetic_search.hpp"
#include <algorithm>
#include <tuple>
#include <unordered_set>
#include "four_step_leaf.hpp"
int main(){
    int n;N limit,t;
    while(std::cin>>n>>limit>>t){
        std::vector<N> v(n);for(N& x:v)std::cin>>x;
        four_step_leaf(v,limit,t,[](N a,N b,N c,N t,Op first,Op second,Op third,Op last){
            std::cout<<a<<' '<<b<<' '<<c<<' '<<t;
            for(Op r:{first,second,third,last})std::cout<<' '<<r.op<<' '<<r.a<<' '<<r.b;
            std::cout<<'\n';
        },[]{return false;});
    }
}
