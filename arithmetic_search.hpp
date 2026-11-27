#include <cmath>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>
using N = int64_t;
struct Op { char op; N a,b; };

// Yield all positive in-range binary results for an unordered operand pair.
template<class F> void operations(N a,N b,N limit,F emit) {
    if(a<b) std::swap(a,b);
    if(a<=limit-b) emit(a+b,Op{'+',a,b});
    if(a>b) emit(a-b,Op{'-',a,b});
    if(a<=limit/b) emit(a*b,Op{'*',a,b});
    if(a%b==0) emit(a/b,Op{'/',a,b});
}

// Yield each possible other operand in a final instruction using x to make t.
template<class F> void inverse(N x,N t,N limit,F emit) {
    if(t>x) emit(t-x,Op{'+',x,t-x});
    if(x>t) emit(x-t,Op{'-',x,x-t});
    if(t<=limit-x) emit(t+x,Op{'-',t+x,x});
    if(t%x==0) emit(t/x,Op{'*',x,t/x});
    if(x%t==0) emit(x/t,Op{'/',x,x/t});
    if(t<=limit/x) emit(t*x,Op{'/',t*x,x});
}


