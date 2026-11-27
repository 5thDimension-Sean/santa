#ifdef CPU
#include <stdint.h>
#include <omp.h>
typedef int64_t num;
#define G
#define LAUNCH(name, params, size) extern "C" __declspec(dllexport) void name params { _Pragma("omp parallel for num_threads(8) schedule(static)") for(int i=0;i<(size);++i) {
#define END }}
#else
typedef long num;
#define G __global
#define LAUNCH(name, params, size) __kernel void name params { int i=get_global_id(0); if(i>=(size))return;
#define END }
#endif

// A sorted pair can produce four legal results. Zero denotes an illegal operation.
LAUNCH(make_values,(G const num *v,int n,num lim,G num *out),n*n)
    int x=i/n,y=i%n;
    if(x<=y){
        num a=v[x],b=v[y];
        out[4*i]=a<=lim-b?a+b:0;
        out[4*i+1]=b-a;
        out[4*i+2]=a<=lim/b?a*b:0;
        out[4*i+3]=b%a==0?b/a:0;
    }else{
        for(int j=0;j<4;++j)out[4*i+j]=0;
    }
END

// A root t can be written as one operation on a helper h and a companion.
// Find the cheapest companion among unchanged and one-instruction values.
static inline int find_link(G const num *values,G const int *cost,int n,num h,num t,num lim){
    num options[6]={t>h?t-h:0,h>t?h-t:0,t<=lim-h?t+h:0,
                    t%h==0?t/h:0,h%t==0?h/t:0,t<=lim/h?t*h:0};
    int best=-1,price=3;
    for(int j=0;j<6;++j){
        num want=options[j];
        if(!want)continue;
        int lo=0,hi=n;
        while(lo<hi){int k=(lo+hi)/2;if(values[k]<want)lo=k+1;else hi=k;}
        if(lo<n && values[lo]==want){
            int extra=values[lo]==h?0:cost[lo];
            if(extra<price){price=extra;best=lo*6+j;if(!price)break;}
        }
    }
    return best;
}

LAUNCH(links,(G const num *values,G const int *cost,int n,G const num *roots,int m,num lim,G int *out),n*m)
    out[i]=find_link(values,cost,n,values[i/m],roots[i%m],lim);
END

LAUNCH(frontier_links,(G const num *values,G const int *cost,int n,G const num *helpers,int nh,G const num *roots,int m,num lim,G int *out),nh*m)
    out[i]=find_link(values,cost,n,helpers[i/m],roots[i%m],lim);
END
