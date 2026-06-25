#include "model.h"
#include "fwd.h"
#include "step.h"
static model_t*m; static trainer_t tr;
int main(void){
    srand(1); vkc_t c=vkc_init();
    mcfg_t cfg=(mcfg_t){.d=32,.n_layer=1,.n_head=2,.seqlen=12,.batch=2,.ffn_hidden=64};
    m=model_init(&c,cfg); tr=trainer_make(m);
    corpus_t cp=corpus_load("data/wiki.txt");
    make_batch(&cp,cfg.batch,cfg.seqlen,tr.sc.ids.ptr,tr.sc.target.ptr,tr.sc.mask.ptr,tr.sc.weight.ptr);
    train_step(&tr);
    // isolate: wo (after wkv), wv (simplest projection, no sigmoid/norm), wk (norm), ww (sigmoid)
    const char* nm[]={"wo","wv","wr","wk","ww","wa","shift_mix","n1"};
    param_t* ps[]={m->layers[0].wo,m->layers[0].wv,m->layers[0].wr,m->layers[0].wk,m->layers[0].ww,m->layers[0].wa,m->layers[0].shift_mix,m->layers[0].n1};
    double eps=1e-3;
    for(int s=0;s<8;s++){ param_t*p=ps[s]; float*w=p->w.ptr; float*g=p->g.ptr;
        double worst=0; int cnt = p->n<10?(int)p->n:10;
        for(int t=0;t<cnt;t++){ int i=(t*131)%(int)p->n; float o=w[i];
            w[i]=o+(float)eps; double lp=model_forward(m,&tr.sc);
            w[i]=o-(float)eps; double lm=model_forward(m,&tr.sc); w[i]=o;
            double num=(lp-lm)/(2*eps);
            double rel=fabs(num-g[i])/(fabs(num)+fabs(g[i])+1e-4); if(rel>worst)worst=rel; }
        printf("%-10s rel=%.3f %s\n",nm[s],worst,worst<0.1?"ok":"BAD"); }
    return 0;
}
