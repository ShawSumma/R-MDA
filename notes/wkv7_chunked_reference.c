// chunkref3.c -- test whether the per-column decay can be absorbed so the intra-chunk
// coupling becomes a scalar [C x C] triangular system (the DeltaNet UT transform). Key
// trick: factor S[i,j] = Wcum[t][j] * Shat[i,j] (pull the cumulative column decay out),
// so Shat evolves WITHOUT the per-step column decay, leaving scalar inter-step coupling.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define DH 64
#define T 64
#define C 16

static float K[T][DH], V[T][DH], R[T][DH], W[T][DH], Asc[T];
static float Oser[T][DH], Och[T][DH];
static float frand(void) { return (rand() % 2000 - 1000) / 1000.0f; }

static void serial(void) {
    float S[DH][DH]; memset(S, 0, sizeof(S));
    for (int t = 0; t < T; t = t + 1) {
        float Sk[DH];
        for (int j = 0; j < DH; j = j + 1) { float s=0; for (int i=0;i<DH;i++) s+=S[i][j]*K[t][i]; Sk[j]=s; }
        float a = Asc[t];
        for (int i=0;i<DH;i++) for (int j=0;j<DH;j++)
            S[i][j] = W[t][j]*S[i][j] - a*K[t][i]*Sk[j] + a*K[t][i]*V[t][j];
        for (int j=0;j<DH;j++){ float o=0; for(int i=0;i<DH;i++) o+=R[t][i]*S[i][j]; Oser[t][j]=o; }
    }
}

// chunked with column-decay factored out within each chunk.
// Within a chunk starting at carried state S0: write S_u[i,j] = cw[u][j] * H_u[i,j] + (carry part).
// Actually we keep S0 explicit and track intra corrections H (decay-normalized).
// cw[u][j] = prod_{s=0..u} W[c0+s][j]  (cumulative column decay within chunk, inclusive of u).
// Define for the intra corrections a decay-normalized accumulation so the k-dot coupling
// becomes scalar. We verify numerically that this reproduces the serial output.
static void chunked(void) {
    float S0[DH][DH]; memset(S0, 0, sizeof(S0));   // carried chunk-start state
    for (int c0 = 0; c0 < T; c0 = c0 + C) {
        // cumulative column decay within the chunk: cw[u][j], and cw_excl (before step u)
        float cw[C][DH];
        for (int j=0;j<DH;j++){ float p=1.0f; for(int u=0;u<C;u++){ p*=W[c0+u][j]; cw[u][j]=p; } }
        // u_s vectors (the rank-1 column factors) [C][DH], and the scalar coupling A[u][s].
        float U[C][DH];
        // carry readout part: for step u, carry_u[j] = sum_i S0[i,j]*decay(0..u-1 on j? )*k_u[i]
        // S_{u-1} from carry = (prod W over 0..u-1 on column j) * S0[i,j]. decay BEFORE step u:
        //   cwex[u][j] = prod_{s=0..u-1} W[c0+s][j]  (exclusive)
        for (int u = 0; u < C; u = u + 1) {
            int t = c0 + u;
            float a = Asc[t];
            // carry contribution to Sk_u[j]: (cwex on j) * sum_i S0[i,j]*k_u[i]
            float carrySk[DH];
            for (int j=0;j<DH;j++){
                float cwex = (u==0)?1.0f:cw[u-1][j];
                float s=0; for(int i=0;i<DH;i++) s+=S0[i][j]*K[t][i];
                carrySk[j] = cwex * s;
            }
            // intra contribution to Sk_u[j]: sum_{s<u} (k_s . k_u) * U[s][j] * (decay from s.. to u-1 on j)
            // correction at step s added k_s ⊗ U[s]; by step u-1 it's decayed by prod_{r=s+1..u-1} W[r][j]
            //   = cwex[u][j]/cw[s][j]  (since cw[s] includes step s's own decay which the correction
            //     does NOT get -- the correction enters AFTER decay at step s). Careful:
            //   S_s gets the correction at step s (post-decay-of-step-s). So from s to u-1 it is
            //   decayed by prod_{r=s+1..u-1} W[r][j] = cwex[u][j] / cw[s][j].
            float intraSk[DH];
            for (int j=0;j<DH;j++){
                float cwex = (u==0)?1.0f:cw[u-1][j];
                float s=0;
                for (int sp=0; sp<u; sp++){
                    float kk=0; for(int i=0;i<DH;i++) kk+=K[c0+sp][i]*K[t][i];
                    s += kk * U[sp][j] * (cwex / cw[sp][j]);
                }
                intraSk[j]=s;
            }
            for (int j=0;j<DH;j++) U[u][j] = a*(V[t][j] - carrySk[j] - intraSk[j]);
            // readout out_t[j] = r_t^T S_u. S_u = decayed carry + intra (incl this step's correction)
            for (int j=0;j<DH;j++){
                float cwj = cw[u][j];               // decay 0..u inclusive
                float rc=0; for(int i=0;i<DH;i++) rc += R[t][i]*S0[i][j];
                float carry = cwj * rc;
                float intra=0;
                for (int sp=0; sp<=u; sp++){
                    float rk=0; for(int i=0;i<DH;i++) rk += R[t][i]*K[c0+sp][i];
                    // correction s decayed from s to u on column j = cw[u][j]/cw[sp][j]
                    intra += rk * U[sp][j] * (cw[u][j]/cw[sp][j]);
                }
                Och[t][j] = carry + intra;
            }
        }
        // update carry: S_{c0+C-1}[i,j] = cw[C-1][j]*S0[i,j] + sum_s k_s[i]*U[s][j]*(cw[C-1][j]/cw[s][j])
        float Snew[DH][DH];
        for (int i=0;i<DH;i++) for (int j=0;j<DH;j++){
            float v = cw[C-1][j]*S0[i][j];
            for (int sp=0;sp<C;sp++) v += K[c0+sp][i]*U[sp][j]*(cw[C-1][j]/cw[sp][j]);
            Snew[i][j]=v;
        }
        memcpy(S0, Snew, sizeof(S0));
    }
}

int main(void){
    srand(7);
    for (int t=0;t<T;t++){ for(int d=0;d<DH;d++){K[t][d]=frand();V[t][d]=frand();R[t][d]=frand();W[t][d]=(rand()%1000)/1100.0f;} Asc[t]=(rand()%1000)/2000.0f; }
    serial(); chunked();
    double maxd=0,sum=0;
    for(int t=0;t<T;t++) for(int j=0;j<DH;j++){ double e=fabs(Oser[t][j]-Och[t][j]); sum+=e; if(e>maxd)maxd=e; }
    printf("serial vs chunked(factored) max diff=%.3e mean=%.3e\n", maxd, sum/(T*DH));
    return 0;
}
