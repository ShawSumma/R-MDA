#ifndef MDA_STEP_H
#define MDA_STEP_H

#include <stdbool.h>
#include <stdint.h>

#include "vkc.h"
#include "model.h"
#include "fwd.h"

#define MDA_STEP_ADAM_B1 0.9f
#define MDA_STEP_ADAM_B2 0.999f
#define MDA_STEP_GMUL_GROWTH 2000
#define MDA_STEP_MUON_ITERS 5
#define MDA_STEP_MUON_MOMENTUM 0.95f
#define MDA_STEP_MUON_RMS 0.2f
#define MDA_STEP_MUON_A 3.4445f
#define MDA_STEP_MUON_B -4.7750f
#define MDA_STEP_MUON_C 2.0315f
#define MDA_STEP_MUON_PER_LAYER 6

typedef uint32_t mda_step_muon_phase_t;
typedef struct mda_step_muon_t mda_step_muon_t;
typedef struct mda_step_trainer_t mda_step_trainer_t;

enum {
    MDA_STEP_MUON_PHASE_MOMENTUM,
    MDA_STEP_MUON_PHASE_NORM,
    MDA_STEP_MUON_PHASE_NORMALIZE,
    MDA_STEP_MUON_PHASE_GRAM,
    MDA_STEP_MUON_PHASE_POLY,
    MDA_STEP_MUON_PHASE_ITERATE,
    MDA_STEP_MUON_PHASE_UPDATE,
};

struct mda_step_muon_t {
    mda_vkc_buf_t x;
    mda_vkc_buf_t x16[2];
    mda_vkc_buf_t a;
    mda_vkc_buf_t a16;
    mda_vkc_buf_t b16;
    mda_vkc_buf_t red;
    mda_model_param_t *p;
    uint32_t rows;
    uint32_t cols;
    uint32_t side;
    uint32_t groups;
    bool wide;
};

struct mda_step_trainer_t {
    mda_model_t *m;
    mda_step_muon_t *mats;
    mda_fwd_scratch_t sc;
    mda_vkc_buf_t partials;
    mda_vkc_cmd_t first;
    mda_vkc_cmd_t more;
    mda_vkc_cmd_t opt;
    double loss;
    double ce_sum;
    double masked;
    double kd_sum;
    double kd_n;
    double kd_student_bits;
    double kd_teacher_bits;
    double kd_bytes;
    double kd_teacher_bytes;
    float gnorm;
    uint32_t accum;
    uint32_t n_mats;
    bool muon;
    bool skipped;
};

void mda_step_opt_rec(mda_step_trainer_t *tr, float clip);
mda_step_trainer_t mda_step_init(mda_model_t *m, float clip, uint32_t accum, bool muon);
float mda_step_grad(mda_step_trainer_t *tr);
void mda_step_begin(mda_step_trainer_t *tr, float lr, uint32_t step);
void mda_step_micro(mda_step_trainer_t *tr, uint32_t i);
void mda_step_print_nonfinite(mda_step_trainer_t *tr);
float mda_step_train(mda_step_trainer_t *tr, float lr, uint32_t step);

#endif
