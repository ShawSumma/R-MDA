#ifndef MDA_CHECK_QWEN_H
#define MDA_CHECK_QWEN_H

#include <stdint.h>

#define MDA_CHECK_QWEN_LEN 256
#define MDA_CHECK_QWEN_TEXT 65536
#define MDA_CHECK_QWEN_OUT "models/qwen_check.bin"

typedef struct mda_check_qwen_sink_t mda_check_qwen_sink_t;

struct mda_check_qwen_sink_t {
    float *logits;
    uint32_t vocab;
};

#endif
