#ifndef MDA_IMPORT_H
#define MDA_IMPORT_H

#include <stdbool.h>

#include "model.h"

#define MDA_IMPORT_WEIGHTS_PATH "models/smollm2.safetensors"
#define MDA_IMPORT_OUT_PATH "models/smollm2.bin"
#define MDA_IMPORT_LAYER_MAPS 9

typedef struct mda_import_map_t mda_import_map_t;

struct mda_import_map_t {
    const char *suffix;
    mda_model_param_t *p;
    bool linear;
};

#endif
