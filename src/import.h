#ifndef MDA_IMPORT_H
#define MDA_IMPORT_H

#include <stdbool.h>

#include "model.h"

#define MDA_IMPORT_WEIGHTS_PATH "models/g1/g1k-2.9b.safetensors"
#define MDA_IMPORT_OUT_PATH "models/g1k.bin"
#define MDA_IMPORT_LAYER_MAPS 13

typedef struct mda_import_map_t mda_import_map_t;

struct mda_import_map_t {
    const char *suffix;
    mda_model_param_t *p;
    bool linear;
};

#endif
