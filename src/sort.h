#ifndef MDA_SORT_H
#define MDA_SORT_H

#include <stdbool.h>
#include <stdint.h>

typedef struct mda_sort_item_t mda_sort_item_t;

struct mda_sort_item_t {
    double key;
    uint32_t idx;
};

void mda_sort_items(mda_sort_item_t *v, uint32_t n);

#endif
