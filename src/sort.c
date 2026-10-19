#include "sort.h"

bool mda_sort_less(mda_sort_item_t a, mda_sort_item_t b) {
    if (a.key == b.key) {
        return a.idx < b.idx;
    } else {
        return a.key < b.key;
    }
}

void mda_sort_swap(mda_sort_item_t *v, uint32_t a, uint32_t b) {
    mda_sort_item_t t = v[a];
    v[a] = v[b];
    v[b] = t;
}

uint32_t mda_sort_larger_child(const mda_sort_item_t *v, uint32_t i, uint32_t n) {
    uint32_t left = 2 * i + 1;
    if (left + 1 < n && mda_sort_less(v[left], v[left + 1])) {
        return left + 1;
    } else {
        return left;
    }
}

void mda_sort_sift(mda_sort_item_t *v, uint32_t root, uint32_t n) {
    uint32_t i = root;
    while (2 * i + 1 < n && mda_sort_less(v[i], v[mda_sort_larger_child(v, i, n)])) {
        uint32_t child = mda_sort_larger_child(v, i, n);
        mda_sort_swap(v, i, child);
        i = child;
    }
}

void mda_sort_items(mda_sort_item_t *v, uint32_t n) {
    for (uint32_t i = n / 2; i > 0; i -= 1) {
        mda_sort_sift(v, i - 1, n);
    }
    for (uint32_t end = n; end > 1; end -= 1) {
        mda_sort_swap(v, 0, end - 1);
        mda_sort_sift(v, 0, end - 1);
    }
}
