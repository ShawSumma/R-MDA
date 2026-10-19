#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "util.h"

double mda_util_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}

char *mda_util_path_suffix(const char *path, const char *suffix) {
    size_t n = strlen(path);
    size_t k = strlen(suffix);
    char *s = malloc(n + k + 1);
    memcpy(s, path, n);
    memcpy(s + n, suffix, k + 1);
    return s;
}

const char *mda_util_arg(uint32_t argc, char **argv, uint32_t i, const char *fallback) {
    if (i < argc) {
        return argv[i];
    } else {
        return fallback;
    }
}

const char *mda_util_verdict(bool ok) {
    if (ok) {
        return "ok";
    } else {
        return "FAIL";
    }
}

uint32_t mda_util_max_u32(uint32_t a, uint32_t b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

uint32_t mda_util_min_u32(uint32_t a, uint32_t b) {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}

size_t mda_util_min_size(size_t a, size_t b) {
    if (a < b) {
        return a;
    } else {
        return b;
    }
}
