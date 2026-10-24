#ifndef MDA_UTIL_H
#define MDA_UTIL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

double mda_util_now(void);
char *mda_util_path_suffix(const char *path, const char *suffix);
const char *mda_util_arg(uint32_t argc, char **argv, uint32_t i, const char *fallback);
const char *mda_util_verdict(bool ok);
uint32_t mda_util_max_u32(uint32_t a, uint32_t b);
uint32_t mda_util_min_u32(uint32_t a, uint32_t b);
size_t mda_util_min_size(size_t a, size_t b);
size_t mda_util_max_size(size_t a, size_t b);

#endif
