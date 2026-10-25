#ifndef MDA_UNICODE_H
#define MDA_UNICODE_H

#include <stdbool.h>
#include <stdint.h>

bool mda_unicode_in(const uint32_t (*rs)[2], uint32_t n, uint32_t cp);
bool mda_unicode_letter(uint32_t cp);
bool mda_unicode_number(uint32_t cp);
bool mda_unicode_mark(uint32_t cp);

#endif
