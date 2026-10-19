#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vkc.h"
#include "pc.h"
#include "util.h"

const char *mda_probe_memory_flag(VkFlags f, VkFlags bit, const char *name) {
    if (f & bit) {
        return name;
    } else {
        return "";
    }
}

double mda_probe_memory_gpu(mda_vkc_t *c, mda_vkc_kernel_t *k, mda_vkc_buf_t *a, mda_vkc_buf_t *b, uint32_t n) {
    mda_pc_vec_t pc = {
        .n = n,
        .op = MDA_PC_VEC_COPY,
    };
    mda_vkc_buf_t *bufs[2] = {
        a,
        b,
    };
    uint32_t rounds = 5;
    uint32_t reps = 10;
    double best = 1e30;
    for (uint32_t r = 0; r < rounds; r += 1) {
        double t0 = mda_util_now();
        mda_vkc_begin(c);
        for (uint32_t i = 0; i < reps; i += 1) {
            mda_vkc_run(c, k, bufs, (n + 255) / 256, 1, 1, &pc);
        }
        mda_vkc_flush(c);
        best = fmin(best, (mda_util_now() - t0) / reps);
    }
    return best;
}

void mda_probe_memory_host(mda_vkc_buf_t *a, char *host, VkDeviceSize sz, uint32_t t) {
    uint32_t rounds = 5;
    double rd = 1e30;
    double wr = 1e30;
    for (uint32_t r = 0; r < rounds; r += 1) {
        double t0 = mda_util_now();
        memcpy(host, a->ptr, sz);
        double t1 = mda_util_now();
        memcpy(a->ptr, host, sz);
        double t2 = mda_util_now();
        rd = fmin(rd, t1 - t0);
        wr = fmin(wr, t2 - t1);
    }
    printf("type %u host read %.1f GB/s, host write %.1f GB/s\n", t, sz / rd / 1e9, sz / wr / 1e9);
}

void mda_probe_memory_type(mda_vkc_t *c, mda_vkc_kernel_t *k, uint32_t t, char *host, VkDeviceSize sz) {
    mda_vkc_buf_t a = mda_vkc_alloc_type(c, sz, t);
    mda_vkc_buf_t b = mda_vkc_alloc_type(c, sz, t);
    if (c->fail == false) {
        double best = mda_probe_memory_gpu(c, k, &a, &b, (uint32_t)(sz / sizeof(float)));
        printf("type %u gpu copy %.1f GB/s\n", t, 2.0 * sz / best / 1e9);
    }
    if (c->fail == false && a.ptr != NULL) {
        mda_probe_memory_host(&a, host, sz, t);
    }
    mda_vkc_free(c, &a);
    mda_vkc_free(c, &b);
}

bool mda_probe_memory_types(mda_vkc_t *c, mda_vkc_kernel_t *k, uint32_t type_bits) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(c->pd, &mp);
    for (uint32_t h = 0; h < mp.memoryHeapCount; h += 1) {
        printf("heap %u: %.1f GiB%s\n", h, mp.memoryHeaps[h].size / 1073741824.0, mda_probe_memory_flag(mp.memoryHeaps[h].flags, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT, " device-local"));
    }
    VkDeviceSize sz = 256u << 20;
    char *host = malloc(sz);
    memset(host, 1, sz);
    for (uint32_t t = 0; t < mp.memoryTypeCount && c->fail == false; t += 1) {
        VkMemoryPropertyFlags f = mp.memoryTypes[t].propertyFlags;
        printf("type %u heap %u:%s%s%s%s\n", t, mp.memoryTypes[t].heapIndex, mda_probe_memory_flag(f, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, " device-local"), mda_probe_memory_flag(f, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, " host-visible"), mda_probe_memory_flag(f, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, " coherent"), mda_probe_memory_flag(f, VK_MEMORY_PROPERTY_HOST_CACHED_BIT, " cached"));
        if ((type_bits & (1u << t)) != 0 && mp.memoryHeaps[mp.memoryTypes[t].heapIndex].size >= 4 * sz) {
            mda_probe_memory_type(c, k, t, host, sz);
        }
    }
    free(host);
    return c->fail == false;
}

bool mda_probe_memory_device(mda_vkc_t *c) {
    mda_vkc_kernel_t k = mda_vkc_load(c, "build/vecops.spv", 2, sizeof(mda_pc_vec_t));
    VkBufferCreateInfo bi = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = 4096,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer probe = VK_NULL_HANDLE;
    if (c->fail == false) {
        MDA_VKC_CK(c, vkCreateBuffer(c->dev, &bi, 0, &probe));
    }
    if (c->fail) {
        return false;
    } else {
        VkMemoryRequirements mr;
        vkGetBufferMemoryRequirements(c->dev, probe, &mr);
        vkDestroyBuffer(c->dev, probe, 0);
        return mda_probe_memory_types(c, &k, mr.memoryTypeBits);
    }
}

bool mda_probe_memory_main(void) {
    mda_vkc_t c = mda_vkc_init();
    if (c.fail) {
        return false;
    } else {
        return mda_probe_memory_device(&c);
    }
}

int main(void) {
    bool ok = mda_probe_memory_main();
    if (ok) {
        return 0;
    } else {
        return 1;
    }
}
