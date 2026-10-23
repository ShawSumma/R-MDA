#ifndef MDA_VKC_H
#define MDA_VKC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#define MDA_VKC_CK(c, x) mda_vkc_check((c), (x), __FILE__, __LINE__)
#define MDA_VKC_MAX_BINDINGS 24
#define MDA_VKC_MAX_LABELS 64
#define MDA_VKC_MAX_QUERIES 16384
#define MDA_VKC_NO_FAMILY 0xffffffffu

typedef struct mda_vkc_t mda_vkc_t;
typedef struct mda_vkc_buf_t mda_vkc_buf_t;
typedef struct mda_vkc_kernel_t mda_vkc_kernel_t;
typedef struct mda_vkc_cmd_t mda_vkc_cmd_t;
typedef struct mda_vkc_spv_t mda_vkc_spv_t;

struct mda_vkc_t {
    VkInstance inst;
    VkPhysicalDevice pd;
    VkDevice dev;
    VkQueue q;
    VkCommandPool cpool;
    VkDescriptorPool dpool;
    VkCommandBuffer rec;
    VkDescriptorPool rec_pool;
    VkQueryPool qpool;
    const char **qname;
    const char *prof_name[MDA_VKC_MAX_LABELS];
    double prof_ms[MDA_VKC_MAX_LABELS];
    VkDeviceSize dev_bytes;
    uint32_t prof_count[MDA_VKC_MAX_LABELS];
    uint32_t max_groups[3];
    uint32_t qfi;
    uint32_t mem_dev;
    uint32_t mem_cached;
    uint32_t nq;
    uint32_t prof_n;
    float ts_period;
    bool recording;
    bool nobar;
    bool prof;
    bool prof_rec;
    bool fail;
};

struct mda_vkc_buf_t {
    VkBuffer buf;
    VkDeviceMemory mem;
    void *ptr;
    VkDeviceSize off;
    VkDeviceSize sz;
};

struct mda_vkc_kernel_t {
    VkPipeline pipe;
    VkPipelineLayout pl;
    VkDescriptorSetLayout dsl;
    const char *name;
    uint32_t nbuf;
    uint32_t push_sz;
};

struct mda_vkc_cmd_t {
    VkCommandBuffer cb;
    VkDescriptorPool pool;
};

struct mda_vkc_spv_t {
    uint32_t *b;
    size_t len;
};

void mda_vkc_check(mda_vkc_t *c, VkResult r, const char *file, uint32_t line);
mda_vkc_t mda_vkc_init(void);
mda_vkc_buf_t mda_vkc_alloc_type(mda_vkc_t *c, VkDeviceSize sz, uint32_t type);
mda_vkc_buf_t mda_vkc_alloc(mda_vkc_t *c, VkDeviceSize sz);
mda_vkc_buf_t mda_vkc_alloc_cached(mda_vkc_t *c, VkDeviceSize sz);
void mda_vkc_free(mda_vkc_t *c, mda_vkc_buf_t *b);
mda_vkc_buf_t mda_vkc_view(mda_vkc_buf_t *b, VkDeviceSize off, VkDeviceSize sz);
mda_vkc_kernel_t mda_vkc_load(mda_vkc_t *c, const char *spv, uint32_t nbuf, uint32_t push_sz);
void mda_vkc_begin(mda_vkc_t *c);
void mda_vkc_prof_begin(mda_vkc_t *c);
void mda_vkc_prof_report(mda_vkc_t *c, uint32_t batches);
void mda_vkc_flush(mda_vkc_t *c);
mda_vkc_cmd_t mda_vkc_record(mda_vkc_t *c);
void mda_vkc_end(mda_vkc_t *c);
void mda_vkc_submit(mda_vkc_t *c, mda_vkc_cmd_t *cmd);
void mda_vkc_run(mda_vkc_t *c, mda_vkc_kernel_t *k, mda_vkc_buf_t **bufs, uint32_t gx, uint32_t gy, uint32_t gz, void *push);
void mda_vkc_copy(mda_vkc_t *c, mda_vkc_buf_t *dst, mda_vkc_buf_t *src, VkDeviceSize sz);
void mda_vkc_fill(mda_vkc_t *c, mda_vkc_buf_t *b, uint32_t word);
void *mda_vkc_read(mda_vkc_t *c, mda_vkc_buf_t *b);
void mda_vkc_write(mda_vkc_t *c, mda_vkc_buf_t *b, const void *src);
void mda_vkc_set_nobar(mda_vkc_t *c, bool on);

#endif
