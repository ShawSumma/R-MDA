// vkc.h -- minimal Vulkan compute harness for the RWKV-7 + diffusion trainer.
// One device, host-visible buffers (the 8060S is UMA, so host-visible is fast),
// shaders loaded from build/*.spv and dispatched by handle. No tinygrad, no GGML.
#ifndef VKC_H
#define VKC_H

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VKC_CK(x) vkc_ck((x), __FILE__, __LINE__)

typedef struct vkc_buf_t vkc_buf_t;
typedef struct vkc_t vkc_t;
typedef struct vkc_kernel_t vkc_kernel_t;
typedef struct vkc_read_spv_t vkc_read_spv_t;

struct vkc_buf_t {
    VkBuffer buf;
    VkDeviceMemory mem;
    void *ptr;
    VkDeviceSize sz;
};

struct vkc_t {
    VkInstance inst;
    VkPhysicalDevice pd;
    VkDevice dev;
    VkQueue q;
    uint32_t qfi;
    VkCommandPool cpool;
    VkDescriptorPool dpool;
    // batching: when recording is set, vkc_run records dispatches (separated by a
    // compute-to-compute barrier) into rec instead of submitting one at a time. A
    // vkc_flush submits the whole batch once and waits. This removes the per-op
    // submit + queue-wait-idle round trip, which dominates with many small kernels.
    VkCommandBuffer rec;
    int recording;
    // when set, recorded dispatches skip the trailing barrier -- use only for runs of
    // mutually independent dispatches (e.g. the two bidirectional wkv passes).
    int nobar;
};

struct vkc_kernel_t {
    VkPipeline pipe;
    VkPipelineLayout pl;
    VkDescriptorSetLayout dsl;
    uint32_t nbuf;
    uint32_t push_sz;
};

struct vkc_read_spv_t {
    uint32_t *b;
    size_t len;
};

static inline void vkc_ck(VkResult vkc_r, const char *file, size_t line) {
    if (vkc_r) {
        fprintf(stderr, "vk err %d %s:%d\n", vkc_r, __FILE__, __LINE__);
        exit(1);
    }
}

static uint32_t vkc_memtype(VkPhysicalDevice pd, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) != 0 && (mp.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    fprintf(stderr, "no memtype\n");
    exit(1);
}

static vkc_t vkc_init(void) {
    vkc_t c = (vkc_t) {
        .inst = 0,
    };

    VkApplicationInfo app = (VkApplicationInfo) {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo ici = (VkInstanceCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    VKC_CK(vkCreateInstance(&ici, 0, &c.inst));

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(c.inst, &n, 0);

    VkPhysicalDevice *pds = malloc(n * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(c.inst, &n, pds);

    c.pd = pds[0];

    free(pds);

    VkPhysicalDeviceProperties pr;
    vkGetPhysicalDeviceProperties(c.pd, &pr);
    fprintf(stderr, "vkc GPU: %s\n", pr.deviceName);

    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nqf, 0);
    VkQueueFamilyProperties *qf = malloc(nqf * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(c.pd, &nqf, qf);
    c.qfi = 0;
    for (uint32_t i = 0; i < nqf; ++i) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            c.qfi = i;
            break;
        }
    }
    free(qf);

    float pri = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = c.qfi,
        .queueCount = 1,
        .pQueuePriorities = &pri,
    };
    // feature chain: atomic-float (existing) + cooperative matrix + fp16 shader
    // arithmetic + 16-bit storage, so the matrix-core matmul can load fp16 buffers.
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        .cooperativeMatrix = VK_TRUE,
    };
    VkPhysicalDeviceShaderFloat16Int8Features f16_arith = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        .pNext = &coopmat,
        .shaderFloat16 = VK_TRUE,
    };
    VkPhysicalDevice16BitStorageFeatures storage16 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
        .pNext = &f16_arith,
        .storageBuffer16BitAccess = VK_TRUE,
        .uniformAndStorageBuffer16BitAccess = VK_TRUE,
    };
    VkPhysicalDeviceSubgroupSizeControlFeatures subgroup_ctl = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
        .pNext = &storage16,
        .subgroupSizeControl = VK_TRUE,
        .computeFullSubgroups = VK_TRUE,
    };
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
        .pNext = &subgroup_ctl,
        .shaderBufferFloat32AtomicAdd = VK_TRUE,
    };
    const char *dev_ext[2] = {
        VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
        VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &atomic_float,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 2,
        .ppEnabledExtensionNames = dev_ext,
    };
    VKC_CK(vkCreateDevice(c.pd, &dci, 0, &c.dev));
    vkGetDeviceQueue(c.dev, c.qfi, 0, &c.q);

    VkCommandPoolCreateInfo cpi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = c.qfi,
    };
    VKC_CK(vkCreateCommandPool(c.dev, &cpi, 0, &c.cpool));

    VkDescriptorPoolSize ps = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 8192,
    };
    VkDescriptorPoolCreateInfo dpi = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 2048,
        .poolSizeCount = 1,
        .pPoolSizes = &ps,
    };
    VKC_CK(vkCreateDescriptorPool(c.dev, &dpi, 0, &c.dpool));
    return c;
}

static vkc_buf_t vkc_alloc(vkc_t *c, VkDeviceSize sz) {
    vkc_buf_t b = {
        .sz = sz,
    };
    VkBufferCreateInfo bi = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = sz,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VKC_CK(vkCreateBuffer(c->dev, &bi, 0, &b.buf));

    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(c->dev, b.buf, &mr);
    VkMemoryAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex = vkc_memtype(c->pd, mr.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
    };
    VKC_CK(vkAllocateMemory(c->dev, &ai, 0, &b.mem));
    VKC_CK(vkBindBufferMemory(c->dev, b.buf, b.mem, 0));
    VKC_CK(vkMapMemory(c->dev, b.mem, 0, sz, 0, &b.ptr));
    return b;
}

static vkc_read_spv_t vkc_read_spv(const char *p) {
    FILE *f = fopen(p, "rb");
    if (f == NULL) {
        perror(p);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *b = malloc(n);
    fread(b, 1, n, f);
    fclose(f);
    return (vkc_read_spv_t) {
        .b = b,
        .len = n,
    };
}

static vkc_kernel_t vkc_load(vkc_t *c, const char *spv, uint32_t nbuf, uint32_t push_sz) {
    vkc_kernel_t k = (vkc_kernel_t) {
        .nbuf = nbuf,
        .push_sz = push_sz,
    };
    vkc_read_spv_t sp = vkc_read_spv(spv);
    VkShaderModuleCreateInfo smi = (VkShaderModuleCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sp.len,
        .pCode = sp.b,
    };

    VkShaderModule sm;
    VKC_CK(vkCreateShaderModule(c->dev, &smi, 0, &sm));

    VkDescriptorSetLayoutBinding *b = calloc(nbuf, sizeof(VkDescriptorSetLayoutBinding));
    for (uint32_t i = 0; i < nbuf; ++i) {
        b[i] = (VkDescriptorSetLayoutBinding) {
            .binding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }
    VkDescriptorSetLayoutCreateInfo dli = (VkDescriptorSetLayoutCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = nbuf,
        .pBindings = b,
    };
    VKC_CK(vkCreateDescriptorSetLayout(c->dev, &dli, 0, &k.dsl));
    free(b);

    VkPushConstantRange pcr = (VkPushConstantRange) {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = push_sz != 0 ? push_sz : 4,
    };
    VkPipelineLayoutCreateInfo pli = (VkPipelineLayoutCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &k.dsl,
        .pushConstantRangeCount = push_sz != 0 ? 1 : 0,
        .pPushConstantRanges = push_sz != 0? &pcr : 0,
    };
    VKC_CK(vkCreatePipelineLayout(c->dev, &pli, 0, &k.pl));

    // pin the subgroup to 32 lanes (wave32). RDNA3 WMMA / cooperative-matrix runs in
    // wave32; the device defaults to wave64, which breaks multi-subgroup kernels that
    // assume 32-lane waves. Requiring 32 makes gl_SubgroupID counts deterministic.
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .requiredSubgroupSize = 32,
    };
    VkComputePipelineCreateInfo cpi = (VkComputePipelineCreateInfo) {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = (VkPipelineShaderStageCreateInfo) {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = &rss,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sm,
            .pName = "main",
        },
        .layout = k.pl,
    };
    VKC_CK(vkCreateComputePipelines(c->dev, 0, 1, &cpi, 0, &k.pipe));
    vkDestroyShaderModule(c->dev, sm, 0);
    return k;
}

static void vkc_run(vkc_t *c, vkc_kernel_t *k, vkc_buf_t **bufs, uint32_t gx, uint32_t gy, uint32_t gz, void *push) {
    VkDescriptorSetAllocateInfo dsa = (VkDescriptorSetAllocateInfo) {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = c->dpool,
        .descriptorSetCount = 1,
        .pSetLayouts = &k->dsl,
    };
    VkDescriptorSet ds;
    VKC_CK(vkAllocateDescriptorSets(c->dev, &dsa, &ds));

    VkWriteDescriptorSet *w = calloc(k->nbuf, sizeof(VkWriteDescriptorSet));
    VkDescriptorBufferInfo *bi = calloc(k->nbuf, sizeof(VkDescriptorBufferInfo));
    for (uint32_t i = 0; i < k->nbuf; ++i) {
        bi[i] = (VkDescriptorBufferInfo) {
            .buffer = bufs[i]->buf,
            .range = bufs[i]->sz,
        };
        w[i] = (VkWriteDescriptorSet) {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ds,
            .dstBinding = i,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &bi[i],
        };
    }
    vkUpdateDescriptorSets(c->dev, k->nbuf, w, 0, 0);
    free(w);
    free(bi);

    // batched path: record into the shared command buffer, with a compute-to-compute
    // barrier so the next (dependent) dispatch observes this one's writes. The set is
    // not freed here -- the whole descriptor pool is reset at flush. No submit/wait.
    if (c->recording) {
        vkCmdBindPipeline(c->rec, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipe);
        vkCmdBindDescriptorSets(c->rec, VK_PIPELINE_BIND_POINT_COMPUTE, k->pl, 0, 1, &ds, 0, 0);
        if (k->push_sz != 0) {
            vkCmdPushConstants(c->rec, k->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, k->push_sz, push);
        }
        vkCmdDispatch(c->rec, gx, gy, gz);
        if (c->nobar) {
            return;
        }
        VkMemoryBarrier mb = (VkMemoryBarrier) {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        };
        vkCmdPipelineBarrier(
            c->rec,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            &mb,
            0,
            0,
            0,
            0);
        return;
    }

    VkCommandBufferAllocateInfo cbi = (VkCommandBufferAllocateInfo) {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = c->cpool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    VkCommandBuffer cb;
    VKC_CK(vkAllocateCommandBuffers(c->dev, &cbi, &cb));

    VkCommandBufferBeginInfo bb = (VkCommandBufferBeginInfo) {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VKC_CK(vkBeginCommandBuffer(cb, &bb));

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, k->pl, 0, 1, &ds, 0, 0);
    if (k->push_sz != 0) {
        vkCmdPushConstants(cb, k->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, k->push_sz, push);
    }
    vkCmdDispatch(cb, gx, gy, gz);
    VKC_CK(vkEndCommandBuffer(cb));

    VkSubmitInfo si = (VkSubmitInfo) {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    VKC_CK(vkQueueSubmit(c->q, 1, &si, 0));
    VKC_CK(vkQueueWaitIdle(c->q));
    vkFreeCommandBuffers(c->dev, c->cpool, 1, &cb);
    vkFreeDescriptorSets(c->dev, c->dpool, 1, &ds);
}

// mark subsequent recorded dispatches as barrier-free (for independent runs); pass 0
// to restore the default per-dispatch barrier.
static void vkc_set_nobar(vkc_t *c, int on) {
    c->nobar = on;
}

// begin recording a batch of dispatches into one command buffer.
static void vkc_begin(vkc_t *c) {
    c->nobar = 0;
    VkCommandBufferAllocateInfo cbi = (VkCommandBufferAllocateInfo) {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = c->cpool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VKC_CK(vkAllocateCommandBuffers(c->dev, &cbi, &c->rec));
    VkCommandBufferBeginInfo bb = (VkCommandBufferBeginInfo) {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VKC_CK(vkBeginCommandBuffer(c->rec, &bb));
    c->recording = 1;
}

// submit the recorded batch, wait for it, and reset the command buffer and the
// descriptor pool (freeing every set allocated during the batch).
static void vkc_flush(vkc_t *c) {
    if (c->recording == 0) {
        return;
    }
    VKC_CK(vkEndCommandBuffer(c->rec));
    VkSubmitInfo si = (VkSubmitInfo) {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &c->rec,
    };
    VKC_CK(vkQueueSubmit(c->q, 1, &si, 0));
    VKC_CK(vkQueueWaitIdle(c->q));
    vkFreeCommandBuffers(c->dev, c->cpool, 1, &c->rec);
    VKC_CK(vkResetDescriptorPool(c->dev, c->dpool, 0));
    c->recording = 0;
}

#endif
