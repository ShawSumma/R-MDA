#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vkc.h"
#include "util.h"
#include "sort.h"

const char *const mda_vkc_dev_ext[2] = {
    VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
    VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME,
};

void mda_vkc_check(mda_vkc_t *c, VkResult r, const char *file, uint32_t line) {
    if (r != VK_SUCCESS) {
        fprintf(stderr, "vk err %d %s:%u\n", r, file, line);
        c->fail = true;
    }
}

int32_t mda_vkc_memtype(VkPhysicalDevice pd, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t i = 0;
    while (i < mp.memoryTypeCount && (mp.memoryTypes[i].propertyFlags & want) != want) {
        i += 1;
    }
    if (i == mp.memoryTypeCount) {
        return -1;
    } else {
        return (int32_t)i;
    }
}

bool mda_vkc_has_ext(VkPhysicalDevice pd, const char *name) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(pd, 0, &n, 0);
    VkExtensionProperties *e = malloc(n * sizeof(VkExtensionProperties));
    vkEnumerateDeviceExtensionProperties(pd, 0, &n, e);
    uint32_t i = 0;
    while (i < n && strcmp(e[i].extensionName, name) != 0) {
        i += 1;
    }
    free(e);
    return i < n;
}

bool mda_vkc_has_tile(VkInstance inst, VkPhysicalDevice pd) {
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR get_props = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    uint32_t n = 0;
    get_props(pd, &n, 0);
    VkCooperativeMatrixPropertiesKHR *cp = calloc(n, sizeof(VkCooperativeMatrixPropertiesKHR));
    for (uint32_t i = 0; i < n; i += 1) {
        cp[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    }
    get_props(pd, &n, cp);
    uint32_t i = 0;
    while (i < n && (cp[i].MSize != 16 || cp[i].NSize != 16 || cp[i].KSize != 16 || cp[i].AType != VK_COMPONENT_TYPE_FLOAT16_KHR || cp[i].BType != VK_COMPONENT_TYPE_FLOAT16_KHR || cp[i].CType != VK_COMPONENT_TYPE_FLOAT32_KHR || cp[i].ResultType != VK_COMPONENT_TYPE_FLOAT32_KHR || cp[i].scope != VK_SCOPE_SUBGROUP_KHR)) {
        i += 1;
    }
    free(cp);
    return i < n;
}

uint32_t mda_vkc_compute_family(VkPhysicalDevice pd) {
    uint32_t n = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, 0);
    VkQueueFamilyProperties *qf = malloc(n * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &n, qf);
    uint32_t i = 0;
    while (i < n && (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
        i += 1;
    }
    free(qf);
    if (i == n) {
        return MDA_VKC_NO_FAMILY;
    } else {
        return i;
    }
}

const char *mda_vkc_missing_ext(VkPhysicalDevice pd) {
    uint32_t i = 0;
    while (i < 2 && mda_vkc_has_ext(pd, mda_vkc_dev_ext[i])) {
        i += 1;
    }
    if (i < 2) {
        return mda_vkc_dev_ext[i];
    } else {
        return NULL;
    }
}

const char *mda_vkc_missing_feature(VkInstance inst, VkPhysicalDevice pd) {
    VkPhysicalDeviceVulkan11Properties p11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES,
    };
    VkPhysicalDeviceVulkan13Properties p13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES,
        .pNext = &p11,
    };
    VkPhysicalDeviceProperties2 pr = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &p13,
    };
    vkGetPhysicalDeviceProperties2(pd, &pr);
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
    };
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
        .pNext = &coopmat,
    };
    VkPhysicalDeviceVulkan11Features f11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &atomic_float,
    };
    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &f11,
    };
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &f12,
    };
    VkPhysicalDeviceFeatures2 f = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &f13,
    };
    vkGetPhysicalDeviceFeatures2(pd, &f);
    if (coopmat.cooperativeMatrix == 0 || atomic_float.shaderBufferFloat32AtomicAdd == 0 || f11.storageBuffer16BitAccess == 0 || f12.shaderFloat16 == 0 || f13.subgroupSizeControl == 0) {
        return "missing a required feature";
    } else if (p13.minSubgroupSize > 32 || p13.maxSubgroupSize < 32 || (p13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0) {
        return "no wave32 compute";
    } else if ((p11.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) == 0 || (p11.subgroupSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0) {
        return "no subgroup arithmetic in compute";
    } else if (mda_vkc_has_tile(inst, pd) == false) {
        return "no 16x16x16 f16 cooperative matrix";
    } else if (mda_vkc_compute_family(pd) == MDA_VKC_NO_FAMILY) {
        return "no compute queue";
    } else if (mda_vkc_memtype(pd, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) < 0 || mda_vkc_memtype(pd, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT) < 0) {
        return "no host-visible device-local or host-cached memory";
    } else {
        return NULL;
    }
}

const char *mda_vkc_unsuitable(VkInstance inst, VkPhysicalDevice pd) {
    VkPhysicalDeviceProperties pr;
    vkGetPhysicalDeviceProperties(pd, &pr);
    if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
        return "a CPU device";
    } else if (pr.apiVersion < VK_API_VERSION_1_3) {
        return "no Vulkan 1.3";
    } else if (mda_vkc_missing_ext(pd) != NULL) {
        return mda_vkc_missing_ext(pd);
    } else {
        return mda_vkc_missing_feature(inst, pd);
    }
}

int32_t mda_vkc_rank(VkPhysicalDeviceType t) {
    switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return 2;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return 1;
    default:
        return 0;
    }
}

VkPhysicalDevice mda_vkc_pick(VkInstance inst) {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, 0);
    VkPhysicalDevice *pds = malloc(n * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(inst, &n, pds);
    VkPhysicalDevice best = VK_NULL_HANDLE;
    int32_t best_rank = -1;
    for (uint32_t i = 0; i < n; i += 1) {
        VkPhysicalDeviceProperties pr;
        vkGetPhysicalDeviceProperties(pds[i], &pr);
        const char *why = mda_vkc_unsuitable(inst, pds[i]);
        if (why != NULL) {
            fprintf(stderr, "vkc: skipping %s: %s\n", pr.deviceName, why);
        } else if (mda_vkc_rank(pr.deviceType) > best_rank) {
            best_rank = mda_vkc_rank(pr.deviceType);
            best = pds[i];
        }
    }
    free(pds);
    return best;
}

mda_vkc_t mda_vkc_init_pools(mda_vkc_t c) {
    vkGetDeviceQueue(c.dev, c.qfi, 0, &c.q);
    VkCommandPoolCreateInfo cpi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = c.qfi,
    };
    MDA_VKC_CK(&c, vkCreateCommandPool(c.dev, &cpi, 0, &c.cpool));
    VkDescriptorPoolSize ps = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 32768,
    };
    VkDescriptorPoolCreateInfo dpi = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 8192,
        .poolSizeCount = 1,
        .pPoolSizes = &ps,
    };
    MDA_VKC_CK(&c, vkCreateDescriptorPool(c.dev, &dpi, 0, &c.dpool));
    VkQueryPoolCreateInfo qpi = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = MDA_VKC_MAX_QUERIES,
    };
    MDA_VKC_CK(&c, vkCreateQueryPool(c.dev, &qpi, 0, &c.qpool));
    c.qname = malloc(MDA_VKC_MAX_QUERIES * sizeof(const char *));
    return c;
}

mda_vkc_t mda_vkc_init_device(mda_vkc_t c) {
    VkPhysicalDeviceProperties pr;
    vkGetPhysicalDeviceProperties(c.pd, &pr);
    fprintf(stderr, "vkc GPU: %s\n", pr.deviceName);
    for (uint32_t i = 0; i < 3; i += 1) {
        c.max_groups[i] = pr.limits.maxComputeWorkGroupCount[i];
    }
    c.ts_period = pr.limits.timestampPeriod;
    c.mem_dev = (uint32_t)mda_vkc_memtype(c.pd, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    c.mem_cached = (uint32_t)mda_vkc_memtype(c.pd, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
    c.qfi = mda_vkc_compute_family(c.pd);
    float pri = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = c.qfi,
        .queueCount = 1,
        .pQueuePriorities = &pri,
    };
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopmat = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        .cooperativeMatrix = VK_TRUE,
    };
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT,
        .pNext = &coopmat,
        .shaderBufferFloat32AtomicAdd = VK_TRUE,
    };
    VkPhysicalDeviceVulkan11Features f11 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
        .pNext = &atomic_float,
        .storageBuffer16BitAccess = VK_TRUE,
    };
    VkPhysicalDeviceVulkan12Features f12 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &f11,
        .shaderFloat16 = VK_TRUE,
    };
    VkPhysicalDeviceVulkan13Features f13 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = &f12,
        .subgroupSizeControl = VK_TRUE,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &f13,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 2,
        .ppEnabledExtensionNames = mda_vkc_dev_ext,
    };
    MDA_VKC_CK(&c, vkCreateDevice(c.pd, &dci, 0, &c.dev));
    if (c.fail) {
        return c;
    } else {
        return mda_vkc_init_pools(c);
    }
}

mda_vkc_t mda_vkc_init_instance(mda_vkc_t c) {
    c.pd = mda_vkc_pick(c.inst);
    if (c.pd == VK_NULL_HANDLE) {
        fprintf(stderr, "vkc: no suitable Vulkan device\n");
        c.fail = true;
        return c;
    } else {
        return mda_vkc_init_device(c);
    }
}

mda_vkc_t mda_vkc_init(void) {
    mda_vkc_t c = {
        .inst = VK_NULL_HANDLE,
    };
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = VK_API_VERSION_1_3,
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    MDA_VKC_CK(&c, vkCreateInstance(&ici, 0, &c.inst));
    if (c.fail) {
        return c;
    } else {
        return mda_vkc_init_instance(c);
    }
}

mda_vkc_buf_t mda_vkc_bind_memory(mda_vkc_t *c, mda_vkc_buf_t b, uint32_t type) {
    MDA_VKC_CK(c, vkBindBufferMemory(c->dev, b.buf, b.mem, 0));
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(c->pd, &mp);
    if (c->fail == false && (mp.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        MDA_VKC_CK(c, vkMapMemory(c->dev, b.mem, 0, b.sz, 0, &b.ptr));
    }
    return b;
}

mda_vkc_buf_t mda_vkc_alloc_memory(mda_vkc_t *c, mda_vkc_buf_t b, uint32_t type) {
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(c->dev, b.buf, &mr);
    if ((mr.memoryTypeBits & (1u << type)) == 0) {
        fprintf(stderr, "memory type %u can't back a storage buffer\n", type);
        c->fail = true;
        return b;
    } else {
        VkMemoryAllocateInfo ai = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = mr.size,
            .memoryTypeIndex = type,
        };
        MDA_VKC_CK(c, vkAllocateMemory(c->dev, &ai, 0, &b.mem));
        return b;
    }
}

mda_vkc_buf_t mda_vkc_alloc_type(mda_vkc_t *c, VkDeviceSize sz, uint32_t type) {
    mda_vkc_buf_t b = {
        .sz = sz,
    };
    VkBufferCreateInfo bi = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = sz,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    if (c->fail == false) {
        MDA_VKC_CK(c, vkCreateBuffer(c->dev, &bi, 0, &b.buf));
    }
    if (c->fail == false) {
        b = mda_vkc_alloc_memory(c, b, type);
    }
    if (c->fail == false) {
        b = mda_vkc_bind_memory(c, b, type);
    }
    return b;
}

mda_vkc_buf_t mda_vkc_alloc(mda_vkc_t *c, VkDeviceSize sz) {
    return mda_vkc_alloc_type(c, sz, c->mem_dev);
}

mda_vkc_buf_t mda_vkc_alloc_cached(mda_vkc_t *c, VkDeviceSize sz) {
    return mda_vkc_alloc_type(c, sz, c->mem_cached);
}

void mda_vkc_free(mda_vkc_t *c, mda_vkc_buf_t *b) {
    vkDestroyBuffer(c->dev, b->buf, 0);
    vkFreeMemory(c->dev, b->mem, 0);
}

void *mda_vkc_offset(void *ptr, VkDeviceSize off) {
    if (ptr == NULL) {
        return NULL;
    } else {
        return (char *)ptr + off;
    }
}

mda_vkc_buf_t mda_vkc_view(mda_vkc_buf_t *b, VkDeviceSize off, VkDeviceSize sz) {
    return (mda_vkc_buf_t) {
        .buf = b->buf,
        .mem = b->mem,
        .ptr = mda_vkc_offset(b->ptr, off),
        .off = b->off + off,
        .sz = sz,
    };
}

mda_vkc_spv_t mda_vkc_spv_none(void) {
    return (mda_vkc_spv_t) {
        .b = NULL,
        .len = 0,
    };
}

mda_vkc_spv_t mda_vkc_read_file(FILE *f, const char *p) {
    fseek(f, 0, SEEK_END);
    int64_t n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *b = malloc(n);
    size_t got = fread(b, 1, n, f);
    fclose(f);
    if (got != (size_t)n) {
        fprintf(stderr, "%s: short read\n", p);
        free(b);
        return mda_vkc_spv_none();
    } else {
        return (mda_vkc_spv_t) {
            .b = b,
            .len = (size_t)n,
        };
    }
}

mda_vkc_spv_t mda_vkc_read_spv(const char *p) {
    FILE *f = fopen(p, "rb");
    if (f == NULL) {
        perror(p);
        return mda_vkc_spv_none();
    } else {
        return mda_vkc_read_file(f, p);
    }
}

mda_vkc_kernel_t mda_vkc_load_pipeline(mda_vkc_t *c, mda_vkc_kernel_t k, VkShaderModule sm) {
    VkDescriptorSetLayoutBinding *b = calloc(k.nbuf, sizeof(VkDescriptorSetLayoutBinding));
    for (uint32_t i = 0; i < k.nbuf; i += 1) {
        b[i] = (VkDescriptorSetLayoutBinding) {
            .binding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }
    VkDescriptorSetLayoutCreateInfo dli = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = k.nbuf,
        .pBindings = b,
    };
    MDA_VKC_CK(c, vkCreateDescriptorSetLayout(c->dev, &dli, 0, &k.dsl));
    free(b);
    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = mda_util_max_u32(k.push_sz, 4),
    };
    VkPipelineLayoutCreateInfo pli = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &k.dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr,
    };
    MDA_VKC_CK(c, vkCreatePipelineLayout(c->dev, &pli, 0, &k.pl));
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .requiredSubgroupSize = 32,
    };
    VkComputePipelineCreateInfo cpi = {
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
    MDA_VKC_CK(c, vkCreateComputePipelines(c->dev, 0, 1, &cpi, 0, &k.pipe));
    vkDestroyShaderModule(c->dev, sm, 0);
    return k;
}

mda_vkc_kernel_t mda_vkc_load_module(mda_vkc_t *c, mda_vkc_kernel_t k, mda_vkc_spv_t sp) {
    VkShaderModuleCreateInfo smi = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sp.len,
        .pCode = sp.b,
    };
    VkShaderModule sm = VK_NULL_HANDLE;
    MDA_VKC_CK(c, vkCreateShaderModule(c->dev, &smi, 0, &sm));
    free(sp.b);
    if (c->fail) {
        return k;
    } else {
        return mda_vkc_load_pipeline(c, k, sm);
    }
}

mda_vkc_kernel_t mda_vkc_load_spv(mda_vkc_t *c, mda_vkc_kernel_t k) {
    mda_vkc_spv_t sp = mda_vkc_read_spv(k.name);
    if (sp.b == NULL) {
        c->fail = true;
        return k;
    } else {
        return mda_vkc_load_module(c, k, sp);
    }
}

mda_vkc_kernel_t mda_vkc_load(mda_vkc_t *c, const char *spv, uint32_t nbuf, uint32_t push_sz) {
    mda_vkc_kernel_t k = {
        .name = spv,
        .nbuf = nbuf,
        .push_sz = push_sz,
    };
    if (c->fail) {
        return k;
    } else if (nbuf > MDA_VKC_MAX_BINDINGS) {
        fprintf(stderr, "%s: %u bindings, at most %u supported\n", spv, nbuf, MDA_VKC_MAX_BINDINGS);
        c->fail = true;
        return k;
    } else {
        return mda_vkc_load_spv(c, k);
    }
}

void mda_vkc_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
    };
    VkPipelineStageFlags st = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    vkCmdPipelineBarrier(cb, st, st, 0, 1, &mb, 0, 0, 0, 0);
}

void mda_vkc_host_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier mb = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, 0, 0, 0);
}

VkCommandBuffer mda_vkc_cb_begin(mda_vkc_t *c, VkCommandBufferUsageFlags flags) {
    VkCommandBufferAllocateInfo cbi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = c->cpool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBufferBeginInfo bb = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = flags,
    };
    VkCommandBuffer cb = VK_NULL_HANDLE;
    MDA_VKC_CK(c, vkAllocateCommandBuffers(c->dev, &cbi, &cb));
    if (c->fail == false) {
        MDA_VKC_CK(c, vkBeginCommandBuffer(cb, &bb));
    }
    return cb;
}

void mda_vkc_cb_submit(mda_vkc_t *c, VkCommandBuffer cb) {
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    if (c->fail == false) {
        MDA_VKC_CK(c, vkQueueSubmit(c->q, 1, &si, 0));
    }
    if (c->fail == false) {
        MDA_VKC_CK(c, vkQueueWaitIdle(c->q));
    }
}

void mda_vkc_begin(mda_vkc_t *c) {
    if (c->fail == false) {
        c->nobar = false;
        c->rec = mda_vkc_cb_begin(c, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        c->rec_pool = c->dpool;
        c->recording = true;
        c->prof_rec = c->prof;
    }
    if (c->fail == false && c->prof_rec) {
        vkCmdResetQueryPool(c->rec, c->qpool, 0, MDA_VKC_MAX_QUERIES);
        vkCmdWriteTimestamp(c->rec, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, c->qpool, 0);
        c->nq = 1;
    }
}

void mda_vkc_prof_mark(mda_vkc_t *c, const char *name) {
    if (c->prof_rec && c->nq < MDA_VKC_MAX_QUERIES) {
        vkCmdWriteTimestamp(c->rec, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, c->qpool, c->nq);
        c->qname[c->nq] = name;
        c->nq += 1;
    }
}

uint32_t mda_vkc_prof_label(mda_vkc_t *c, const char *name) {
    uint32_t i = 0;
    while (i < c->prof_n && c->prof_name[i] != name) {
        i += 1;
    }
    if (i == c->prof_n && i < MDA_VKC_MAX_LABELS) {
        c->prof_name[i] = name;
        c->prof_ms[i] = 0.0;
        c->prof_count[i] = 0;
        c->prof_n += 1;
    }
    return i;
}

void mda_vkc_prof_collect(mda_vkc_t *c) {
    uint64_t *ts = malloc(c->nq * sizeof(uint64_t));
    MDA_VKC_CK(c, vkGetQueryPoolResults(c->dev, c->qpool, 0, c->nq, c->nq * sizeof(uint64_t), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    for (uint32_t q = 1; q < c->nq && c->fail == false; q += 1) {
        uint32_t i = mda_vkc_prof_label(c, c->qname[q]);
        if (i < MDA_VKC_MAX_LABELS) {
            c->prof_ms[i] += (double)(ts[q] - ts[q - 1]) * c->ts_period / 1e6;
            c->prof_count[i] += 1;
        }
    }
    free(ts);
}

void mda_vkc_prof_begin(mda_vkc_t *c) {
    c->prof = true;
    c->prof_n = 0;
}

void mda_vkc_prof_report(mda_vkc_t *c, uint32_t batches) {
    c->prof = false;
    double total = 0.0;
    mda_sort_item_t order[MDA_VKC_MAX_LABELS];
    for (uint32_t i = 0; i < c->prof_n; i += 1) {
        total += c->prof_ms[i];
        order[i] = (mda_sort_item_t) {
            .key = -c->prof_ms[i],
            .idx = i,
        };
    }
    mda_sort_items(order, c->prof_n);
    printf("%-28s %10s %7s %10s\n", "kernel", "ms/batch", "share", "per batch");
    for (uint32_t r = 0; r < c->prof_n; r += 1) {
        uint32_t i = order[r].idx;
        printf("%-28s %10.3f %6.1f%% %10u\n", c->prof_name[i], c->prof_ms[i] / batches, 100.0 * c->prof_ms[i] / total, c->prof_count[i] / batches);
    }
    printf("%-28s %10.3f\n", "total", total / batches);
}

void mda_vkc_flush(mda_vkc_t *c) {
    if (c->fail == false) {
        mda_vkc_host_barrier(c->rec);
        MDA_VKC_CK(c, vkEndCommandBuffer(c->rec));
        mda_vkc_cb_submit(c, c->rec);
        vkFreeCommandBuffers(c->dev, c->cpool, 1, &c->rec);
        MDA_VKC_CK(c, vkResetDescriptorPool(c->dev, c->dpool, 0));
        c->recording = false;
    }
    if (c->fail == false && c->prof_rec) {
        mda_vkc_prof_collect(c);
        c->prof_rec = false;
    }
}

mda_vkc_cmd_t mda_vkc_record(mda_vkc_t *c) {
    mda_vkc_cmd_t cmd = {
        .cb = VK_NULL_HANDLE,
        .pool = VK_NULL_HANDLE,
    };
    VkDescriptorPoolSize ps = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 32768,
    };
    VkDescriptorPoolCreateInfo dpi = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 8192,
        .poolSizeCount = 1,
        .pPoolSizes = &ps,
    };
    if (c->fail == false) {
        MDA_VKC_CK(c, vkCreateDescriptorPool(c->dev, &dpi, 0, &cmd.pool));
    }
    if (c->fail == false) {
        c->nobar = false;
        c->rec = mda_vkc_cb_begin(c, 0);
        c->rec_pool = cmd.pool;
        c->recording = true;
        c->prof_rec = false;
        cmd.cb = c->rec;
    }
    return cmd;
}

void mda_vkc_end(mda_vkc_t *c) {
    if (c->fail == false) {
        mda_vkc_host_barrier(c->rec);
        MDA_VKC_CK(c, vkEndCommandBuffer(c->rec));
        c->recording = false;
    }
}

void mda_vkc_submit(mda_vkc_t *c, mda_vkc_cmd_t *cmd) {
    mda_vkc_cb_submit(c, cmd->cb);
}

void mda_vkc_bind_set(mda_vkc_t *c, mda_vkc_kernel_t *k, mda_vkc_buf_t **bufs, VkDescriptorSet ds) {
    VkWriteDescriptorSet w[MDA_VKC_MAX_BINDINGS];
    VkDescriptorBufferInfo bi[MDA_VKC_MAX_BINDINGS];
    for (uint32_t i = 0; i < k->nbuf; i += 1) {
        bi[i] = (VkDescriptorBufferInfo) {
            .buffer = bufs[i]->buf,
            .offset = bufs[i]->off,
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
    vkCmdBindPipeline(c->rec, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipe);
    vkCmdBindDescriptorSets(c->rec, VK_PIPELINE_BIND_POINT_COMPUTE, k->pl, 0, 1, &ds, 0, 0);
}

void mda_vkc_dispatch(mda_vkc_t *c, mda_vkc_kernel_t *k, mda_vkc_buf_t **bufs, uint32_t gx, uint32_t gy, uint32_t gz, void *push) {
    VkDescriptorSetAllocateInfo dsa = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = c->rec_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &k->dsl,
    };
    VkDescriptorSet ds = VK_NULL_HANDLE;
    MDA_VKC_CK(c, vkAllocateDescriptorSets(c->dev, &dsa, &ds));
    if (c->fail == false) {
        mda_vkc_bind_set(c, k, bufs, ds);
    }
    if (c->fail == false && k->push_sz != 0) {
        vkCmdPushConstants(c->rec, k->pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, k->push_sz, push);
    }
    if (c->fail == false) {
        vkCmdDispatch(c->rec, gx, gy, gz);
    }
}

void mda_vkc_finish(mda_vkc_t *c, bool oneshot, const char *name) {
    if (oneshot) {
        mda_vkc_flush(c);
    } else if (c->fail == false && c->nobar == false) {
        mda_vkc_barrier(c->rec);
        mda_vkc_prof_mark(c, name);
    }
}

void mda_vkc_run(mda_vkc_t *c, mda_vkc_kernel_t *k, mda_vkc_buf_t **bufs, uint32_t gx, uint32_t gy, uint32_t gz, void *push) {
    bool oneshot = c->recording == false;
    if (c->fail == false && (gx > c->max_groups[0] || gy > c->max_groups[1] || gz > c->max_groups[2])) {
        fprintf(stderr, "vkc: dispatch %ux%ux%u exceeds the device limit\n", gx, gy, gz);
        c->fail = true;
    }
    if (c->fail == false && oneshot) {
        mda_vkc_begin(c);
    }
    if (c->fail == false) {
        mda_vkc_dispatch(c, k, bufs, gx, gy, gz, push);
        mda_vkc_finish(c, oneshot, k->name);
    }
}

void mda_vkc_copy(mda_vkc_t *c, mda_vkc_buf_t *dst, mda_vkc_buf_t *src, VkDeviceSize sz) {
    bool oneshot = c->recording == false;
    VkBufferCopy r = {
        .srcOffset = src->off,
        .dstOffset = dst->off,
        .size = sz,
    };
    if (c->fail == false && oneshot) {
        mda_vkc_begin(c);
    }
    if (c->fail == false) {
        vkCmdCopyBuffer(c->rec, src->buf, dst->buf, 1, &r);
        mda_vkc_finish(c, oneshot, "copy");
    }
}

void mda_vkc_fill(mda_vkc_t *c, mda_vkc_buf_t *b, uint32_t word) {
    bool oneshot = c->recording == false;
    if (c->fail == false && oneshot) {
        mda_vkc_begin(c);
    }
    if (c->fail == false) {
        vkCmdFillBuffer(c->rec, b->buf, b->off, b->sz, word);
        mda_vkc_finish(c, oneshot, "fill");
    }
}

void mda_vkc_set_nobar(mda_vkc_t *c, bool on) {
    c->nobar = on;
}
