// probe_coopmat.c -- enumerate the cooperative-matrix configurations RADV exposes
// for this GPU, so the fp16 matmul is built to a hardware-native (M, N, K, dtype)
// tile instead of a shape that silently falls back to scalar code.
#include "vkc.h"

static const char *component_name(VkComponentTypeKHR t) {
    if (t == VK_COMPONENT_TYPE_FLOAT16_KHR) {
        return "f16";
    }
    if (t == VK_COMPONENT_TYPE_FLOAT32_KHR) {
        return "f32";
    }
    if (t == VK_COMPONENT_TYPE_SINT8_KHR) {
        return "s8";
    }
    if (t == VK_COMPONENT_TYPE_SINT32_KHR) {
        return "s32";
    }
    if (t == VK_COMPONENT_TYPE_UINT8_KHR) {
        return "u8";
    }
    return "?";
}

static const char *scope_name(VkScopeKHR s) {
    if (s == VK_SCOPE_SUBGROUP_KHR) {
        return "subgroup";
    }
    if (s == VK_SCOPE_WORKGROUP_KHR) {
        return "workgroup";
    }
    return "?";
}

int main(void) {
    vkc_t c = vkc_init();

    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR get_props =
        (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
        vkGetInstanceProcAddr(c.inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (get_props == NULL) {
        fprintf(stderr, "cooperative matrix not supported by the driver\n");
        return 1;
    }

    uint32_t count = 0;
    get_props(c.pd, &count, NULL);
    VkCooperativeMatrixPropertiesKHR *props = malloc(count * sizeof(VkCooperativeMatrixPropertiesKHR));
    for (uint32_t i = 0; i < count; i = i + 1) {
        props[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        props[i].pNext = NULL;
    }
    get_props(c.pd, &count, props);

    printf("cooperative-matrix configs: %u\n", count);
    printf("  MxNxK   A    B    C    Result  scope\n");
    for (uint32_t i = 0; i < count; i = i + 1) {
        VkCooperativeMatrixPropertiesKHR *p = &props[i];
        printf(
            "  %2ux%2ux%2u  %-4s %-4s %-4s %-4s    %s\n",
            p->MSize,
            p->NSize,
            p->KSize,
            component_name(p->AType),
            component_name(p->BType),
            component_name(p->CType),
            component_name(p->ResultType),
            scope_name(p->scope));
    }
    free(props);
    return 0;
}
