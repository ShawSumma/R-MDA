#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "vkc.h"

const char *mda_probe_coopmat_component(VkComponentTypeKHR t) {
    switch (t) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR:
        return "f16";
    case VK_COMPONENT_TYPE_FLOAT32_KHR:
        return "f32";
    case VK_COMPONENT_TYPE_SINT8_KHR:
        return "s8";
    case VK_COMPONENT_TYPE_SINT32_KHR:
        return "s32";
    case VK_COMPONENT_TYPE_UINT8_KHR:
        return "u8";
    default:
        return "?";
    }
}

const char *mda_probe_coopmat_scope(VkScopeKHR s) {
    switch (s) {
    case VK_SCOPE_SUBGROUP_KHR:
        return "subgroup";
    case VK_SCOPE_WORKGROUP_KHR:
        return "workgroup";
    default:
        return "?";
    }
}

void mda_probe_coopmat_list(mda_vkc_t *c, PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR get_props) {
    uint32_t count = 0;
    get_props(c->pd, &count, NULL);
    VkCooperativeMatrixPropertiesKHR *props = malloc(count * sizeof(VkCooperativeMatrixPropertiesKHR));
    for (uint32_t i = 0; i < count; i += 1) {
        props[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        props[i].pNext = NULL;
    }
    get_props(c->pd, &count, props);
    printf("cooperative-matrix configs: %u\n", count);
    printf("%3s %3s %3s %-4s %-4s %-4s %-6s %s\n", "M", "N", "K", "A", "B", "C", "Result", "scope");
    for (uint32_t i = 0; i < count; i += 1) {
        VkCooperativeMatrixPropertiesKHR *p = &props[i];
        printf(
            "%3u %3u %3u %-4s %-4s %-4s %-6s %s\n",
            p->MSize,
            p->NSize,
            p->KSize,
            mda_probe_coopmat_component(p->AType),
            mda_probe_coopmat_component(p->BType),
            mda_probe_coopmat_component(p->CType),
            mda_probe_coopmat_component(p->ResultType),
            mda_probe_coopmat_scope(p->scope)
        );
    }
    free(props);
}

bool mda_probe_coopmat_device(mda_vkc_t *c) {
    PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR get_props = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(c->inst, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (get_props == NULL) {
        fprintf(stderr, "cooperative matrix not supported by the driver\n");
        return false;
    } else {
        mda_probe_coopmat_list(c, get_props);
        return true;
    }
}

bool mda_probe_coopmat_main(void) {
    mda_vkc_t c = mda_vkc_init();
    if (c.fail) {
        return false;
    } else {
        return mda_probe_coopmat_device(&c);
    }
}

int main(void) {
    bool ok = mda_probe_coopmat_main();
    if (ok) {
        return 0;
    } else {
        return 1;
    }
}
