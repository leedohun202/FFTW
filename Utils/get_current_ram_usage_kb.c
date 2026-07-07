#include "radar_utils.h"

int get_current_ram_usage_kb() {
    FILE* file = fopen("/proc/self/status", "r");
    char line[128]; int vmrss = 0;
    if (file != NULL) {
        while (fgets(line, 128, file) != NULL) {
            if (strncmp(line, "VmRSS:", 6) == 0) { sscanf(line, "VmRSS: %d kB", &vmrss); break; }
        }
        fclose(file);
    }
    return vmrss;
}