#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <csignal>
#include <chrono>
#include <nvml.h>

// Global flag to stop measurements on CTRL+C
static volatile sig_atomic_t g_stopFlag = 0;
void signalHandler(int signum) {
    g_stopFlag = 1;
}

int initializeNVML() {
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "NVML init failed: %s\n", nvmlErrorString(result));
        return 0;
    }
    return 1;
}

int cleanupNVML() {
    nvmlReturn_t result = nvmlShutdown();
    if (result != NVML_SUCCESS) {
        fprintf(stderr, "NVML shutdown failed: %s\n", nvmlErrorString(result));
        return 0;
    }
    return 1;
}

int acquireDeviceHandle(int devIndex, nvmlDevice_t* device) {
    nvmlReturn_t res;
    unsigned int totalDevices = 0;
    res = nvmlDeviceGetCount(&totalDevices);
    if (res != NVML_SUCCESS) {
        fprintf(stderr, "Error obtaining device count: %s\n", nvmlErrorString(res));
        return 0;
    }
    if (devIndex >= static_cast<int>(totalDevices)) {
        fprintf(stderr, "Invalid device index: %d (total devices: %u)\n", devIndex, totalDevices);
        return 0;
    }
    res = nvmlDeviceGetHandleByIndex(devIndex, device);
    if (res != NVML_SUCCESS) {
        fprintf(stderr, "Error getting handle for device %d: %s\n", devIndex, nvmlErrorString(res));
        return 0;
    }
    char nameBuffer[NVML_DEVICE_NAME_BUFFER_SIZE];
    res = nvmlDeviceGetName(*device, nameBuffer, NVML_DEVICE_NAME_BUFFER_SIZE);
    if (res != NVML_SUCCESS) {
        fprintf(stderr, "Error getting name for device %d: %s\n", devIndex, nvmlErrorString(res));
        return 0;
    }
    return 1;
}

void displayHelp() {
    printf("Usage: dumpGpuPower [options]\n");
    printf("Options:\n");
    printf(" -h                 Display help information\n");
    printf(" -c                 Output in CSV format (default)\n");
    printf(" -d <device id>     Select the NVIDIA device (default 0)\n");
    printf(" -r <rate(ms)>      Sampling interval in milliseconds (default 100)\n");
    printf(" -n <sample count>  Number of samples to capture (-1 for continuous until CTRL+C)\n");
    printf(" -p                 Include PSU info\n");
    printf(" -a <process name>  Profile a specific process (stop when process terminates)\n");
    printf(" -t <temp(C)>       Stop profiling if GPU temperature exceeds this cutoff\n");
    printf("\nPress CTRL+C to terminate measurements.\n");
}

int parseArguments(int argc, char** argv, int* csvMode, int* devId, int* intervalMs, int* numSamples, int* psuFlag, char** processName, int* tempCutoff) {
    int opt;
    opterr = 0;
    while ((opt = getopt(argc, argv, "hc:d:r:n:pa:t:")) != -1) {
        switch (opt) {
            case 'h':
                displayHelp();
                return 2;
            case 'c':
                *csvMode = 1;
                break;
            case 'd':
                *devId = atoi(optarg);
                break;
            case 'r':
                *intervalMs = atoi(optarg);
                break;
            case 'n':
                *numSamples = atoi(optarg);
                break;
            case 'p':
                *psuFlag = 1;
                break;
            case 'a':
                *processName = optarg;
                break;
            case 't':
                *tempCutoff = atoi(optarg);
                break;
            default:
                fprintf(stderr, "Unknown option encountered.\n");
                return 0;
        }
    }
    return 1;
}

int monitorGPU(int csv, int devIdx, nvmlDevice_t* device, int sampleInterval, int numSamples, int printPSU, int monitoredPid, int tempThreshold) {
    nvmlReturn_t res;
    unsigned int powerMilliWatts = 0;
    unsigned int temperature = 0;
    nvmlUtilization_t utilization;
    unsigned int coreClock = 0, memClock = 0;
    unsigned long long energy = 0;

    // Print CSV header
    printf("sample,power(W),gpu_util(%),core_clock(MHz),mem_clock(MHz),timestamp_ns,temp(C),energy(mJ)\n");

    if (numSamples == -1) {
        unsigned int sampleCounter = 0;
        while (!g_stopFlag) {
            res = nvmlDeviceGetUtilizationRates(*device, &utilization);
            res = nvmlDeviceGetTemperature(*device, NVML_TEMPERATURE_GPU, &temperature);
            res = nvmlDeviceGetPowerUsage(*device, &powerMilliWatts);
            res = nvmlDeviceGetClockInfo(*device, NVML_CLOCK_GRAPHICS, &coreClock);
            res = nvmlDeviceGetClockInfo(*device, NVML_CLOCK_MEM, &memClock);
            res = nvmlDeviceGetTotalEnergyConsumption(*device, &energy);

            printf("%u,%.4f,%u,%u,%u,%ld,%u,%llu\n",
                   sampleCounter,
                   (double)powerMilliWatts / 1000.0,
                   utilization.gpu,
                   coreClock,
                   memClock,
                   std::chrono::high_resolution_clock::now().time_since_epoch().count(),
                   temperature,
                   energy);

            sampleCounter++;
            usleep(sampleInterval * 1000);
        }
    } else {
        for (int i = 0; i < numSamples; ++i) {
            if (g_stopFlag)
                break;
            res = nvmlDeviceGetUtilizationRates(*device, &utilization);
            res = nvmlDeviceGetTemperature(*device, NVML_TEMPERATURE_GPU, &temperature);
            res = nvmlDeviceGetPowerUsage(*device, &powerMilliWatts);
            res = nvmlDeviceGetClockInfo(*device, NVML_CLOCK_GRAPHICS, &coreClock);
            res = nvmlDeviceGetClockInfo(*device, NVML_CLOCK_MEM, &memClock);
            res = nvmlDeviceGetTotalEnergyConsumption(*device, &energy);

            printf("%d,%.4f,%u,%u,%u,%ld,%u,%llu\n",
                   i,
                   (double)powerMilliWatts / 1000.0,
                   utilization.gpu,
                   coreClock,
                   memClock,
                   std::chrono::high_resolution_clock::now().time_since_epoch().count(),
                   temperature,
                   energy);

            usleep(sampleInterval * 1000);
        }
    }
    return 1;
}

int main(int argc, char** argv) {
    nvmlDevice_t gpuDevice;
    int result = 0;

    // Default configuration values
    int deviceId = 0;
    int intervalMs = 100;
    int sampleCount = -1;
    int csvOutput = 1;  // CSV output enabled by default
    int printPSUInfo = 0;
    char* procName = NULL;
    int monitoredPid = 0;
    int temperatureCutoff = 0;

    // Setup CTRL+C handler
    signal(SIGINT, signalHandler);

    result = parseArguments(argc, argv, &csvOutput, &deviceId, &intervalMs, &sampleCount, &printPSUInfo, &procName, &temperatureCutoff);
    if (result == 0)
        return EXIT_FAILURE;
    if (result == 2)
        return EXIT_SUCCESS;

    if (!initializeNVML())
        return EXIT_FAILURE;

    if (!acquireDeviceHandle(deviceId, &gpuDevice)) {
        cleanupNVML();
        return EXIT_FAILURE;
    }

    // If a process name is given, attempt to retrieve its PID (optional)
    if (procName != NULL) {
        char command[256];
        FILE* pipe;
        char shortName[16];
        int len = strlen(procName);
        int copyLen = (len < 15 ? len : 15);
        strncpy(shortName, procName, copyLen);
        shortName[copyLen] = '\0';
        sprintf(command, "ps | grep '%s' | awk '{print $1}'", shortName);
        pipe = popen(command, "r");
        if (pipe == NULL) {
            fprintf(stderr, "Failed to run process lookup command.\n");
            cleanupNVML();
            return EXIT_FAILURE;
        }
        char pidBuffer[8];
        if (fgets(pidBuffer, sizeof(pidBuffer), pipe))
            monitoredPid = atoi(pidBuffer);
        else {
            pclose(pipe);
            fprintf(stderr, "Process '%s' not found. Ensure it is running before profiling.\n", shortName);
            cleanupNVML();
            return EXIT_FAILURE;
        }
        pclose(pipe);
    }

    // Begin GPU monitoring; output is printed to stdout in CSV format.
    monitorGPU(csvOutput, deviceId, &gpuDevice, intervalMs, sampleCount, printPSUInfo, monitoredPid, temperatureCutoff);

    cleanupNVML();
    return EXIT_SUCCESS;
}
