TARGETS = dumpGpuPower

all: $(TARGETS)

dumpGpuPower: dumpGpuPower.cpp
	g++ -o $@ $< -I/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/math_libs/include -L/opt/nvidia/hpc_sdk/Linux_aarch64/24.3/math_libs/lib64 -lnvidia-ml
		
clean:
	rm -rf $(TARGETS)


