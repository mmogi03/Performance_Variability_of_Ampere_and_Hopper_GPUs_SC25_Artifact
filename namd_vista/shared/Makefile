TARGETS = dumpGpuPower

all: $(TARGETS)

dumpGpuPower: dumpGpuPower.cpp
	g++ -o $@ $< -I/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include \
	    -L/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64 -lnvidia-ml
		
clean:
	rm -rf $(TARGETS)