SM_ARCH ?= 90

all: BF16/src/main.cu
	nvcc -o BF16/src/main -arch=sm_$(SM_ARCH) BF16/src/main.cu -I/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include -L/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64 -lcublas -lcurand
	
clean:
	rm -f BF16/src/main