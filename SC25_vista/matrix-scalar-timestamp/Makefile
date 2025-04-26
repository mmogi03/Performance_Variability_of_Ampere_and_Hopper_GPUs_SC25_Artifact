SM_ARCH ?= 80

all: FP32/src/main.cu FP64/src/main.cu
	nvcc -o FP32/src/main -arch=sm_$(SM_ARCH) FP32/src/main.cu -I/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include -L/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64 -lcublas -lcurand
	nvcc -o FP64/src/main -arch=sm_$(SM_ARCH) FP64/src/main.cu -I/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include -L/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64 -lcublas -lcurand

clean:
	rm -f FP32/src/main FP64/src/main