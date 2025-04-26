SM_ARCH ?= 80

all: BF16/src/main.cu FP16/src/main.cu FP32/src/main.cu FP64/src/main.cu
	nvcc -o BF16/src/main -arch=sm_$(SM_ARCH) BF16/src/main.cu -I/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include -L/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64 -lcublas -lcurand
	nvcc -o FP16/src/main -arch=sm_$(SM_ARCH) FP16/src/main.cu -I/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include -L/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64 -lcublas -lcurand
	nvcc -o FP32/src/main -arch=sm_$(SM_ARCH) FP32/src/main.cu -I/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include -L/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64 -lcublas -lcurand
	nvcc -o FP64/src/main -arch=sm_$(SM_ARCH) FP64/src/main.cu -I/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/include -L/home1/apps/nvidia/Linux_aarch64/24.7/math_libs/lib64 -lcublas -lcurand

clean:
	rm -f BF16/src/main FP16/src/main FP32/src/main FP64/src/main