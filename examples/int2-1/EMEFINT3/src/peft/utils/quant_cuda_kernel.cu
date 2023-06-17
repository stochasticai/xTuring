#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>  

// atomicAdd for double-precision floating-point numbers on hardware with
// compute capability < 6.0 from:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(
    double* address,
    double val
) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(
      address_as_ull,
      assumed,
      __double_as_longlong(val + __longlong_as_double(assumed))
    );

  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

template <typename scalar_t>
__global__ void VecQuant2MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  		int* __restrict__ zeros,
    int batch,
    int vec_height, 	
    int height,
    int width,
    int zero_width,
    int groupsize
);

template <typename scalar_t>
__global__ void VecQuant3MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height, 	
    int height,
    int width,
    int zero_width,
    int groupsize
);

template <typename scalar_t>
__global__ void VecQuant3DecodeKernel(
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int height,
    int width,
    int zero_width,
    int groupsize
);

__global__ void VecQuant3DecodeHalfKernel(
    const       int* __restrict__ mat,
            float* __restrict__ mul,
    const   half* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int height,
    int width,
    int zero_width,
    int groupsize
);

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height, 	
    int height,
    int width,
    int zero_width,
    int groupsize
);

template <typename scalar_t>
__global__ void VecQuant8MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height, 	
    int height,
    int width,
    int zero_width,
    int groupsize
);

__global__ void transposeCoalesced(int *odata, const int *idata, const int iwidth, const int iheight);

const int BLOCKWIDTH  = 256;
const int BLOCKHEIGHT2 =  16;
const int BLOCKHEIGHT3 =  24;
const int BLOCKHEIGHT4 =  32; 
const int BLOCKHEIGHT8 =  64;

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

void vecquant2matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant2matmul_cuda", ([&] {
      VecQuant2MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, vec_height, height, width, zero_width, groupsize
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant2MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  int b = blockIdx.z;
  int h = BLOCKHEIGHT2 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
  __syncthreads();

  scalar_t res = 0;
  int i = width * h + w;
  int g_h = h * 16;
  int k = 0;
  
  int z_w = w / 16; 
  int z_mod = (w % 16) * 2;

  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp = as_unsigned(mat[i]);
	
    int g = (g_h + k) / groupsize;
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scale * scalar_t((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod & 0x3) + 1);
	
    res += (scale * scalar_t((tmp >> 0) & 0x3) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp >> 2) & 0x3) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp >> 4) & 0x3) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp >> 6) & 0x3) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp >> 8) & 0x3) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp >> 10) & 0x3) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp >> 12) & 0x3) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp >> 14) & 0x3) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp >> 16) & 0x3) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp >> 18) & 0x3) - zero) * blockvec[k + 9];
    res += (scale * scalar_t((tmp >> 20) & 0x3) - zero) * blockvec[k + 10];
    res += (scale * scalar_t((tmp >> 22) & 0x3) - zero) * blockvec[k + 11];
    res += (scale * scalar_t((tmp >> 24) & 0x3) - zero) * blockvec[k + 12];
    res += (scale * scalar_t((tmp >> 26) & 0x3) - zero) * blockvec[k + 13];
    res += (scale * scalar_t((tmp >> 28) & 0x3) - zero) * blockvec[k + 14];
    res += (scale * scalar_t((tmp >> 30) & 0x3) - zero) * blockvec[k + 15];
	
    i += width;
    k += 16;
  }

  atomicAdd(&mul[b * width + w], res);
}

void vecquant3matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant3matmul_cuda", ([&] {
      VecQuant3MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, vec_height, height, width, zero_width, groupsize
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant3MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  int b = blockIdx.z;
  int h = BLOCKHEIGHT3 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
  __syncthreads();

  scalar_t res = 0;
  int i = width * h + w;
  int g_h = (h / 3) * 32;
  int k = 0;
  
  int z_w = (w / 32) * 3; // ((w / 256) * 24) / 3 
  int z_mod = w % 32;
  int z_bit;
  
  if (z_mod != 10){
    if (z_mod != 21){
      z_bit = z_mod;
      if (z_bit > 21){
        z_bit -= 22;
        z_bit *= 3;
        z_bit += 2;
        z_w += 2;
      } else if (z_bit > 10){
        z_bit -= 11;
        z_bit *= 3;
        z_bit += 1;
        z_w += 1;
      } else {
        z_bit *= 3;
      }
    } else {
      z_w += 1;
    }
  }
 
  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;
  unsigned int z_tmp;

  while (k < BLOCKWIDTH) {
    tmp1 = as_unsigned(mat[i]);
	
    int g = (g_h + k) / groupsize;
    scalar_t scale = scales[g * width + w];
    scalar_t zero;
    if (z_mod == 10) {
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 30) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 2) & 0x4);
      zero = scale * scalar_t((z_tmp) + 1);
    } else if (z_mod == 21){
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 31) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 1) & 0x6);
      zero = scale * scalar_t((z_tmp) + 1);
    } else {
      zero = scale * scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_bit) & 0x7) + 1);
    }
	
    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];
	
    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
    tmp2 >>= 1;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;
	
    res += (scale * scalar_t((tmp2 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp2 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp2 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp2 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp2 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp2 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp2 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp2 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp2 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp2 >> 27) & 0x7) - zero) * blockvec[k + 9];
	
    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
    tmp1 >>= 2;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;
	
    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];
	
    i += width;
    k += 10;
  }

  atomicAdd(&mul[b * width + w], res);
}


void vecquant3decode_cuda(
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize
) {
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    mul.type(), "vecquant3decode_cuda", ([&] {
      VecQuant3DecodeKernel<<<blocks, threads>>>(
        mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        height, width, zero_width, groupsize
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant3DecodeKernel(
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  int h = BLOCKHEIGHT3 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  int i = width * h + w;
  int g_h = (h / 3) * 32;
  int k = (height / 3) * 32 * w + blockIdx.x * BLOCKWIDTH;
  int k_end = (height / 3) * 32 * w + blockIdx.x * BLOCKWIDTH + BLOCKWIDTH;
  
  int z_w = (w / 32) * 3; // ((w / 256) * 24) / 3 
  int z_mod = w % 32;
  int z_bit;
  
  if (z_mod != 10){
    if (z_mod != 21){
      z_bit = z_mod;
      if (z_bit > 21){
        z_bit -= 22;
        z_bit *= 3;
        z_bit += 2;
        z_w += 2;
      } else if (z_bit > 10){
        z_bit -= 11;
        z_bit *= 3;
        z_bit += 1;
        z_w += 1;
      } else {
        z_bit *= 3;
      }
    } else {
      z_w += 1;
    }
  }
 
  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;
  unsigned int z_tmp;

  while (k < k_end) {
    tmp1 = as_unsigned(mat[i]);
	
    int g = (g_h + k + BLOCKWIDTH - k_end) / groupsize;
    scalar_t scale = scales[g * width + w];
    scalar_t zero;
    if (z_mod == 10) {
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 30) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 2) & 0x4);
      zero = scale * scalar_t((z_tmp) + 1);
    } else if (z_mod == 21){
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 31) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 1) & 0x6);
      zero = scale * scalar_t((z_tmp) + 1);
    } else {
      zero = scale * scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_bit) & 0x7) + 1);
    }
	
    mul[k + 0] = (scale * scalar_t((tmp1 >>  0) & 0x7) - zero);
    mul[k + 1] = (scale * scalar_t((tmp1 >>  3) & 0x7) - zero);
    mul[k + 2] = (scale * scalar_t((tmp1 >>  6) & 0x7) - zero);
    mul[k + 3] = (scale * scalar_t((tmp1 >>  9) & 0x7) - zero);
    mul[k + 4] = (scale * scalar_t((tmp1 >> 12) & 0x7) - zero);
    mul[k + 5] = (scale * scalar_t((tmp1 >> 15) & 0x7) - zero);
    mul[k + 6] = (scale * scalar_t((tmp1 >> 18) & 0x7) - zero);
    mul[k + 7] = (scale * scalar_t((tmp1 >> 21) & 0x7) - zero);
    mul[k + 8] = (scale * scalar_t((tmp1 >> 24) & 0x7) - zero);
    mul[k + 9] = (scale * scalar_t((tmp1 >> 27) & 0x7) - zero);

    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
    tmp2 >>= 1;
    mul[k + 10] = (scale * scalar_t(tmp) - zero);
    k += 11;
	
    mul[k + 0] = (scale * scalar_t((tmp2 >>  0) & 0x7) - zero);
    mul[k + 1] = (scale * scalar_t((tmp2 >>  3) & 0x7) - zero);
    mul[k + 2] = (scale * scalar_t((tmp2 >>  6) & 0x7) - zero);
    mul[k + 3] = (scale * scalar_t((tmp2 >>  9) & 0x7) - zero);
    mul[k + 4] = (scale * scalar_t((tmp2 >> 12) & 0x7) - zero);
    mul[k + 5] = (scale * scalar_t((tmp2 >> 15) & 0x7) - zero);
    mul[k + 6] = (scale * scalar_t((tmp2 >> 18) & 0x7) - zero);
    mul[k + 7] = (scale * scalar_t((tmp2 >> 21) & 0x7) - zero);
    mul[k + 8] = (scale * scalar_t((tmp2 >> 24) & 0x7) - zero);
    mul[k + 9] = (scale * scalar_t((tmp2 >> 27) & 0x7) - zero);
	
    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
    tmp1 >>= 2;
    mul[k + 10] = (scale * scalar_t(tmp) - zero);
    k += 11;
	
    mul[k + 0] = (scale * scalar_t((tmp1 >>  0) & 0x7) - zero);
    mul[k + 1] = (scale * scalar_t((tmp1 >>  3) & 0x7) - zero);
    mul[k + 2] = (scale * scalar_t((tmp1 >>  6) & 0x7) - zero);
    mul[k + 3] = (scale * scalar_t((tmp1 >>  9) & 0x7) - zero);
    mul[k + 4] = (scale * scalar_t((tmp1 >> 12) & 0x7) - zero);
    mul[k + 5] = (scale * scalar_t((tmp1 >> 15) & 0x7) - zero);
    mul[k + 6] = (scale * scalar_t((tmp1 >> 18) & 0x7) - zero);
    mul[k + 7] = (scale * scalar_t((tmp1 >> 21) & 0x7) - zero);
    mul[k + 8] = (scale * scalar_t((tmp1 >> 24) & 0x7) - zero);
    mul[k + 9] = (scale * scalar_t((tmp1 >> 27) & 0x7) - zero);
	
    i += width;
    k += 10;
  }

}


void vecquant3decode_half_cuda(
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize
) {
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3DecodeHalfKernel<<<blocks, threads>>>(
    mat.data<int>(), mul.data<float>(),
    (half*) scales.data_ptr(), zeros.data<int>(),
    height, width, zero_width, groupsize
  );

}

__global__ void VecQuant3DecodeHalfKernel(
    const       int* __restrict__ mat,
           float* __restrict__ mul,
    const  half* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  int h = BLOCKHEIGHT3 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  int i = width * h + w;
  int g_h = (h / 3) * 32;
  int k = (height / 3) * 32 * w + blockIdx.x * BLOCKWIDTH;
  int k_end = (height / 3) * 32 * w + blockIdx.x * BLOCKWIDTH + BLOCKWIDTH;
  
  int z_w = (w / 32) * 3; // ((w / 256) * 24) / 3 
  int z_mod = w % 32;
  int z_bit;
  
  if (z_mod != 10){
    if (z_mod != 21){
      z_bit = z_mod;
      if (z_bit > 21){
        z_bit -= 22;
        z_bit *= 3;
        z_bit += 2;
        z_w += 2;
      } else if (z_bit > 10){
        z_bit -= 11;
        z_bit *= 3;
        z_bit += 1;
        z_w += 1;
      } else {
        z_bit *= 3;
      }
    } else {
      z_w += 1;
    }
  }
 
  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;
  unsigned int z_tmp;

  while (k < k_end) {
    tmp1 = as_unsigned(mat[i]);
	
    int g = (g_h + k + BLOCKWIDTH - k_end) / groupsize;
    //float scale = scales[g * width + w];
    float scale = __half2float(scales[g * width + w]);
    float zero;
    if (z_mod == 10) {
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 30) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 2) & 0x4);
      zero = scale * float((z_tmp) + 1);
    } else if (z_mod == 21){
      z_tmp = (as_unsigned(zeros[g * zero_width + z_w]) >> 31) | ((as_unsigned(zeros[g * zero_width + (z_w + 1)]) << 1) & 0x6);
      zero = scale * float((z_tmp) + 1);
    } else {
      zero = scale * float(((as_unsigned(zeros[g * zero_width + z_w]) >> z_bit) & 0x7) + 1);
    }
	
    mul[k + 0] = (scale * float((tmp1 >>  0) & 0x7) - zero);
    mul[k + 1] = (scale * float((tmp1 >>  3) & 0x7) - zero);
    mul[k + 2] = (scale * float((tmp1 >>  6) & 0x7) - zero);
    mul[k + 3] = (scale * float((tmp1 >>  9) & 0x7) - zero);
    mul[k + 4] = (scale * float((tmp1 >> 12) & 0x7) - zero);
    mul[k + 5] = (scale * float((tmp1 >> 15) & 0x7) - zero);
    mul[k + 6] = (scale * float((tmp1 >> 18) & 0x7) - zero);
    mul[k + 7] = (scale * float((tmp1 >> 21) & 0x7) - zero);
    mul[k + 8] = (scale * float((tmp1 >> 24) & 0x7) - zero);
    mul[k + 9] = (scale * float((tmp1 >> 27) & 0x7) - zero);

    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
    tmp2 >>= 1;
    mul[k + 10] = (scale * float(tmp) - zero);
    k += 11;
	
    mul[k + 0] = (scale * float((tmp2 >>  0) & 0x7) - zero);
    mul[k + 1] = (scale * float((tmp2 >>  3) & 0x7) - zero);
    mul[k + 2] = (scale * float((tmp2 >>  6) & 0x7) - zero);
    mul[k + 3] = (scale * float((tmp2 >>  9) & 0x7) - zero);
    mul[k + 4] = (scale * float((tmp2 >> 12) & 0x7) - zero);
    mul[k + 5] = (scale * float((tmp2 >> 15) & 0x7) - zero);
    mul[k + 6] = (scale * float((tmp2 >> 18) & 0x7) - zero);
    mul[k + 7] = (scale * float((tmp2 >> 21) & 0x7) - zero);
    mul[k + 8] = (scale * float((tmp2 >> 24) & 0x7) - zero);
    mul[k + 9] = (scale * float((tmp2 >> 27) & 0x7) - zero);
	
    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
    tmp1 >>= 2;
    mul[k + 10] = (scale * float(tmp) - zero);
    k += 11;
	
    mul[k + 0] = (scale * float((tmp1 >>  0) & 0x7) - zero);
    mul[k + 1] = (scale * float((tmp1 >>  3) & 0x7) - zero);
    mul[k + 2] = (scale * float((tmp1 >>  6) & 0x7) - zero);
    mul[k + 3] = (scale * float((tmp1 >>  9) & 0x7) - zero);
    mul[k + 4] = (scale * float((tmp1 >> 12) & 0x7) - zero);
    mul[k + 5] = (scale * float((tmp1 >> 15) & 0x7) - zero);
    mul[k + 6] = (scale * float((tmp1 >> 18) & 0x7) - zero);
    mul[k + 7] = (scale * float((tmp1 >> 21) & 0x7) - zero);
    mul[k + 8] = (scale * float((tmp1 >> 24) & 0x7) - zero);
    mul[k + 9] = (scale * float((tmp1 >> 27) & 0x7) - zero);
	
    i += width;
    k += 10;

    if (k-32 <= 2023474 && 2023474 < k) {
      printf("k: %d, g: %d, h: %d, w: %d, scale: %f, zero: %f, w_out: %f\n", k, g, (g_h + k + BLOCKWIDTH - k_end), w, scale, zero, mul[2023474]);
    }
    if (k-32 <= 3588468 && 3588468 < k) {
      printf("k: %d, g: %d, h: %d, w: %d, scale: %f, zero: %f, w_out: %f\n", k, g, (g_h + k + BLOCKWIDTH - k_end), w, scale, zero, mul[3588468]);
    }
  }

}


void vecquant4matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    4
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant4matmul_cuda", ([&] {
      VecQuant4MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, vec_height, height, width, zero_width, groupsize
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  int b = blockIdx.z;
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
  __syncthreads();

  scalar_t res = 0;
  int i = width * h + w;
  int g_h = h * 8;
  int k = 0;

  int z_w = w / 8; 
  int z_mod = (w % 8) * 4;

  unsigned int tmp;

  while (k < BLOCKWIDTH) {
    tmp = as_unsigned(mat[i]);
	
    int g = (g_h + k) / groupsize;
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scale * scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1);
	
    res += (scale * scalar_t((tmp >> 0) & 0xF) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp >> 4) & 0xF) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp >> 8) & 0xF) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp >> 12) & 0xF) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp >> 16) & 0xF) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp >> 20) & 0xF) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp >> 24) & 0xF) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp >> 28) & 0xF) - zero) * blockvec[k + 7];
	
    i += width;
    k += 8;
  }

  atomicAdd(&mul[b * width + w], res);
}

void vecquant8matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT8 - 1) / BLOCKHEIGHT8,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant8matmul_cuda", ([&] {
      VecQuant8MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, vec_height, height, width, zero_width, groupsize
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant8MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
  int b = blockIdx.z;
  int h = BLOCKHEIGHT8 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
  __syncthreads();

  scalar_t res = 0;
  int i = width * h + w;
  int g_h = h * 4;
  int k = 0;
  
  int z_w = w / 4; 
  int z_mod = (w % 4) * 8;

  unsigned int tmp;

  while (k < BLOCKWIDTH) { 
    tmp = as_unsigned(mat[i]);
	
    int g = (g_h + k) / groupsize;
    scalar_t scale = scales[g * width + w];
    scalar_t zero = scale * scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xFF) + 1);
	
    res += (scale * scalar_t((tmp >> 0) & 0xFF) - zero) * blockvec[k + 0];
    res += (scale * scalar_t((tmp >> 8) & 0xFF) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp >> 16) & 0xFF) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp >> 24) & 0xFF) - zero) * blockvec[k + 3];
	
    i += width;
    k += 4;
  }

  atomicAdd(&mul[b * width + w], res);
}

//const int TILE_DIM = 32;
//const int BLOCK_ROWS = 4;
//const int PACK_COUNT = 8;
//const int BITS = 4;
//const int BITS_MASK = 0xF;

const int TILE_DIM = 16;
const int BLOCK_ROWS = 4;
const int BITS = 4;
const int PACK_COUNT = 32 / BITS;
const int BITS_MASK = 0xF;

void transpose_cuda(
  torch::Tensor odata,
  torch::Tensor idata
) {
  int iheight = idata.size(0);
  int iwidth = idata.size(1);

  dim3 blocks(
    (iwidth + TILE_DIM - 1) / TILE_DIM,
    (iheight + TILE_DIM - 1) / TILE_DIM
  );
  dim3 threads(TILE_DIM, TILE_DIM);

  transposeCoalesced<<<blocks, threads>>>(
    odata.data<int>(), idata.data<int>(), iwidth, iheight
  );
}

__global__ void transposeCoalesced(int *odata, const int *idata, const int iwidth, const int iheight)
{
  __shared__ int tile[TILE_DIM][TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int m, n;

  if (x < iwidth && y < iheight) {
    tile[threadIdx.y][threadIdx.x] = idata[y * iwidth + x];
  }

  __syncthreads();

  const int pack_x = threadIdx.y % PACK_COUNT;
  const int pack_y = threadIdx.y / PACK_COUNT;
  const int owidth = iheight * PACK_COUNT;
  const int oheight = iwidth / PACK_COUNT;
  const unsigned int thread_bit_mask = BITS_MASK << pack_x*BITS;

  x = (blockIdx.y * TILE_DIM + threadIdx.x) * PACK_COUNT + pack_x; // transpose block offset
  y = (blockIdx.x * TILE_DIM) / PACK_COUNT + pack_y;

  int output = 0;

  if (x < owidth && y < oheight) {
    //if (threadIdx.x == 2, threadIdx.y == 2){
    //  printf("blockx: %d, blocky: %d, threadx: %d, thready: %d, x: %d, y: %d, packx: %d, packy: %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, pack_x, pack_y);
    //}
    
    // Pass all the positions after this block
    for (m = 0; m < PACK_COUNT-pack_x; m++) {
      output = output | ((tile[threadIdx.x][threadIdx.y + m] & thread_bit_mask) << m*BITS);
    }
    // Pass all the positions before this block
    for (n = 1; n <= pack_x; n++) {
      output = output | ((tile[threadIdx.x][threadIdx.y - n] & thread_bit_mask) >> n*BITS);
    }

    odata[y * owidth + x] = output;
  }
}

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel_Back(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  	int* __restrict__ zeros,
    int batch,
    int vec_height, 	
    int height,
    int width,
    int zero_width,
    int groupsize
);

void vecquant4matmul_back_cuda(
    torch::Tensor vec,
    torch::Tensor mat,
    torch::Tensor mul,
    torch::Tensor scales,
    torch::Tensor zeros,
    int groupsize
) {
    int batch = vec.size(0);
    int vec_height = vec.size(1);
    int height = mat.size(0);
    int width = mat.size(1);
    int zero_width = zeros.size(1);

    dim3 blocks(
        (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
        (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
        batch
    );
    dim3 threads(BLOCKWIDTH);

    AT_DISPATCH_FLOATING_TYPES(
        vec.type(), "vecquant4matmul_back_cuda", ([&] {
        VecQuant4MatMulKernel_Back<<<blocks, threads>>>(
            vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
            scales.data<scalar_t>(), zeros.data<int>(),
            batch, vec_height, height, width, zero_width, groupsize
        );
        })
    );
}

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel_Back(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width,
    int groupsize
) {
    int b = blockIdx.z;
    int h = BLOCKHEIGHT4 * blockIdx.x;
    int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

    __shared__ scalar_t blockvec[BLOCKWIDTH];
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
    __syncthreads();

    scalar_t res = 0;
    int i = width * h + w;
    //int g_h = h * 8;
    int k = 0;

    //int z_w = w / 8; 
    //int z_mod = (w % 8) * 4;

    unsigned int tmp;

    while (k < BLOCKWIDTH) {
        tmp = as_unsigned(mat[i]);
        
        //int g = (g_h + k) / groupsize;
        unsigned int zero_pack = as_unsigned(zeros[h]);

        scalar_t scale = scales[h * 8 + 0];
        scalar_t zero = scale * scalar_t(((zero_pack >> 0) & 0xF) + 1);
        scalar_t decompressed_x = (scale * scalar_t((tmp >> 0) & 0xF) - zero);
        // if (threadIdx.x == 1) {
        //   printf("tmp: %u, i: %d, blockidx: %d, blockidy: %d, threadx: %d, w: %f, dim0: %d, dim1: %d, zero: %f, zero_pack: %d\n", tmp, i, blockIdx.x, blockIdx.y, threadIdx.x, decompressed_x, h * 8 + 0, w, zero, zero_pack);
        // }
        // res += (scale * scalar_t((tmp >> 0) & 0xF) - zero) * blockvec[k + 0];
        res += decompressed_x * blockvec[k + 0];

        scale = scales[h * 8 + 1];
        zero = scale * scalar_t(((zero_pack >> 4) & 0xF) + 1);
        res += (scale * scalar_t((tmp >> 4) & 0xF) - zero) * blockvec[k + 1];

        scale = scales[h * 8 + 2];
        zero = scale * scalar_t(((zero_pack >> 8) & 0xF) + 1);
        res += (scale * scalar_t((tmp >> 8) & 0xF) - zero) * blockvec[k + 2];

        scale = scales[h * 8 + 3];
        zero = scale * scalar_t(((zero_pack >> 12) & 0xF) + 1);
        res += (scale * scalar_t((tmp >> 12) & 0xF) - zero) * blockvec[k + 3];

        scale = scales[h * 8 + 4];
        zero = scale * scalar_t(((zero_pack >> 16) & 0xF) + 1);
        res += (scale * scalar_t((tmp >> 16) & 0xF) - zero) * blockvec[k + 4];

        scale = scales[h * 8 + 5];
        zero = scale * scalar_t(((zero_pack >> 20) & 0xF) + 1);
        res += (scale * scalar_t((tmp >> 20) & 0xF) - zero) * blockvec[k + 5];

        scale = scales[h * 8 + 6];
        zero = scale * scalar_t(((zero_pack >> 24) & 0xF) + 1);
        res += (scale * scalar_t((tmp >> 24) & 0xF) - zero) * blockvec[k + 6];

        scale = scales[h * 8 + 7];
        zero = scale * scalar_t(((zero_pack >> 28) & 0xF) + 1);
        res += (scale * scalar_t((tmp >> 28) & 0xF) - zero) * blockvec[k + 7];
        
        i += width;
        k += 8;
        h += 1;
    }

    atomicAdd(&mul[b * width + w], res);
}