#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

void vecquant2matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
); 

void vecquant2matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant2matmul_cuda(vec, mat, mul, scales, zeros,groupsize);
}

void vecquant3matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
); 

void vecquant3matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_cuda(vec, mat, mul, scales, zeros, groupsize);
}

void vecquant3decode_cuda(
  torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
); 

void vecquant3decode(
  torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant3decode_cuda(mat, mul, scales, zeros, groupsize);
}

void vecquant3decode_half_cuda(
  torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
); 

void vecquant3decode_half(
  torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant3decode_half_cuda(mat, mul, scales, zeros, groupsize);
}

void vecquant4matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
); 

void vecquant4matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_cuda(vec, mat, mul, scales, zeros, groupsize);
}

void vecquant8matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
); 

void vecquant8matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  int groupsize
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant8matmul_cuda(vec, mat, mul, scales, zeros, groupsize);
}

void transpose_cuda(
    torch::Tensor odata,
    torch::Tensor idata
);

void transpose_packedint(
  torch::Tensor odata,
  torch::Tensor idata
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(idata));
  transpose_cuda(odata, idata);
}

void vecquant4matmul_back_cuda(
    torch::Tensor vec,
    torch::Tensor mat,
    torch::Tensor mul,
    torch::Tensor scales,
    torch::Tensor zeros,
    int groupsize
);

void vecquant4matmul_back(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  int groupsize
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_back_cuda(vec, mat, mul, scales, zeros, groupsize);
}

/*
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant3decode", &vecquant3decode, "Vector 3-bit Quantized Decode and Transpose (CUDA)");
  m.def("vecquant3decode_half", &vecquant3decode_half, "Vector 3-bit FP16 Quantized Decode and Transpose (CUDA)");
  m.def("vecquant4matmul_back", &vecquant4matmul_back, "Vector 4-bit Quantized.T Matrix Multiplication Backward (CUDA)");
  m.def("transpose_packedint", &transpose_packedint, "Transpose Packed Integer Matrix (CUDA)");
  m.def("vecquant2matmul", &vecquant2matmul, "Vector 2-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant3matmul", &vecquant3matmul, "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant4matmul", &vecquant4matmul, "Vector 4-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant8matmul", &vecquant8matmul, "Vector 8-bit Quantized Matrix Multiplication (CUDA)");
}
*/

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant3decode_half", &vecquant3decode_half, "Vector 3-bit FP16 Quantized Decode and Transpose (CUDA)");
}

