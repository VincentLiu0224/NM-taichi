import numpy as np
import taichi as ti
import time

# Initialize Taichi and run it on CPU (default)
# - `arch=ti.gpu`: Run Taichi on GPU and has Taichi automatically detect the suitable backend
# - `arch=ti.cuda`: For the NVIDIA CUDA backend
# - `arch=ti.metal`: [macOS] For the Apple Metal backend
# - `arch=ti.opengl`: For the OpenGL backend
# - `arch=ti.vulkan`: For the Vulkan backend
# - `arch=ti.dx11`: For the DX11 backend
ti.init(arch=ti.gpu)
#ti.init(arch=ti.cuda, device_memory_fraction=0.3)

ti.init(default_fp=ti.f32)

@ti.kernel
def test(x: ti.f32) -> ti.f32: # The return value is type hinted
    return 1.0

x = test(5)
print(x)

N = 6
matN = ti.types.matrix(N, N, ti.i32)

a= ti.types.matrix(n=3, m=3, dtype=ti.f32)
v = a([10., 200., 1.], [10., 5., 1.], [50., 4., 5.])
print(v)
v[0, 0] = 2.
print(v)

# A region of shape(N, M) with each elements being influenced by a field of vector/tensor, which dimensions
# are defined as n=, m=
i =0.01
s = 1-i
r = 0.
u0 = ti.Vector.field(n=3, dtype=ti.f32, shape=(1, 100))
print(u0[1:-1])
