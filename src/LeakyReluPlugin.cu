#include "LeakyReluPlugin.h"

//cuda
__global__ void _leakyReluKer(float const *in, float *out, int size, float negative_slope) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= size)
        return ;

    if (in[index] < 0)
        out[index] = in[index] * negative_slope;
    else
        out[index] = in[index];
}

// currently it is not possible for a plugin to execute "in place". Therefore we memcpy the data from the input to the output buffer
int LeakyReluPlugin::enqueue(int batchSize, const void*const *inputs, void** outputs, void*, cudaStream_t stream) {
	int block_size = 256;
	int grid_size = (m_param.size + block_size - 1) / block_size;
	_leakyReluKer<<<grid_size, block_size>>>(
            reinterpret_cast<float const*>(inputs[0]),
            reinterpret_cast<float*>(outputs[0]), m_param.size, m_param.negative_slope);

	return 0;
}

