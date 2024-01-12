#ifndef SRC_LAYER_GPU_FORWARD_H
#define SRC_LAYER_GPU_FORWARD_H

class GPUInterface
{
    public:
    void get_device_properties();
    void conv_forward_gpu_prolog(
        const float *host_y,
        const float *host_x,
        const float *host_k,
        float **device_y_ptr,
        float **device_x_ptr,
        float **device_k_ptr,
        const int batch_size,
        const int outputFeatureMapsQuantity,
        const int inputFeatureMapsQuantity,
        const int dimensionHeight,
        const int dimensionWeight,
        const int kernelSize
    );
    void conv_forward_gpu(
        float *device_y,
        const float *device_x,
        const float *device_k,
        const int batch_size,
        const int outputFeatureMapsQuantity,
        const int inputFeatureMapsQuantity,
        const int dimensionHeight,
        const int dimensionWeight,
        const int kernelSize
    );
    void conv_forward_gpu_epilog(
        float *host_y,
        float *device_y,
        float *device_x,
        float *device_k,
        const int batch_size,
        const int outputFeatureMapsQuantity,
        const int inputFeatureMapsQuantity,
        const int dimensionHeight,
        const int dimensionWeight,
        const int kernelSize
    );
};

#endif