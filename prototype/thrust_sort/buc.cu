/**
 * File   : thrust_bask_lib.cpp
 * License: GNU v3.0
 * Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
 * Date   : 26.03.2024
 */


#include <cstdint>

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>

#include <buc.hpp>


// Templated C++ functions
template<typename T>
void buc_sort(T* device_vector, int length) {
    thrust::device_ptr<T> device_vector_wrap(device_vector);
    thrust::sort(device_vector_wrap, device_vector_wrap + length);
}

template<typename I, typename T>
void buc_upper_bound(I* d_out, T* d_vector, int vector_len, T* d_elements, int elements_len) {
    thrust::device_ptr<I> d_out_wrap(d_out);
    thrust::device_ptr<T> d_vector_wrap(d_vector);
    thrust::device_ptr<T> d_elements_wrap(d_elements);

    thrust::upper_bound(
        d_vector_wrap, d_vector_wrap + vector_len,
        d_elements_wrap, d_elements_wrap + elements_len,
        d_out_wrap);
}


// Exported C functions
// Sorting functions
BUC_EXPORT void buc_sort_int16(int16_t* device_vector, int length) {
    buc_sort<int16_t>(device_vector, length);
}

BUC_EXPORT void buc_sort_int32(int32_t* device_vector, int length) {
    buc_sort<int32_t>(device_vector, length);
}

BUC_EXPORT void buc_sort_int64(int64_t* device_vector, int length) {
    buc_sort<int64_t>(device_vector, length);
}

BUC_EXPORT void buc_sort_uint16(uint16_t* device_vector, int length) {
    buc_sort<uint16_t>(device_vector, length);
}

BUC_EXPORT void buc_sort_uint32(uint32_t* device_vector, int length) {
    buc_sort<uint32_t>(device_vector, length);
}

BUC_EXPORT void buc_sort_uint64(uint64_t* device_vector, int length) {
    buc_sort<uint64_t>(device_vector, length);
}

BUC_EXPORT void buc_sort_float32(float* device_vector, int length) {
    buc_sort<float>(device_vector, length);
}

BUC_EXPORT void buc_sort_float64(double* device_vector, int length) {
    buc_sort<double>(device_vector, length);
}

// Upper bound functions
BUC_EXPORT void buc_upper_bound_int16(
    int64_t* d_out,
    int16_t* d_vector,
    int vector_len,
    int16_t* d_elements,
    int elements_len
) {
    buc_upper_bound<int64_t, int16_t>(d_out, d_vector, vector_len, d_elements, elements_len);
}

BUC_EXPORT void buc_upper_bound_int32(
    int64_t* d_out,
    int32_t* d_vector,
    int vector_len,
    int32_t* d_elements,
    int elements_len
) {
    buc_upper_bound<int64_t, int32_t>(d_out, d_vector, vector_len, d_elements, elements_len);
}

BUC_EXPORT void buc_upper_bound_int64(
    int64_t* d_out,
    int64_t* d_vector,
    int vector_len,
    int64_t* d_elements,
    int elements_len
) {
    buc_upper_bound<int64_t, int64_t>(d_out, d_vector, vector_len, d_elements, elements_len);
}

BUC_EXPORT void buc_upper_bound_uint16(
    int64_t* d_out,
    uint16_t* d_vector,
    int vector_len,
    uint16_t* d_elements,
    int elements_len
) {
    buc_upper_bound<int64_t, uint16_t>(d_out, d_vector, vector_len, d_elements, elements_len);
}

BUC_EXPORT void buc_upper_bound_uint32(
    int64_t* d_out,
    uint32_t* d_vector,
    int vector_len,
    uint32_t* d_elements,
    int elements_len
) {
    buc_upper_bound<int64_t, uint32_t>(d_out, d_vector, vector_len, d_elements, elements_len);
}

BUC_EXPORT void buc_upper_bound_uint64(
    int64_t* d_out,
    uint64_t* d_vector,
    int vector_len,
    uint64_t* d_elements,
    int elements_len
) {
    buc_upper_bound<int64_t, uint64_t>(d_out, d_vector, vector_len, d_elements, elements_len);
}

BUC_EXPORT void buc_upper_bound_float32(
    int64_t* d_out,
    float* d_vector,
    int vector_len,
    float* d_elements,
    int elements_len
) {
    buc_upper_bound<int64_t, float>(d_out, d_vector, vector_len, d_elements, elements_len);
}

BUC_EXPORT void buc_upper_bound_float64(
    int64_t* d_out,
    double* d_vector,
    int vector_len,
    double* d_elements,
    int elements_len
) {
    buc_upper_bound<int64_t, double>(d_out, d_vector, vector_len, d_elements, elements_len);
}

