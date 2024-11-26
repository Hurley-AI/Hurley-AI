/**
 * File   : thrust_bask_lib.hpp
 * License: GNU v3.0
 * Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
 * Date   : 26.03.2024
 */


#ifndef BUC_HPP
#define BUC_HPP


#include <stdint.h>


// Declare C++ functions
template<typename T>
void buc_sort(T* device_vector, int length);

template<typename I, typename T>
void buc_upper_bound(I* d_out, T* d_vector, int vector_length, T* d_elements, int elements_length);


// Declare exposed C functions with no name mangling
#define BUC_EXPORT extern "C"

// Sorting functions
BUC_EXPORT void buc_sort_int16(int16_t* device_vector, int length);
BUC_EXPORT void buc_sort_int32(int32_t* device_vector, int length);
BUC_EXPORT void buc_sort_int64(int64_t* device_vector, int length);

BUC_EXPORT void buc_sort_uint16(uint16_t* device_vector, int length);
BUC_EXPORT void buc_sort_uint32(uint32_t* device_vector, int length);
BUC_EXPORT void buc_sort_uint64(uint64_t* device_vector, int length);

BUC_EXPORT void buc_sort_float32(float* device_vector, int length);
BUC_EXPORT void buc_sort_float64(double* device_vector, int length);

// Upper bound functions
BUC_EXPORT void buc_upper_bound_int16(int64_t* d_out, int16_t* d_vector, int vector_len,
                                      int16_t* d_elements, int elements_len);

BUC_EXPORT void buc_upper_bound_int32(int64_t* d_out, int32_t* d_vector, int vector_len,
                                      int32_t* d_elements, int elements_len);

BUC_EXPORT void buc_upper_bound_int64(int64_t* d_out, int64_t* d_vector, int vector_len,
                                      int64_t* d_elements, int elements_len);

BUC_EXPORT void buc_upper_bound_uint16(int64_t* d_out, uint16_t* d_vector, int vector_len,
                                       uint16_t* d_elements, int elements_len);

BUC_EXPORT void buc_upper_bound_uint32(int64_t* d_out, uint32_t* d_vector, int vector_len,
                                       uint32_t* d_elements, int elements_len);

BUC_EXPORT void buc_upper_bound_uint64(int64_t* d_out, uint64_t* d_vector, int vector_len,
                                       uint64_t* d_elements, int elements_len);

BUC_EXPORT void buc_upper_bound_float32(int64_t* d_out, float* d_vector, int vector_len,
                                        float* d_elements, int elements_len);

BUC_EXPORT void buc_upper_bound_float64(int64_t* d_out, double* d_vector, int vector_len,
                                        double* d_elements, int elements_len);



#endif // BUC_HPP

