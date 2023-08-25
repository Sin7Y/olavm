//Edit by Malone and Longson
//creat data:2023.6.10
#include <cstdint> 

 extern "C" {
 void evaluate_poly(uint64_t* vec, uint64_t n);
 void evaluate_poly_with_offset(uint64_t* vec, uint64_t n, uint64_t domain_offset, uint64_t blowup_factor, uint64_t* result, uint64_t result_len);
 void interpolate_poly(uint64_t* vec, uint64_t n);
 void interpolate_poly_with_offset(uint64_t* vec, uint64_t n, uint64_t domain_offset);
 }