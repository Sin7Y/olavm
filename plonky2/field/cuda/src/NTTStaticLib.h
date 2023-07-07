#include <cstdint> 	/* uint64_t */

 extern "C" {
    void evaluate_poly(uint64_t* vec, uint64_t N);
    void evaluate_poly_with_offset(uint64_t* vec, uint64_t N, uint64_t domain_offset, uint64_t blowup_factor);
    void interpolate_poly(uint64_t* vec, uint64_t N);
    void interpolate_poly_with_offset(uint64_t* vec, uint64_t N, uint64_t domain_offset);
 }