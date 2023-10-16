//Edit by Piaobo
//data:2023.2.10

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cinttypes>
#include <string.h>
#include <math.h>
#include <crt/device_functions.h>
//#include "ff_p.h"

namespace internal {

	typedef std::uint32_t u32;
	typedef std::uint64_t u64;

	__device__ __forceinline__
		void
		addc(u32& s, u32 a, u32 b) {
		asm("addc.u32 %0, %1, %2;"
			: "=r"(s)
			: "r"(a), "r" (b));
	}

	__device__ __forceinline__
		void
		add_cc(u32& s, u32 a, u32 b) {
		asm("add.cc.u32 %0, %1, %2;"
			: "=r"(s)
			: "r"(a), "r" (b));
	}

	__device__ __forceinline__
		void
		addc_cc(u32& s, u32 a, u32 b) {
		asm("addc.cc.u32 %0, %1, %2;"
			: "=r"(s)
			: "r"(a), "r" (b));
	}

	__device__ __forceinline__
		void
		addc(u64& s, u64 a, u64 b) {
		asm("addc.u64 %0, %1, %2;"
			: "=l"(s)
			: "l"(a), "l" (b));
	}

	__device__ __forceinline__
		void
		add_cc(u64& s, u64 a, u64 b) {
		asm("add.cc.u64 %0, %1, %2;"
			: "=l"(s)
			: "l"(a), "l" (b));
	}

	__device__ __forceinline__
		void
		addc_cc(u64& s, u64 a, u64 b) {
		asm("addc.cc.u64 %0, %1, %2;"
			: "=l"(s)
			: "l"(a), "l" (b));
	}

	/*
	 * hi * 2^n + lo = a * b
	 */
	__device__ __forceinline__
		void
		mul_hi(u32& hi, u32 a, u32 b) {
		asm("mul.hi.u32 %0, %1, %2;"
			: "=r"(hi)
			: "r"(a), "r"(b));
	}

	__device__ __forceinline__
		void
		mul_hi(u64& hi, u64 a, u64 b) {
		asm("mul.hi.u64 %0, %1, %2;"
			: "=l"(hi)
			: "l"(a), "l"(b));
	}


	/*
	 * hi * 2^n + lo = a * b
	 */
	__device__ __forceinline__
		void
		mul_wide(u32& hi, u32& lo, u32 a, u32 b) {
		// TODO: Measure performance difference between this and the
		// equivalent:
		//   mul.hi.u32 %0, %2, %3
		//   mul.lo.u32 %1, %2, %3
		asm("{\n\t"
			" .reg .u64 tmp;\n\t"
			" mul.wide.u32 tmp, %2, %3;\n\t"
			" mov.b64 { %1, %0 }, tmp;\n\t"
			"}"
			: "=r"(hi), "=r"(lo)
			: "r"(a), "r"(b));
	}

	__device__ __forceinline__
		void
		mul_wide(u64& hi, u64& lo, u64 a, u64 b) {
		asm("mul.hi.u64 %0, %2, %3;\n\t"
			"mul.lo.u64 %1, %2, %3;\n\t"
			: "=l"(hi), "=l"(lo)
			: "l"(a), "l"(b));
	}

	/*
	 * (hi, lo) = a * b + c
	 */
	__device__ __forceinline__
		void
		mad_wide(u32& hi, u32& lo, u32 a, u32 b, u32 c) {
		asm("{\n\t"
			" .reg .u64 tmp;\n\t"
			" mad.wide.u32 tmp, %2, %3, %4;\n\t"
			" mov.b64 { %1, %0 }, tmp;\n\t"
			"}"
			: "=r"(hi), "=r"(lo)
			: "r"(a), "r"(b), "r"(c));
	}

	__device__ __forceinline__
		void
		mad_wide(u64& hi, u64& lo, u64 a, u64 b, u64 c) {
		asm("mad.lo.cc.u64 %1, %2, %3, %4;\n\t"
			"madc.hi.u64 %0, %2, %3, 0;"
			: "=l"(hi), "=l"(lo)
			: "l"(a), "l" (b), "l"(c));
	}

	// lo = a * b + c (mod 2^n)
	__device__ __forceinline__
		void
		mad_lo(u32& lo, u32 a, u32 b, u32 c) {
		asm("mad.lo.u32 %0, %1, %2, %3;"
			: "=r"(lo)
			: "r"(a), "r" (b), "r"(c));
	}

	__device__ __forceinline__
		void
		mad_lo(u64& lo, u64 a, u64 b, u64 c) {
		asm("mad.lo.u64 %0, %1, %2, %3;"
			: "=l"(lo)
			: "l"(a), "l" (b), "l"(c));
	}


	// as above but with carry in cy
	__device__ __forceinline__
		void
		mad_lo_cc(u32& lo, u32 a, u32 b, u32 c) {
		asm("mad.lo.cc.u32 %0, %1, %2, %3;"
			: "=r"(lo)
			: "r"(a), "r" (b), "r"(c));
	}

	__device__ __forceinline__
		void
		mad_lo_cc(u64& lo, u64 a, u64 b, u64 c) {
		asm("mad.lo.cc.u64 %0, %1, %2, %3;"
			: "=l"(lo)
			: "l"(a), "l" (b), "l"(c));
	}

	__device__ __forceinline__
		void
		madc_lo_cc(u32& lo, u32 a, u32 b, u32 c) {
		asm("madc.lo.cc.u32 %0, %1, %2, %3;"
			: "=r"(lo)
			: "r"(a), "r" (b), "r"(c));
	}

	__device__ __forceinline__
		void
		madc_lo_cc(u64& lo, u64 a, u64 b, u64 c) {
		asm("madc.lo.cc.u64 %0, %1, %2, %3;"
			: "=l"(lo)
			: "l"(a), "l" (b), "l"(c));
	}

	__device__ __forceinline__
		void
		mad_hi(u32& hi, u32 a, u32 b, u32 c) {
		asm("mad.hi.u32 %0, %1, %2, %3;"
			: "=r"(hi)
			: "r"(a), "r" (b), "r"(c));
	}

	__device__ __forceinline__
		void
		mad_hi(u64& hi, u64 a, u64 b, u64 c) {
		asm("mad.hi.u64 %0, %1, %2, %3;"
			: "=l"(hi)
			: "l"(a), "l" (b), "l"(c));
	}

	__device__ __forceinline__
		void
		mad_hi_cc(u32& hi, u32 a, u32 b, u32 c) {
		asm("mad.hi.cc.u32 %0, %1, %2, %3;"
			: "=r"(hi)
			: "r"(a), "r" (b), "r"(c));
	}

	__device__ __forceinline__
		void
		mad_hi_cc(u64& hi, u64 a, u64 b, u64 c) {
		asm("mad.hi.cc.u64 %0, %1, %2, %3;"
			: "=l"(hi)
			: "l"(a), "l" (b), "l"(c));
	}

	__device__ __forceinline__
		void
		madc_hi_cc(u32& hi, u32 a, u32 b, u32 c) {
		asm("madc.hi.cc.u32 %0, %1, %2, %3;"
			: "=r"(hi)
			: "r"(a), "r" (b), "r"(c));
	}

	__device__ __forceinline__
		void
		madc_hi_cc(u64& hi, u64 a, u64 b, u64 c) {
		asm("madc.hi.cc.u64 %0, %1, %2, %3;\n\t"
			: "=l"(hi)
			: "l"(a), "l" (b), "l"(c));
	}

}

using namespace internal;

__host__ __device__ __forceinline__ void _uint96_modP(const unsigned long long& a, const unsigned int& b, const unsigned long long& p, unsigned long long& c);

class uint128_t
{
public:

	unsigned long long low;
	unsigned long long high;

	__host__ __device__ __forceinline__ uint128_t()
	{
		low = 0;
		high = 0;
	}

	__host__ __device__ __forceinline__ uint128_t(const uint64_t& x)
	{
		low = x;
		high = 0;
	}

	__host__ __device__ __forceinline__ void operator=(const uint128_t& r)
	{
		low = r.low;
		high = r.high;
	}

	__host__ __device__ __forceinline__ void operator=(const uint64_t& r)
	{
		low = r;
		high = 0;
	}

	__host__ __device__ __forceinline__ uint128_t operator<<(const unsigned& shift)
	{
		uint128_t z;
		if (shift != 0)
		{
			if (shift != 64)
			{
				z.high = high << shift;
				z.high = (low >> (64 - shift)) | z.high;
				z.low = low << shift;
			}
			else
			{
				z.high = low; z.low = 0;
			}
		}
		else
		{
			z.high = high; z.low = low;
		}
		return z;
	}

	__host__ __device__ __forceinline__ uint128_t operator>>(const unsigned& shift)
	{
		uint128_t z;
		if (shift != 0)
		{
			if (shift != 64)
			{
				z.low = low >> shift;
				z.low = (high << (64 - shift)) | z.low;
				z.high = high >> shift;
			}
			else
			{
				z.high = 0; z.low = high;
			}

		}
		else
		{
			z.high = high; z.low = low;
		}
		return z;
	}

	__host__ __device__ __forceinline__ static void shiftr(uint128_t& x, const unsigned& shift)
	{
		x.low = x.low >> shift;
		x.low = (x.high << (64 - shift)) | x.low;
		x.high = x.high >> shift;

	}

	__host__ static uint128_t exp2(const int& e)
	{
		uint128_t z;

		if (e < 64)
			z.low = 1ull << e;
		else
			z.high = 1ull << (e - 64);

		return z;
	}

	__host__ static int log_2(const uint128_t& x)
	{
		int z = 0;

		if (x.high != 0)
			z = log2((double)x.high) + 64;
		else
			z = log2((double)x.low);

		return z;
	}

	__host__ __device__ __forceinline__ static int clz(uint128_t x)
	{
		unsigned cnt = 0;

		if (x.high == 0)
		{
			while (x.low != 0)
			{
				cnt++;
				x.low = x.low >> 1;
			}

			return 128 - cnt;
		}
		else
		{
			while (x.high != 0)
			{
				cnt++;
				x.high = x.high >> 1;
			}

			return 64 - cnt;
		}
	}

};

__host__ __device__ __forceinline__ static void operator<<=(uint128_t& x, const unsigned& shift)
{
	x.low = x.low >> shift;
	x.low = (x.high << (64 - shift)) | x.low;
	x.high = x.high >> shift;

}

__host__ __device__ __forceinline__ bool operator==(const uint128_t& l, const uint128_t& r)
{
	if ((l.low == r.low) && (l.high == r.high))
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator<(const uint128_t& l, const uint128_t& r)
{
	if (l.high < r.high)
		return true;
	else if (l.high > r.high)
		return false;
	else if (l.low < r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator<(const uint128_t& l, const uint64_t& r)
{
	if (l.high != 0)
		return false;
	else if (l.low > r)
		return false;
	else
		return true;
}

__host__ __device__ __forceinline__ bool operator>(const uint128_t& l, const uint128_t& r)
{
	if (l.high > r.high)
		return true;
	else if (l.high < r.high)
		return false;
	else if (l.low > r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator<=(const uint128_t& l, const uint128_t& r)
{
	if (l.high < r.high)
		return true;
	else if (l.high > r.high)
		return false;
	else if (l.low <= r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ bool operator>=(const uint128_t& l, const uint128_t& r)
{
	if (l.high > r.high)
		return true;
	else if (l.high < r.high)
		return false;
	else if (l.low >= r.low)
		return true;
	else
		return false;
}

__host__ __device__ __forceinline__ uint128_t operator+(const uint128_t& x, const uint128_t& y)
{
	uint128_t z;

	z.low = x.low + y.low;
	z.high = x.high + y.high + (z.low < x.low);

	return z;
}

__host__ __device__ __forceinline__ uint128_t operator+(const uint128_t& x, const uint64_t& y)
{
	uint128_t z;

	z.low = x.low + y;
	z.high = x.high + (z.low < x.low);

	return z;
}

__host__ __device__ __forceinline__ uint128_t operator-(const uint128_t& x, const uint128_t& y)
{
	uint128_t z;

	z.low = x.low - y.low;
	z.high = x.high - y.high - (x.low < y.low);

	return z;

}

__host__ __device__ __forceinline__ void operator-=(uint128_t& x, const uint128_t& y)
{
	x.high = x.high - y.high - (x.low < y.low);
	x.low = x.low - y.low;
}

__host__ __device__ __forceinline__ uint128_t operator-(const uint128_t& x, const uint64_t& y)
{
	uint128_t z;

	z.low = x.low - y;
	z.high = x.high - (x.low < y);

	return z;

}

__host__ __device__ __forceinline__ uint128_t operator/(uint128_t x, const uint64_t& y)
{
	uint128_t z;
	uint128_t ycomp(y);
	uint128_t d(y);

	unsigned shift = uint128_t::clz(d) - uint128_t::clz(x);

	d = d << shift;

	while (shift != 0)
	{
		shift--;
		z = z << 1;
		if (d <= x)
		{
			x = x - d;
			z = z + 1;
		}
		d = d >> 1;
	}

	z = z << 1;
	if (d <= x)
	{
		x = x - d;
		z = z + 1;
	}
	d = d >> 1;

	return z;
}

__host__ __device__ __forceinline__ uint128_t operator%(uint128_t x, const uint64_t& y)
{
	if (x < y)
		return x;

	uint128_t z;
	uint128_t ycomp(y);
	uint128_t d(y);

	unsigned shift = uint128_t::clz(d) - uint128_t::clz(x);

	d = d << shift;

	while (shift != 0)
	{
		shift--;
		//z = z << 1;
		while (d <= x)
		{
			x = x - d;
			//	z = z + 1;
		}
		d = d >> 1;
	}

	//z = z << 1;
	while (d <= x)
	{
		x = x - d;
		//z = z + 1;
	}
	//d = d >> 1;

	return x;
}

__host__ inline static uint128_t host64x2(const uint64_t& x, const uint64_t& y)
{
	uint128_t z;

	uint128_t ux(x);
	uint128_t uy(y);

	int shift = 0;

	// hello elementary school
	while (uy.low != 0)
	{
		if (uy.low & 1)
		{
			if (shift == 0)
				z = z + ux;
			else
				z = z + (ux << shift);
		}

		shift++;

		uint128_t::shiftr(uy, 1);

	}

	return z;
}

__device__ __forceinline__ void sub128(uint128_t& a, const uint128_t& b)
{
	//asm("{\n\t"
	//	"sub.cc.u64      %1, %3, %5;    \n\t"
	//	"subc.u64        %0, %2, %4;    \n\t"
	//	"}"
	//	: "=l"(a.high), "=l"(a.low)
	//	: "l"(a.high), "l"(a.low), "l"(b.high), "l"(b.low));

	uint128_t z;

	z.low = a.low - b.low;
	z.high = a.high - b.high - (a.low < b.low);
	a = z;
	//return z;


}

__host__ __device__ __forceinline__ void mul64mod(const unsigned long long& a, const unsigned long long& b, const unsigned long long& p, unsigned long long& d)
{
	uint64_t x1 = (uint64_t)(unsigned)a * (unsigned)b;
	uint64_t x2 = (uint64_t)(unsigned)(a >> 32) * (unsigned)b;
	uint64_t x3 = (uint64_t)(unsigned)a * (unsigned)(b >> 32);
	uint64_t x4 = (uint64_t)(unsigned)(a >> 32) * (unsigned)(b >> 32);

	uint128_t ux2_3 = uint128_t(x2) + x3;
	ux2_3 = ux2_3 << 32;
	uint128_t c(x1);
	//c.low = x1;
	c.high = x4;
	c = ux2_3 + c;

	uint64_t tt = ((c % p).low);

	uint128_t ux2_6(c.low);
	uint128_t Temp2 = (c.high << 32);
	ux2_6 = ux2_6 + Temp2.low;
	uint128_t ux2_4 = (c.high >> 32) + (uint64_t)((uint32_t)c.high);
	ux2_3 = ux2_6 + p - ux2_4;

	//uint128_t ux2_5 = p + c.low + (exp2(32) - 1) * (uint32_t)(c.high) - (c.high >> 32);
	//while (ux2_5 >= p)
	//{
	//	ux2_5 = ux2_5 - p;
	//}

	while (ux2_3 >= p)
	{
		ux2_3 = ux2_3 - p;
	}
	d = ux2_3.low;
}

__host__ __device__ __forceinline__ void add64mod(const unsigned long long& a, const unsigned long long& b, const unsigned long long& p, unsigned long long& d)
{

}

__host__ __device__ __forceinline__ void mod128(const uint128_t& c, const unsigned long long& p, unsigned long long& d)
{
	uint128_t ux2_6 = uint128_t(c.high << 32) + c.low + p;
	ux2_6 = ux2_6 - (c.high >> 32) - (uint64_t)((uint32_t)c.high);

	while (ux2_6 >= p)
	{
		ux2_6 = ux2_6 - p;
	}
	d = ux2_6.low;
}



__host__ __device__ __forceinline__ void mul64modSub(const unsigned long long& a, const unsigned long long& b, const unsigned long long& Subc, const unsigned long long& p, unsigned long long& d)
{
	uint64_t x1 = (uint64_t)(unsigned)a * (unsigned)b;
	uint64_t x2 = (uint64_t)(unsigned)(a >> 32) * (unsigned)b;
	uint64_t x3 = (uint64_t)(unsigned)a * (unsigned)(b >> 32);
	uint64_t x4 = (uint64_t)(unsigned)(a >> 32) * (unsigned)(b >> 32);

	uint128_t ux2_3 = uint128_t(x2) + x3;
	ux2_3 = ux2_3 << 32;
	uint128_t c(x1);
	//c.low = x1;
	c.high = x4;
	c = ux2_3 + c;

	uint128_t ux2_6(c.low);
	uint128_t Temp2 = (c.high << 32);
	ux2_6 = ux2_6 + Temp2.low;
	uint64_t ux2_4 = (c.high >> 32) + ((uint32_t)c.high);

	//while (ux2_6 >= p)
	//{
	//	ux2_6 = ux2_6 - p;
	//}

	ux2_3 = uint128_t(Subc) + ux2_4 + p + p - ux2_6;

	while (ux2_3 >= p)
	{
		ux2_3 = ux2_3 - p;
	}
	d = ux2_3.low;
}

__host__ __device__ __forceinline__ void mul64modSubNew(const unsigned long long& a, const unsigned long long& b, const unsigned long long& Subc, const unsigned long long& p, unsigned long long& d)
{
	uint32_t* x1_w = (uint32_t*)&a;
	uint32_t* x2_w = (uint32_t*)&b;

	uint64_t x1 = uint64_t(x1_w[0]) * x2_w[0]; // ac  // 需要优化
	uint64_t x2 = uint64_t(x1_w[1]) * x2_w[1]; // bd
	uint64_t x3 = uint64_t(x1_w[0]) * x2_w[1]; // ad
	uint64_t x4 = uint64_t(x1_w[1]) * x2_w[0]; // bc

	uint128_t ux2_4 = uint128_t(x3) + x4 + x2;
	ux2_4 = ux2_4 << 32;
	ux2_4 = ux2_4 + x1 - x2;

	x1_w = (uint32_t*)&(ux2_4.high);
	x1 = x1_w[0] + x1_w[1];
	ux2_4 = uint128_t(ux2_4.low) + (ux2_4.high << 32);

	uint128_t ux2_5 = (uint128_t(p) << 1) + x1 + Subc;
	//uint128_t ux2_5 = (uint128_t(p)) + p + x1 + Subc;

	/*if (ux2_4 >= ux2_5)
	{
		ux2_4 = ux2_5 + p - ux2_4;
	}
	else
	{
		ux2_4 = ux2_5 - ux2_4;
	}*/

	ux2_4 = ux2_5 - ux2_4;

	while (ux2_4 >= p)
	{
		ux2_4 = ux2_4 - p;
	}

	d = ux2_4.low;
}


__host__ __device__ __forceinline__ void modSubNewV2(const unsigned long long& mul64hi, const unsigned long long& mul64lo, const unsigned long long& Subc, const unsigned long long& p, unsigned long long& d)
{
	//uint64_t mul64hi;
	//uint64_t mul64lo;

	//mul_wide(mul64hi, mul64lo, a, b);

	//uint32_t* x1_w = (uint32_t*)&a;
	//uint32_t* x2_w = (uint32_t*)&b;

	//uint64_t x1 = uint64_t(x1_w[0]) * x2_w[0]; // ac  // 需要优化
	//uint64_t x2 = uint64_t(x1_w[1]) * x2_w[1]; // bd
	//uint64_t x3 = uint64_t(x1_w[0]) * x2_w[1]; // ad
	//uint64_t x4 = uint64_t(x1_w[1]) * x2_w[0]; // bc

	//uint128_t ux2_4 = uint128_t(x3) + x4 + x2;
	//ux2_4 = ux2_4 << 32;
	//ux2_4 = ux2_4 + x1 - x2;

	uint32_t* x1_w = (uint32_t*)&(mul64hi);
	uint64_t x1 = x1_w[0] + x1_w[1];
	uint128_t ux2_4 = uint128_t(mul64lo) + (mul64hi << 32);

	uint128_t ux2_5 = (uint128_t(p) << 1) + x1 + Subc;
	//uint128_t ux2_5 = (uint128_t(p)) + p + x1 + Subc;

	/*if (ux2_4 >= ux2_5)
	{
		ux2_4 = ux2_5 + p - ux2_4;
	}
	else
	{
		ux2_4 = ux2_5 - ux2_4;
	}*/

	ux2_4 = ux2_5 - ux2_4;

	while (ux2_4 >= p)
	{
		ux2_4 = ux2_4 - p;
	}

	d = ux2_4.low;
}

__host__ __device__ __forceinline__ void mul64modAdd(const unsigned long long& a, const unsigned long long& b, const uint128_t& Addc, const unsigned long long& p, unsigned long long& d)
{
	uint64_t x1 = (uint64_t)(unsigned)a * (unsigned)b;
	uint64_t x2 = (uint64_t)(unsigned)(a >> 32) * (unsigned)b;
	uint64_t x3 = (uint64_t)(unsigned)a * (unsigned)(b >> 32);
	uint64_t x4 = (uint64_t)(unsigned)(a >> 32) * (unsigned)(b >> 32);

	uint128_t ux2_3 = uint128_t(x2) + x3;
	ux2_3 = ux2_3 << 32;
	uint128_t c(x1);
	//c.low = x1;
	c.high = x4;
	c = ux2_3 + c;

	uint128_t ux2_6(c.low);
	uint64_t ux2_4 = (c.high >> 32) + ((uint32_t)c.high);
	ux2_6 = ux2_6 + (c.high << 32);

	ux2_3 = ux2_6 + p + Addc - ux2_4;

	while (ux2_3 >= p)
	{
		ux2_3 = ux2_3 - p;
	}
	d = ux2_3.low;
}

__host__ __device__ __forceinline__ void modAddNewV2(const unsigned long long& mul64hi, const unsigned long long& mul64lo, const unsigned long long& Adda, const unsigned long long& Addb, const unsigned long long& p, unsigned long long& d)
{
	//uint64_t mul64hi;
	//uint64_t mul64lo;

	//mad_wide(mul64hi, mul64lo, a, b, Addc);

	//uint32_t* x1_w = (uint32_t*)&a;
	//uint32_t* x2_w = (uint32_t*)&b;

	//uint64_t x1 = (x1_w[0]) * x2_w[0]; // ac  // 需要优化
	//uint64_t x2 = (x1_w[1]) * x2_w[1]; // bd
	//uint64_t x3 = (x1_w[0]) * x2_w[1]; // ad
	//uint64_t x4 = (x1_w[1]) * x2_w[0]; // bc

	//uint64_t x1 = uint64_t(x1_w[0]) * x2_w[0]; // ac  // 需要优化
	//uint64_t x2 = uint64_t(x1_w[1]) * x2_w[1]; // bd
	//uint64_t x3 = uint64_t(x1_w[0]) * x2_w[1]; // ad
	//uint64_t x4 = uint64_t(x1_w[1]) * x2_w[0]; // bc

	//uint128_t ux2_4 = uint128_t(x3) + x4 + x2;
	//ux2_4 = ux2_4 << 32;
	//ux2_4 = ux2_4 + Addc + x1 - x2;

	uint32_t* x1_w = (uint32_t*)&(mul64hi);
	uint64_t x1 = uint64_t(x1_w[0]) + x1_w[1];
	uint128_t ux2_4 = uint128_t(mul64lo) + (mul64hi << 32) + Adda + Addb; //;

	ux2_4 = ux2_4 + p - x1;

	while (ux2_4 >= p)
	{
		ux2_4 = ux2_4 - p;
	}

	d = ux2_4.low;
}

__host__ __device__ __forceinline__ void mul64modAddNew(const unsigned long long& a, const unsigned long long& b, const uint64_t& Addc, const unsigned long long& p, unsigned long long& d)
{
	uint32_t* x1_w = (uint32_t*)&a;
	uint32_t* x2_w = (uint32_t*)&b;

	//uint64_t x1 = (x1_w[0]) * x2_w[0]; // ac  // 需要优化
	//uint64_t x2 = (x1_w[1]) * x2_w[1]; // bd
	//uint64_t x3 = (x1_w[0]) * x2_w[1]; // ad
	//uint64_t x4 = (x1_w[1]) * x2_w[0]; // bc

	uint64_t x1 = uint64_t(x1_w[0]) * x2_w[0]; // ac  // 需要优化
	uint64_t x2 = uint64_t(x1_w[1]) * x2_w[1]; // bd
	uint64_t x3 = uint64_t(x1_w[0]) * x2_w[1]; // ad
	uint64_t x4 = uint64_t(x1_w[1]) * x2_w[0]; // bc

	uint128_t ux2_4 = uint128_t(x3) + x4 + x2;
	ux2_4 = ux2_4 << 32;
	ux2_4 = ux2_4 + Addc + x1 - x2;

	x1_w = (uint32_t*)&(ux2_4.high);
	x1 = x1_w[0] + x1_w[1];
	ux2_4 = uint128_t(ux2_4.low) + (ux2_4.high << 32); //;

	ux2_4 = ux2_4 + p - x1;

	while (ux2_4 >= p)
	{
		ux2_4 = ux2_4 - p;
	}

	d = ux2_4.low;
}

__device__ __forceinline__ void mul64modNewGPU(const unsigned long long& a, const unsigned long long& b, const unsigned long long& p, unsigned long long& d)
{
	//uint32_t* x1_w = (uint32_t*)&a;
	//uint32_t* x2_w = (uint32_t*)&b;

	////uint64_t x1 = (x1_w[0]) * x2_w[0]; // ac  // 需要优化
	////uint64_t x2 = (x1_w[1]) * x2_w[1]; // bd
	////uint64_t x3 = (x1_w[0]) * x2_w[1]; // ad
	////uint64_t x4 = (x1_w[1]) * x2_w[0]; // bc

	//uint64_t x1 = uint64_t(x1_w[0]) * x2_w[0]; // ac  // 需要优化
	//uint64_t x2 = uint64_t(x1_w[1]) * x2_w[1]; // bd
	//uint64_t x3 = uint64_t(x1_w[0]) * x2_w[1]; // ad
	//uint64_t x4 = uint64_t(x1_w[1]) * x2_w[0]; // bc

	//uint128_t ux2_4 = uint128_t(x3) + x4 + x2;
	//ux2_4 = ux2_4 << 32;
	//ux2_4 = ux2_4 + x1 - x2;

	uint64_t _mul64hi;
	uint64_t _mul64lo;

	mul_wide(_mul64hi, _mul64lo, a, b);

	uint32_t* x1_w = (uint32_t*)&(_mul64lo);
	uint32_t* x2_w = (uint32_t*)&(_mul64hi);

	x1_w[1] += x2_w[0];
	uint32_t TempLocal = x1_w[1] < x2_w[0];

	uint32_t TempLocal2 = x1_w[0] < x2_w[0];
	x1_w[0] -= x2_w[0];

	x1_w[1] -= TempLocal2;

	if (TempLocal == 1)
	{
		x1_w[0] += 0xffffffff;
		x1_w[1] += x1_w[0] < 0xffffffff;
	}

	if (_mul64lo < x2_w[1])
	{
		d = p - x2_w[1] + _mul64lo;
	}
	else
	{
		d = _mul64lo - x2_w[1];
	}

	if (d >= p)
	{
		d = d - p;
	}


}

__host__ __device__ __forceinline__ void mul64modNew(const unsigned long long& a, const unsigned long long& b, const unsigned long long& p, unsigned long long& d)
{
	uint32_t* x1_w = (uint32_t*)&a;
	uint32_t* x2_w = (uint32_t*)&b;

	//uint64_t x1 = (x1_w[0]) * x2_w[0]; // ac  // 需要优化
	//uint64_t x2 = (x1_w[1]) * x2_w[1]; // bd
	//uint64_t x3 = (x1_w[0]) * x2_w[1]; // ad
	//uint64_t x4 = (x1_w[1]) * x2_w[0]; // bc

	uint64_t x1 = uint64_t(x1_w[0]) * x2_w[0]; // ac  // 需要优化
	uint64_t x2 = uint64_t(x1_w[1]) * x2_w[1]; // bd
	uint64_t x3 = uint64_t(x1_w[0]) * x2_w[1]; // ad
	uint64_t x4 = uint64_t(x1_w[1]) * x2_w[0]; // bc

	uint128_t ux2_4 = uint128_t(x3) + x4 + x2;
	ux2_4 = ux2_4 << 32;
	ux2_4 = ux2_4 + x1 - x2;


	x1_w = (uint32_t*)&(ux2_4.high);
	x1 = x1_w[0] + x1_w[1];
	ux2_4 = uint128_t(ux2_4.low) + (ux2_4.high << 32); //;

	ux2_4 = ux2_4 + p - x1;

	//_uint96_modP(ux2_4.low, (uint32_t)ux2_4.high,d);

	while (ux2_4 >= p)
	{
		ux2_4 = ux2_4 - p;
	}

	d = ux2_4.low;
}

__host__ __device__ __forceinline__ void mul64(const unsigned long long& a, const unsigned long long& b, uint128_t& c)
{
	uint64_t x1 = (uint64_t)(unsigned)a * (unsigned)b;
	uint64_t x2 = (uint64_t)(unsigned)(a >> 32) * (unsigned)b;
	uint64_t x3 = (uint64_t)(unsigned)a * (unsigned)(b >> 32);
	uint64_t x4 = (uint64_t)(unsigned)(a >> 32) * (unsigned)(b >> 32);

	uint128_t ux2_3 = uint128_t(x2) + x3;

	//std::cout << std::endl << hex << x1;
	//std::cout << std::endl << hex << ux2_3.low;
	//std::cout << std::endl << hex << ux2_3.high;
	//std::cout << std::endl << hex << x4;

	ux2_3 = ux2_3 << 32;

	//std::cout << std::endl << hex << ux2_3.low;
	//std::cout << std::endl << hex << ux2_3.high;

	c.low = x1;
	c.high = x4;
	c = ux2_3 + c;

	//std::cout << std::endl << hex << c.low;
	//std::cout << std::endl << hex << c.high;
	//uint4 res;

	//asm("{\n\t"
	//	"mul.lo.u32      %3, %5, %7;    \n\t"
	//	"mul.hi.u32      %2, %5, %7;    \n\t" //alow * blow
	//	"mad.lo.cc.u32   %2, %4, %7, %2;\n\t"
	//	"madc.hi.u32     %1, %4, %7,  0;\n\t" //ahigh * blow
	//	"mad.lo.cc.u32   %2, %5, %6, %2;\n\t"
	//	"madc.hi.cc.u32  %1, %5, %6, %1;\n\t" //alow * bhigh
	//	"madc.hi.u32     %0, %4, %6,  0;\n\t"
	//	"mad.lo.cc.u32   %1, %4, %6, %1;\n\t" //ahigh * bhigh
	//	"addc.u32        %0, %0, 0;     \n\t" //add final carry
	//	"}"
	//	: "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
	//	: "r"((unsigned)(a >> 32)), "r"((unsigned)a), "r"((unsigned)(b >> 32)), "r"((unsigned)b));

	//c.high = ((unsigned long long)res.x << 32) + res.y;
	//c.low = ((unsigned long long)res.z << 32) + res.w;;
}

__host__ __device__ __forceinline__ void _uint96_modP(const unsigned long long& a, const unsigned int& b, const unsigned long long& p, unsigned long long& c)
{
	c = a;
	uint32_t* Data = (uint32_t*)&c;

	Data[1] += b;

	uint32_t TempLocal = Data[1] < b;

	Data[1] -= Data[0] < b;
	Data[0] -= b;

	if (TempLocal == 1)
	{
		Data[0] += 0xffffffff;
		Data[1] += Data[0] < 0xffffffff;
	}

	if (c >= p)
	{
		c = c - p;
	}
}

//

__device__ __forceinline__ void ff_p_add(const unsigned long long& a, const unsigned long long& b, unsigned long long& d) {
	//if (b >= MOD) {
	//    b -= MOD;
	//}

	uint64_t res_0 = a + b;
	bool over_0 = a > UINT64_MAX - b;

	uint32_t zero = 0;
	uint64_t tmp_0 = (uint64_t)(zero - (uint32_t)(over_0 ? 1 : 0));

	uint64_t res_1 = res_0 + tmp_0;
	bool over_1 = res_0 > UINT64_MAX - tmp_0;

	uint64_t tmp_1 = (uint64_t)(zero - (uint32_t)(over_1 ? 1 : 0));
	uint64_t res = res_1 + tmp_1;

	d = res;
}

__device__ __forceinline__ void ff_p_sub(const unsigned long long& a, const unsigned long long& b, unsigned long long& d) {
	//if (b >= MOD) {
	//    b -= MOD;
	//}

	uint64_t res_0 = a - b;
	bool under_0 = a < b;

	uint32_t zero = 0;
	uint64_t tmp_0 = (uint64_t)(zero - (uint32_t)(under_0 ? 1 : 0));

	uint64_t res_1 = res_0 - tmp_0;
	bool under_1 = res_0 < tmp_0;

	uint64_t tmp_1 = (uint64_t)(zero - (uint32_t)(under_1 ? 1 : 0));
	d = res_1 + tmp_1;
}

__device__ __forceinline__ void ff_p_mult(const unsigned long long& a, const unsigned long long& b, unsigned long long& d) {
	//if (b >= MOD) {
	//    b -= MOD;
	//}

	uint64_t ab = a * b;
	uint64_t cd = __umul64hi(a, b);
	uint64_t c = cd & 0x00000000ffffffff;
	uint64_t din = cd >> 32;

	uint64_t res_0 = ab - din;
	bool under_0 = ab < din;

	uint32_t zero = 0;
	uint64_t tmp_0 = (uint64_t)(zero - (uint32_t)(under_0 ? 1 : 0));
	res_0 -= tmp_0;

	uint64_t tmp_1 = (c << 32) - c;

	uint64_t res_1 = res_0 + tmp_1;
	bool over_0 = res_0 > UINT64_MAX - tmp_1;

	uint64_t tmp_2 = (uint64_t)(zero - (uint32_t)(over_0 ? 1 : 0));
	d = res_1 + tmp_2;

}

#pragma once
