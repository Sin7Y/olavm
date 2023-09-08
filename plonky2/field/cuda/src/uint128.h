//Edit by Malone and Longson
//creat data:2023.2.1

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cinttypes>
#include <string>
#include <math.h>

class uint128_t
{
public:

	uint64_t low;
	uint64_t high;

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
		while (d <= x)
		{
			x = x - d;
		}
		d = d >> 1;
	}
	while (d <= x)
	{
		x = x - d;
	}

	return x;
}

__host__ inline static uint128_t host64x2(const uint64_t& x, const uint64_t& y)
{
	uint128_t z;

	uint128_t ux(x);
	uint128_t uy(y);

	int shift = 0;

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
	uint128_t z;

	z.low = a.low - b.low;
	z.high = a.high - b.high - (a.low < b.low);
	a = z;

}

__host__ __device__ __forceinline__ void mul64mod(const uint64_t& a, const uint64_t& b, const uint64_t& p, uint64_t& d)
{
	uint64_t x1 = (uint64_t)(unsigned)a * (unsigned)b;
	uint64_t x2 = (uint64_t)(unsigned)(a >> 32) * (unsigned)b;
	uint64_t x3 = (uint64_t)(unsigned)a * (unsigned)(b >> 32);
	uint64_t x4 = (uint64_t)(unsigned)(a >> 32) * (unsigned)(b >> 32);

	uint128_t ux2_3 = uint128_t(x2) + x3;
	ux2_3 = ux2_3 << 32;
	uint128_t c(x1);
	c.high = x4;
	c = ux2_3 + c;

	uint64_t tt = ((c % p).low);

	uint128_t ux2_6(c.low);
	uint128_t Temp2 = (c.high << 32);
	ux2_6 = ux2_6 + Temp2.low;
	uint128_t ux2_4 = (c.high >> 32) + (uint64_t)((uint32_t)c.high);
	ux2_3 = ux2_6 + p - ux2_4;


	while (ux2_3 >= p)
	{
		ux2_3 = ux2_3 - p;
	}
	d = ux2_3.low;
}

__host__ __device__ __forceinline__ void mul64modSub(const uint64_t& a, const uint64_t& b, const uint64_t& Subc, const uint64_t& p, uint64_t& d)
{
	uint64_t x1 = (uint64_t)(unsigned)a * (unsigned)b;
	uint64_t x2 = (uint64_t)(unsigned)(a >> 32) * (unsigned)b;
	uint64_t x3 = (uint64_t)(unsigned)a * (unsigned)(b >> 32);
	uint64_t x4 = (uint64_t)(unsigned)(a >> 32) * (unsigned)(b >> 32);

	uint128_t ux2_3 = uint128_t(x2) + x3;
	ux2_3 = ux2_3 << 32;
	uint128_t c(x1);
	c.high = x4;
	c = ux2_3 + c;

	uint128_t ux2_6(c.low);
	uint128_t Temp2 = (c.high << 32);
	ux2_6 = ux2_6 + Temp2.low;
	uint64_t ux2_4 = (c.high >> 32) + ((uint32_t)c.high);

	ux2_3 = uint128_t(Subc) + ux2_4 + p + p - ux2_6;

	while (ux2_3 >= p)
	{
		ux2_3 = ux2_3 - p;
	}
	d = ux2_3.low;
}

__host__ __device__ __forceinline__ void mul64modAdd(const uint64_t& a, const uint64_t& b, const uint128_t& Addc, const uint64_t& p, uint64_t& d)
{
	uint64_t x1 = (uint64_t)(unsigned)a * (unsigned)b;
	uint64_t x2 = (uint64_t)(unsigned)(a >> 32) * (unsigned)b;
	uint64_t x3 = (uint64_t)(unsigned)a * (unsigned)(b >> 32);
	uint64_t x4 = (uint64_t)(unsigned)(a >> 32) * (unsigned)(b >> 32);

	uint128_t ux2_3 = uint128_t(x2) + x3;
	ux2_3 = ux2_3 << 32;
	uint128_t c(x1);
	c.high = x4;
	c = ux2_3 + c;

	uint128_t ux2_6(c.low);
	uint128_t Temp2 = (c.high << 32);
	ux2_6 = ux2_6 + Temp2.low;
	uint64_t ux2_4 = (c.high >> 32) + ((uint32_t)c.high);

	ux2_3 = ux2_6 + p + Addc - ux2_4;

	while (ux2_3 >= p)
	{
		ux2_3 = ux2_3 - p;
	}
	d = ux2_3.low;
}

__host__ __device__ __forceinline__ void mul64(const uint64_t& a, const uint64_t& b, uint128_t& c)
{
	uint64_t x1 = (uint64_t)(unsigned)a * (unsigned)b;
	uint64_t x2 = (uint64_t)(unsigned)(a >> 32) * (unsigned)b;
	uint64_t x3 = (uint64_t)(unsigned)a * (unsigned)(b >> 32);
	uint64_t x4 = (uint64_t)(unsigned)(a >> 32) * (unsigned)(b >> 32);

	uint128_t ux2_3 = uint128_t(x2) + x3;

	ux2_3 = ux2_3 << 32;
	c.low = x1;
	c.high = x4;
	c = ux2_3 + c;
}

__host__ __device__ __forceinline__ void mul64modNew(const uint64_t& a, const uint64_t& b, const uint64_t& p, uint64_t& d)
{
	uint32_t* x1_w = (uint32_t*)&a;
	uint32_t* x2_w = (uint32_t*)&b;

	uint64_t x1 = uint64_t(x1_w[0]) * x2_w[0]; 
	uint64_t x2 = uint64_t(x1_w[1]) * x2_w[1]; 
	uint64_t x3 = uint64_t(x1_w[0]) * x2_w[1]; 
	uint64_t x4 = uint64_t(x1_w[1]) * x2_w[0]; 

	uint128_t ux2_4 = uint128_t(x3) + x4 + x2;
	ux2_4 = ux2_4 << 32;
	ux2_4 = ux2_4 + x1 - x2;


	x1_w = (uint32_t*)&(ux2_4.high);
	x1 = x1_w[0] + x1_w[1];
	ux2_4 = uint128_t(ux2_4.low) + (ux2_4.high << 32); 

	ux2_4 = ux2_4 + p - x1;


	while (ux2_4 >= p)
	{
		ux2_4 = ux2_4 - p;
	}

	d = ux2_4.low;
}

#pragma once
