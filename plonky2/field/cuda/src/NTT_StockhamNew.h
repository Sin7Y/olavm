
#ifndef NTT_STOCKHAMNEW_H
#define NTT_STOCKHAMNEW_H

#define WARP 32

//int device = 0;

class NTT_ConstParams {
public:
	static const int NTT_exp = -1;
	static const int NTT_length = -1;
	static const int NTT_half = -1;
	static const int warp = 32;
};

class NTT_64 : public NTT_ConstParams {
public:
	static const int NTT_exp = 6;
	static const int NTT_quarter = 16;
	static const int NTT_half = 32;
	static const int NTT_threequarters = 48;
	static const int NTT_length = 64;

};

class NTT_128 : public NTT_ConstParams {
public:
	static const int NTT_exp = 7;
	static const int NTT_quarter = 32;
	static const int NTT_half = 64;
	static const int NTT_threequarters = 96;
	static const int NTT_length = 128;

};

class NTT_256 : public NTT_ConstParams {
public:
	static const int NTT_exp = 8;
	static const int NTT_quarter = 64;
	static const int NTT_half = 128;
	static const int NTT_threequarters = 192;
	static const int NTT_length = 256;

};

class NTT_512 : public NTT_ConstParams {
public:
	static const int NTT_exp = 9;
	static const int NTT_quarter = 128;
	static const int NTT_half = 256;
	static const int NTT_threequarters = 384;
	static const int NTT_length = 512;
};

class NTT_1024 : public NTT_ConstParams {
public:
	static const int NTT_exp = 10;
	static const int NTT_quarter = 256;
	static const int NTT_half = 512;
	static const int NTT_threequarters = 768;
	static const int NTT_length = 1024;
};

class NTT_2048 : public NTT_ConstParams {
public:
	static const int NTT_exp = 11;
	static const int NTT_quarter = 512;
	static const int NTT_half = 1024;
	static const int NTT_threequarters = 1536;
	static const int NTT_length = 2048;
};

class NTT_4096 : public NTT_ConstParams {
public:
	static const int NTT_exp = 12;
	static const int NTT_quarter = 1024;
	static const int NTT_half = 2048;
	static const int NTT_threequarters = 3072;
	static const int NTT_length = 4096;
};


//template<class const_params>
//__device__ void do_NTT_Stockham_mk6(uint64_t* s_input, const uint64_t* weightUser);
//
//template<class const_params>
//__global__ void NTT_GPU_external(uint64_t* d_input, uint64_t* d_output, const uint64_t* weightUser);
//
//template<class const_params>
//__global__ void NTT_GPU_multiple(uint64_t* d_input, uint64_t* d_output, const uint64_t* weightUser);

//int Max_columns_in_memory_shared(int NTT_size, int nNTTs);


#endif