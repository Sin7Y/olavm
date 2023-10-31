#[repr(C)]
struct NTTParam {
    G: u64,
    P: u64,
    wCoeff: u64,
    numSTREAMS: c_int,
    streams: *mut cudaStream_t,
    NTTLen: u64,
    NTTLen_Inverse: bool,
    d_round_one: *mut u64,
    d_round_two: *mut u64,
    h_dataIn: *mut u64,
    h_dataOut: *mut u64,
    cudatwiddleFactorArray2D_coeff: *mut u64,
    cudatwiddleFactorArray3D_coeff: *mut u64,
    cudatwiddleFactorArray_Normcoeff: *mut u64,
    NTTLen1D: u64,
    NTTLen1D_blkNum: u32,
    cudawCoeff1D_weight: *mut u64,
    NTTLen2D: u64,
    NTTLen2D_blkNum: u32,
    cudawCoeff2D_weight: *mut u64,
    NTTLen3D: u64,
    NTTLen3D_blkNum: u32,
    cudawCoeff3D_weight: *mut u64,
    twiddleSymbol_1st: *mut u64,
    twiddleSymbol_2nd: *mut u64,
}

// 定义NTTParamFB结构体
#[repr(C)]
struct NTTParamFB {
    NTTParamForward: *mut NTTParam,
    NTTParamBackward: *mut NTTParam,
}

// 定义NTTParamGroup结构体
#[repr(C)]
struct NTTParamGroup {
    pNTTParamFB: *mut NTTParamFB,
    DataLen: u64,
}