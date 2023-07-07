//Edit by Piaobo
//data:2023.2.15


#define CPURUAN                   0
#define BATCHSIZE                 1
#define THREDS_PER_BLOCK          64
#define MAX_GRID                  128
#define LENTH_TYPE                0  //1为4096测试数据  0为生产的其他长度数据
#define NTT_SIZE                  4096
#define LOG_NTT_SIZE              12
#define NTT_SIZE2                 512*16384//1024*1024//94194304
#define LOG_NTT_SIZE2             24//94194304
#define NTT_SIZE3                 1048576//4096*16*4//4096*16//4096*4//1048576
#define LOG_NTT_SIZE3             20//18//16//14//20

#define N_INVERSE_LOG_20          18446726477228544001
#define NUM_STREAMS				  8
#define KENEL_FUNCTION			  1//不同的核函数   0为初始版本  1为优化版本

