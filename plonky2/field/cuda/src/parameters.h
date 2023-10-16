#pragma once

#define TILE_DIM_2D					16
#define BLOCK_ROWS_2D					8

//#define TILE_DIM_3D					8
//#define BLOCK_ROWS_3D					4

#define NUM_STREAMS				  (1)

#define THREDS_PER_BLOCK          16
#define MAX_GRID                  (4096*16)
#define LENTH_TYPE                0  //1为4096测试数据  0为生产的其他长度数据
#define NTT_SIZE                  4096
#define LOG_NTT_SIZE              12
//#define NTT_SIZE2                 1024*16384//1024*1024//94194304
#define NTT_SIZE2                 (4096*4096/2)
#define LOG_NTT_SIZE2             23//94194304
#define NTT_SIZE3                 1048576//4096*16*4//4096*16//4096*4//1048576
#define LOG_NTT_SIZE3             20//18//16//14//20

#define NTT_SIZE_USED			  NTT_SIZE2

//#define NTTLEN_1D				  1024
////#define NTTLEN_2D				  ( NTT_SIZE_USED/NTTLEN_1D )
//#define NTTLEN_2D				  ( NTTLEN_1D )
//#define NTTLEN_3D				  ( NTT_SIZE_USED/NTTLEN_1D/NTTLEN_2D )

#define MAX_THREADBLK				  1024
#define TRANS_TIMES				  1

#define BDIMX 16
#define BDIMY 16
#define IPAD 2
#define LEN_BOUNDARY			  NTTLEN_1D