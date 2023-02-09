# OlaVM

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Sin7Y/olavm/blob/main/LICENSE)
[![CI checks](https://github.com/Sin7Y/olavm/actions/workflows/rust.yml/badge.svg)](https://github.com/Sin7Y/olavm/actions/workflows/unit_test.yml)
[![issues](https://img.shields.io/github/issues/Sin7Y/olavm)](https://github.com/Sin7Y/olavm/issues?q=is%3Aopen)

// todo a short general introduce in one line and warn that this project has not finished yet.

## Overview

// todo general introduce of OlaVM, may link to some articles of OlaVM learning resources.

### Features

// high light features, including implemented and to be implemented.

### Status

// project status and our plan.

### Project structure

This project consists of several crates:

| Crate                      | Description |
|----------------------------|-------------|
| [core](core)               | todo        |
| [circuits](circuits)       | todo        |
| [executor](executor)       | todo        |
| [client](client)           | todo        |
| [plonky2](plonky2)         | todo        |
| [infrastructure](circuits) | todo        |

## Performance

Many optimizations have not been applied yet, and we expect that there will be some speedup once we dedicate some time to performance optimizations. The benchmarks below should be viewed only as a rough guide for expected future performance. 

Overall, we don't expect the benchmarks to change significantly, but there will definitely be some deviation from the below numbers in the future.

A few general notes on performance:
- replace poseidon hash function to blake3 hash function(!fix me)
- replace concurrent fft(!fix me)

In the benchmarks below, the VM executes the same Fibonacci calculator program for 2^20 cycles at 100-bit target security level on a high-end 64-core CPU:

| VM cycles | Execution time | Proving time | RAM consumed | Proof size |
|-----------|----------------|--------------|--------------|------------|
| 2^18      | 81.115 ms      | 6.791 s      | 5.6 GB       | 478 KB     |
| 2^19      | 159.80 ms      | 14.688 s     | 11.1 GB      | 496 KB     |
| 2^20      | 318.08 ms      | 29.766 s     | 23.2 GB      | 515 KB     |
| 2^21      | 627.38 ms      | 65.057 s     | 45.3 GB      | 533 KB     |
| 2^22      | 1240.4 ms      | 133.08 s     | 86.6 GB      | 576 KB     |
| 2^23      | 2453.8 ms      | 271.04 s     | 176 GB       | 597 KB     |

## References

// todo learning resources of math, zkp; some other projects.

## License

This project is under [MIT licensed](./LICENSE).