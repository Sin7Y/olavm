# OlaVM

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Sin7Y/olavm/blob/main/LICENSE)
[![CI checks](https://github.com/Sin7Y/olavm/actions/workflows/rust.yml/badge.svg)](https://github.com/Sin7Y/olavm/actions/workflows/unit_test.yml)
[![issues](https://img.shields.io/github/issues/Sin7Y/olavm)](https://github.com/Sin7Y/olavm/issues?q=is%3Aopen)

OlaVM is a STARK-based ZK-friendly ZKVM, it is built on a **small finite field** called [Goldilocks field](https://github.com/mir-protocol/plonky2/blob/main/field/src/goldilocks_field.rs). As the most important component of the Ola system, OlaVM is mainly used to execute a program and generate a valid proof for the **programmable scalable** case and the **programmable private** case.

**Warning: This repository shouldn't be used for production case, as it is still in development phase and has not been audited, so it might contain many bugs and security holes.**.

## Overview

OlaVM is a Turing complete VM, which means that it can execute any computation on it and at the same time it could generate a valid proof for it. For getting a smaller proof time, we have many powerful designs that are relevant with ZK-friendly.

- if you want to know more about ZK-friendly and VM designs, check out the [doc/olavm](https://github.com/Sin7Y/olavm/blob/main/docs/olavm/olavm_sepc.pdf);
- if you want to know more about the circuit design, check out [circuit](circuits) crate;
- if you want to know more about the performance of OlaVM, check out [Performance](#performance) section;
- if you want to know more about STARKS, check out [References](#references) section;

### Key Features

There are a lot of tricks to get a very ZK-friendly ZKVM in OlaVM. We would like to highlight some of them:

- **Algebraic RISC**. The property of the instruction set of OlaVM is Algebraic RISC: "Algebraic" refers to the supported operations are field operation, "RISC" refers to the minimality of the instruction set. We can achieve a concise transition constraint based on this, see [circuit/cpu](https://github.com/Sin7Y/olavm/tree/main/circuits/src/cpu) to learn more;

- **Small finite field**. The word defined in OlaVM is a finite field, [Goldilocks](https://github.com/mir-protocol/plonky2/blob/main/field/src/goldilocks_field.rs). The prime of Goldilocks is p = 2^64 - 2^32 + 1, which is less than 64 bits. The computation based on these field elements could be expected to be [much faster]((https://twitter.com/rel_zeta_tech/status/1622984483359129601)) than other large finite fields;

- **Builtins**. Since the cyclic group size is limited, it would be better if the trace table could contain as many transactions as possible. This means that if there are some computation cost a large trace lines in transaction logic, we should remove them from the main trace table and add a special sub-trace table to store them, this is the reason that introduce builtins, like hash, bitwise operation and so on, check the [doc/olavm](https://github.com/Sin7Y/olavm/blob/main/docs/olavm/olavm_sepc.pdf) for more details;

- **Prophets**. It is designed for non-deterministic computation, which means "implementation is expensive, verification is cheap". So to some extent Prophets is more like an external helper. It helps you compute the results of some non-deterministic computation, and then you verify the results using the instruction sets, should note that this verification is done by the VM, not by constraints which is different with builtins, see the [doc/olavm](https://github.com/Sin7Y/olavm/blob/main/docs/olavm/olavm_sepc.pdf) for more details;

### Status

| Features                   |         Status         |
|----------------------------|------------------------|
| Algebraic RISC             | $\color{Green}{Done}$  |
| Small finite field         | $\color{Green}{Done}$  |
| Builtins - bitwise         | $\color{Green}{Done}$  |
| Builtins - rangecheck      | $\color{Green}{Done}$  |
| Builtins - cmp             | $\color{Green}{Done}$  |
| Builtins - poseidon        | $\color{Yellow}{Doing}$|
| Builtins - ecdsa           | $\color{Yellow}{Doing}$|
| Prover optimization        |  $\color{Green}{Done}$|
| Prophets lib               | $\color{Red}{Todo}$    |
| u32/u64/u256 lib           | $\color{Red}{Todo}$    |
| Support privacy            | $\color{Red}{Todo}$    |

### Project structure

This project consists of several crates:

| Crate                      | Description |
|----------------------------|-------------|
| [core](core)               | Define instruction structure and instruction sets       |
| [circuits](circuits)       | 1. Constraints for instruction sets, builtins, memory; 2. Generate proof        |
| [executor](executor)       | Execute the programme and generate the execution trace for the Ola-prover |
| [client](client)           | Some commands can be used by developers        |
| [plonky2](plonky2)         | A SNARK implementation based on techniques from PLONK and FRI techniques  |
| [infrastructure](circuits) | Write the execution trace to an Excel file       |

## Performance

Many optimizations have not yet been applied, and we expect to see some speed improvements as we devote more time to performance optimization. The benchmarks below should only be used as a rough guide to expected future performance.

In the benchmarks below, the VM executes the same Fibonacci calculator program for 2^20 cycles at 100-bit target security level on a high-end 64-core CPU:

| VM cycles | Execution time | Proving time | RAM consumed | Proof size |
|-----------|----------------|--------------|--------------|------------|
| 2^18      | 81.115 ms      | 6.791 s      | 5.6 GB       | 175 KB     |
| 2^19      | 159.80 ms      | 14.688 s     | 11.1 GB      | 181 KB     |
| 2^20      | 318.08 ms      | 29.766 s     | 23.2 GB      | 187 KB     |
| 2^21      | 627.38 ms      | 65.057 s     | 45.3 GB      | 195 KB     |
| 2^22      | 1240.4 ms      | 133.08 s     | 86.6 GB      | 208 KB     |
| 2^23      | 2453.8 ms      | 271.04 s     | 176 GB       | 216 KB     |

Overall, we don't expect the benchmarks to change significantly, but there will definitely be some deviation from the numbers below in the future.

A few general notes on performance:

Software:

- Support the fastest hash algorithm, [Blake3 Hash](https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.pdf) $\color{Green}{Done}$
- Support [parallel FFT](https://github.com/facebook/winterfell/tree/main/math/src/fft) $\color{Green}{Done}$
- Support [parallel polynomial evaluation](https://github.com/facebook/winterfell/tree/main/math/src/fft) $\color{Green}{Done}$
- Support parallel proof

Hardware:

- Integrate GPU accelerated FFT
- Integrate GPU accelerated polynomial evaluation
- Integrate FPGA accelerated FFT
- Integrate FPGA accelerated polynomial evaluation

## References

OlaVM runs based on the Goldilocks field and uses STARK to generate proofs for the inner layer and more will use Plonky2 to generate recursive proofs for inner proofs. There are some resources to learn more about it.

### Goldilocks field & plonky2

- [Goldilocks field used in Winterfell](https://github.com/novifinancial/winterfell/blob/main/math/src/field/f64/mod.rs)
- [Goldilocks field used in plonky2](https://github.com/mir-protocol/plonky2/blob/main/field/src/goldilocks_field.rs)
- [Goldilocks field used in Polygon-Hermez](https://github.com/0xPolygonHermez/goldilocks)
- [Goldilocks field and extension Goldilocks](https://cronokirby.com/notes/2022/09/the-goldilocks-field/)
- [U32 implementation in Goldilocks](https://hackmd.io/@bobbinth/H1aCWWy7F)
- [Plonky2 whitepaper](https://github.com/mir-protocol/plonky2/blob/main/plonky2/plonky2.pdf)

### STARK

- [STARK](https://eprint.iacr.org/2018/046.pdf)
- [ethSTARK](https://eprint.iacr.org/2021/582.pdf)
- [Deep-FRI](https://arxiv.org/pdf/1903.12243.pdf)
- [FRI](https://drops.dagstuhl.de/opus/volltexte/2018/9018/pdf/LIPIcs-ICALP-2018-14.pdf)

### Vitalik Buterin's blog series on zk-STARKs:

- [Part-1](https://vitalik.ca/general/2017/11/09/starks_part_1.html): Proofs with Polynomials
- [Part-2](https://vitalik.ca/general/2017/11/22/starks_part_2.html): Thank Goodness It's FRI-day
- [Part-3](https://vitalik.ca/general/2018/07/21/starks_part_3.html): Into the Weeds

### Alan Szepieniec's STARK tutorials:

- [Part-0](https://aszepieniec.github.io/stark-anatomy/): Introduction
- [Part-1](https://aszepieniec.github.io/stark-anatomy/overview): STARK Overview
- [Part-2](https://aszepieniec.github.io/stark-anatomy/basic-tools): Basic Tools
- [Part-3](https://aszepieniec.github.io/stark-anatomy/fri): FRI
- [Part-4](https://aszepieniec.github.io/stark-anatomy/stark): The STARK Polynomial IOP
- [Part-5](https://aszepieniec.github.io/stark-anatomy/rescue-prime): A Rescue-Prime STARK
- [Part-6](https://aszepieniec.github.io/stark-anatomy/faster): Speeding Things Up

### [Sin7Y](https://twitter.com/Sin7y_Labs)'s STARK explanation:

- [STARK - An Indepth Technical Analysis](https://hackmd.io/@sin7y/HktwgoeKq)
- [The Stark Proof System of Miden V1](https://hackmd.io/@sin7y/HkIELMUu9)

### Privacy:
- [Private delagation for zksnark](https://www.youtube.com/watch?v=mFzwp8gGn-E)
- [ZEXE: Private computation](https://eprint.iacr.org/2018/962.pdf)
- [Zcash protocol](https://github.com/zcash/zips/blob/main/protocol/protocol.pdf)
- [DPC on Aleo](https://www.youtube.com/watch?v=uMmAUssK-PA&t=1705s)

## License

This project is under [MIT licensed](./LICENSE).
