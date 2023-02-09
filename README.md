# OlaVM

[![LICENSE](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Sin7Y/olavm/blob/main/LICENSE)
[![CI checks](https://github.com/Sin7Y/olavm/actions/workflows/rust.yml/badge.svg)](https://github.com/Sin7Y/olavm/actions/workflows/unit_test.yml)
[![issues](https://img.shields.io/github/issues/Sin7Y/olavm)](https://github.com/Sin7Y/olavm/issues?q=is%3Aopen)

OlaVM is a STARK-Based ZK-Friendly ZKVM, it builds on a small finite field which is called [Goldilocks field](https://github.com/mir-protocol/plonky2/blob/main/field/src/goldilocks_field.rs). As the most important component of Ola system, OlaVM is mainly used to execute a program and generate a valid proof for the programmable scalable case and programmable private case.

Warning: This repository shouldn't be used for production case, as it always at development phase and has not been audited, so it maybe contain many bugs and security flaws.

## Overview

OlaVM is a turing complete VM which means that it can execute any computation on it and as the same time it could generate a valid proof for it. For getting a smaller prove time, we have many powerful designs that relevant with ZK-Friendly.

- if you want to know more about ZK-friendly and VM designs, check out the [doc];
- if you want to know more about the circuit design, check out [circuit](circuits) crate;
- if you want to know more about the performance of OlaVM, check out [Performance] section;
- if you want to know more about STARKS, check out [Reference] section;

### KeyFeatures

There are a lot of tricks to get a very ZK-friendly ZKVM in OlaVM. We would like to highlight a few of them:

- Algebraic RISC. The property of instruction set of OlaVM is Algebraic RISC: "Algebraic" refers to the supported operation are field operation, "RISC" refers to the minimality of the instruction set. We can achieve a succinct transition constraints based on this, check out [circuit/cpu](https://github.com/Sin7Y/olavm/tree/main/circuits/src/cpu) to learn more;
- Small finite field. The word defined in OlaVM is a finite field, [Goldilocks]((https://github.com/mir-protocol/plonky2/blob/main/field/src/goldilocks_field.rs)). The prime of Goldilicks is p = 2^64 - 2^32 + 1 which is less than 64 bits. The computation based on those field elements could be exected much faster that other big finite field;
- Builtins. As the cyclic group size is limited, so it would be better of the trace table could contain more transactions as much as possible. That means that if there some computation cost a large trace rows in transation logic, we should remove it from main trace table and add a specific sub trace table to store them, this is the reason that introduce Builtins, like hash, bitwise operation and so on, check out the [doc] for more details;
- Prophets. This is designed for non-deterministic computation which means "implementation is expensive, verify is cheap". So up to a point, prophets is more like a external helper. It helps you compute the results of some non-deterministic computation and then you check the results by using the instruction sets, should be note that this check is execute by VM, not constraits which is different with Builtins, check out the [doc] for more details;

### Status

| Features                   |         Status         |
|----------------------------|------------------------|
| Algebraic RISC             | $\color{Green}{Done}$  |
| Small finite field         | $\color{Green}{Done}$  |
| Builtins - bitwise         | $\color{Green}{Done}$  |
| Builtins - rangecheck      | $\color{Green}{Done}$  |
| Builtins - cmp             | $\color{Green}{Done}$  |
| Builtins - poseidon        | $\color{Golden}{Doing}$|
| Builtins - ecdsa           | $\color{Golden}{Doing}$|
| Prover optimization        | $\color{Golden}{Doing}$|
| Prophets lib               | $\color{Red}{Todo}$    |
| U32/64/256 lib             | $\color{Red}{Todo}$    |
| Support privacy            | $\color{Red}{Todo}$    |

### Project structure

This project consists of several crates:

| Crate                      | Description |
|----------------------------|-------------|
| [core](core)               | Define instruction structure and instruction sets       |
| [circuits](circuits)       | 1. Constraints for instruction sets, builtins, memory; 2. Generate proof        |
| [executor](executor)       | Execute program and generate the execution trace for Ola-prover |
| [client](client)           | Some commands can be used for developer        |
| [plonky2](plonky2)         | A SNARK implementation based on techniques from PLONK and FRI   |
| [infrastructure](circuits) | Write the execution trace to a excel file       |

## Performance

Many optimizations have not been applied yet, and we expect that there will be some speedup once we dedicate some time to performance optimizations. The benchmarks below should be viewed only as a rough guide for expected future performance.

In the benchmarks below, the VM executes the same Fibonacci calculator program for 2^20 cycles at 100-bit target security level on a high-end 64-core CPU:

| VM cycles | Execution time | Proving time | RAM consumed | Proof size |
|-----------|----------------|--------------|--------------|------------|
| 2^18      | 81.115 ms      | 6.791 s      | 5.6 GB       | 478 KB     |
| 2^19      | 159.80 ms      | 14.688 s     | 11.1 GB      | 496 KB     |
| 2^20      | 318.08 ms      | 29.766 s     | 23.2 GB      | 515 KB     |
| 2^21      | 627.38 ms      | 65.057 s     | 45.3 GB      | 533 KB     |
| 2^22      | 1240.4 ms      | 133.08 s     | 86.6 GB      | 576 KB     |
| 2^23      | 2453.8 ms      | 271.04 s     | 176 GB       | 597 KB     |

Overall, we don't expect the benchmarks to change significantly, but there will definitely be some deviation from the below numbers in the future.

A few general notes on performance:

Software:

- Support the fastest hash algorithm, [Blake3 Hash](https://github.com/BLAKE3-team/BLAKE3-specs/blob/master/blake3.pdf)
- Support parallel fft
- Support parallel polynomial evaluation
- Support parallel proof

Hardware:

- Integrate GPU accelerated FFT
- Integrate GPU accelerated polynomial evaluation
- Integrate FPGA accelerated FFT
- Integrate FPGA accelerated polynomial evaluation

## References

OlaVM is defined on Goldilocks field and use STARK to generate proof for inner layer and more will use Plonky2 to generate recursive proof for inner proofs. There is some resources to learn more about them.

### Goldilocks field & plonky2

- [Goldilocks field used in winterfell](https://github.com/novifinancial/winterfell/blob/main/math/src/field/f64/mod.rs)
- [Goldilocks field used in plonky2](https://github.com/mir-protocol/plonky2/blob/main/field/src/goldilocks_field.rs)
- [Goldilocks field used in Polygon-Hermez](https://github.com/0xPolygonHermez/goldilocks)
- [Goldilock field paraments and extension Goldilocks](https://cronokirby.com/notes/2022/09/the-goldilocks-field/)
- [U32 implemention in Goldilocks]((https://hackmd.io/@bobbinth/H1aCWWy7F) )
- [Plonky2 whitepaper](https://github.com/mir-protocol/plonky2/blob/main/plonky2/plonky2.pdf)

### STARK

- [STARK](https://eprint.iacr.org/2018/046.pdf)
- [ethSTARK](https://eprint.iacr.org/2021/582.pdf)
- [Deep-FRI](https://arxiv.org/pdf/1903.12243.pdf)
- [FRI](https://drops.dagstuhl.de/opus/volltexte/2018/9018/pdf/LIPIcs-ICALP-2018-14.pdf)

Vitalik Buterin's blog series on zk-STARKs:

- [Part-1](https://vitalik.ca/general/2017/11/09/starks_part_1.html): Proofs with Polynomials
- [Part-2](https://vitalik.ca/general/2017/11/22/starks_part_2.html): Thank Goodness It's FRI-day
- [Part-3](https://vitalik.ca/general/2018/07/21/starks_part_3.html): Into the Weeds

Alan Szepieniec's STARK tutorials:

- [Part-0](https://aszepieniec.github.io/stark-anatomy/): Introduction
- [Part-1](https://aszepieniec.github.io/stark-anatomy/overview): STARK Overview
- [Part-2](https://aszepieniec.github.io/stark-anatomy/basic-tools): Basic Tools
- [Part-3](https://aszepieniec.github.io/stark-anatomy/fri): FRI
- [Part-4](https://aszepieniec.github.io/stark-anatomy/stark): The STARK Polynomial IOP
- [Part-5](https://aszepieniec.github.io/stark-anatomy/rescue-prime): A Rescue-Prime STARK
- [Part-6](https://aszepieniec.github.io/stark-anatomy/faster): Speeding Things Up

[Sin7Y](https://twitter.com/Sin7y_Labs)'s STARK explaination:

- [STARK - An Indepth Technical Analysis]((https://hackmd.io/@sin7y/HktwgoeKq))
- [The Stark Proof System of Miden V1](https://hackmd.io/@sin7y/HkIELMUu9)

## License

This project is under [MIT licensed](./LICENSE).