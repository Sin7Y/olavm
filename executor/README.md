# Olavm processor

This crate contains an implementation of Olavm VM processor. The purpose of the processor is to execute program and generate the VM's execution trace. This trace is then used by prover to generate a proof for ZKP.

## run benches

If test all bench, use: 

```
cargo bench 
```

if test only one bench, such as fibo_loop, use:

```
cargo bench -- fibo_loop
```