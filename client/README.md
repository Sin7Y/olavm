# Client
This is standalone client of Olavm. 

Olavm comand:

Execute an input raw-code file.

subcommand:

* asm

parameters:

-i: input the file contain Ola-lang assemble code.
-o: output the file contain OlaVM executable instruction code.

```
ola asm -i assembler/testdata/fibo.asm -o fibo.json
```

* run

parameters:

-i: input raw-code file for executing
-o: output trace_table json file

```
ola run -i fibo.json -o trace_table.txt
```
