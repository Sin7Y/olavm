# Olavm circuits

There are 3 types of circuits in Olavm:
* [CPU](./src/cpu/) -- check and constrain all CPU logic, e.g. `ADD`, `JMP`, `RET` etc.
* [RAM](./src/memory) -- check and constrain all memory logic, e.g. memory consistency.
* [Builtin](./src/builtins/) -- plugin modules that has specific scenarios, e.g. `AND`, `RANGE_CHECK` etc.

