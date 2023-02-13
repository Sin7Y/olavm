# Ola Operating manuals

## 1.Download & Installation vscode 

Before writing an ola contract we recommend using vscode as an editor. You can download vscode from here https://code.visualstudio.com/

## 2. Installation Ola vscode Extension

Ola supports writing on vscode, we have developed an extension to vscode to support ola syntax highlighting, and we will continue to improve the plugin in the future.

The extension can be found on the [Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=Sin7y.ola).

## 3. Write an ola contract

The following shows a smart contract that calculates the tenth term of the Fibonacci number

```
contract Fibonacci {

    fn main() {
       fib_recursive(10);
    }

    fn fib_recursive(u32 n) -> (u32) {
        if (n == 1) {
            return 1;
        }
        if (n == 2) {
            return 1;
        }

        return fib_recursive(n -1) + fib_recursive(n -2);
    }

}
```

You can copy it to vscode,  With the ola extension, the code in vscode looks like this

![image-20230213151421535](https://s2.loli.net/2023/02/13/Bh5w4xOuWird1sC.png)

**Note:** We are not yet able to support too complex smart contract statements such as For Loop, global variables, etc. If you encounter a compilation error, relax. We will fix it in a later compiler update :grinning: :grinning: .

## 4. Compiling smart contracts

Once you have finished writing the smart contract, you need to compile the smart contract source code into an assembly format that OlaVM can execute. So next you need to prepare the compilation environment

### 4.1 Prerequisites

#### 4.1.1 [Installation Rust](https://www.rust-lang.org/tools/install)

You can install Rust by entering the following command in your terminal

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

rustup toolchain install nightly

rustup default nightly
```

#### 4.1.2  [Installation LLVM](https://releases.llvm.org/)

In order to build the ola-lang project, you need to have LLVM installed on your system.

For macos, installing llvm with brew is very easy.

You can install LLVM by entering the following command in your terminal

```bash
brew install llvm@14

echo 'export PATH="/usr/local/opt/llvm@14/bin:$PATH"' >> ~/.bash_profile

source ~/.bash_profile

```

You can check if llvm is installed successfully by typing `llvm-config --version` in the terminal 

 #### 4.1.3 Download ola-lang project

You can download ola-lang  repository by entering the following command in your terminal

```bash
git clone https://github.com/Sin7Y/ola-lang.git
cd ola-lang
cargo install --path .
```

The executable `olac` will be install in you environment. You can check this by entering the `olac --help `command

### 4. 2 compile ola contract 

The olac compiler is run on the command line. The ola source file names are provided as command line arguments; the output is an ola asm.

You can compile ola contract by entering the following command in your terminal

```
olac compile ./examples/fib.ola --gen asm
```

After the above command is executed, a file named `fib.asm` will be created in the current folder.

## 5. Generate & Verify proof

The OlaVM will read the fib.asm file in the current directory, execute the opcode contained in that file, and generate an execution trace to prove that the system generated the proof.

Before generating proofs, some preparatory work needs to be done

### 5.1 Download olavm project

You can download olavm  repository by entering the following command in your terminal

```bash
git clone https://github.com/Sin7Y/olavm.git

cd olavm/client/

cargo install --path .
```

The executable `ola` will be install in you environment. You can check this by entering the `ola --help `command

### 5.2 assembly of smart contract

You can assembly ola contract asm format file  by entering the following command in your terminal

```bash
ola asm --input fib.asm --output fib.code
```

After the above command is executed, a file named `fib.code` will be created in the current folder.

### 5.3 Execute smart contract code

You can execute ola contract code by entering the following command in your terminal

```bash
ola run --input fib.code --output fib_trace.json
```

After the above command is executed, a file named `fib_trace.json` will be created in the current folder.

### 5.4 Generate proof

You can generate proof  by entering the following command in your terminal

```bash
ola prove --input fib_trace.json --output fib.proof
```

After the above command is executed, a file named `fib.proof` will be created in the current folder.

### 5.5 Verify proof

You can verify proof  by entering the following command in your terminal

```bash
ola verify --input fib.proof
```

When you see output like the following, congratulations, it's all done

```bash
❯ ola verify --input fib.proof
Loading proof...nput fib.proof                
Input file path: fib.proof
Verify succeed!
```

