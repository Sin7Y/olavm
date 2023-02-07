mov r0 1
mov r2 1
mstore 128 r0
mstore 135 r0
mov r0 8
mov r3 0
.LBL_0_0
EQ r0 r3
cjmp LBL_0_1
mload r1 128
assert r1 r2
mload r2 135
add r4 r1 r2
mstore 128 r2
mstore 135 r4
mov r4 1
add r3 r3 r4
jmp LBL_0_0
.LBL_0_1
range r3
end