$a=2
init:
    input r0
    mov r2 1
    mov r3 1
    cjmp :end
loop:
    mov r1 r2
    mov r2 $a
    add r3 r1 r2
    sub r0 r0 1
    gt r0 2
    cjmp :loop
end:
    ret r3