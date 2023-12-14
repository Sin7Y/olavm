; ModuleID = 'ContractDeployer'
source_filename = "ContractDeployer"

@heap_address = internal global i64 -12884901885

declare void @builtin_assert(i64)

declare void @builtin_range_check(i64)

declare i64 @prophet_u32_sqrt(i64)

declare i64 @prophet_u32_div(i64, i64)

declare i64 @prophet_u32_mod(i64, i64)

declare ptr @prophet_u32_array_sort(ptr, i64)

declare void @get_context_data(ptr, i64)

declare void @get_tape_data(ptr, i64)

declare void @set_tape_data(ptr, i64)

declare void @get_storage(ptr, ptr)

declare void @set_storage(ptr, ptr)

declare void @poseidon_hash(ptr, ptr, i64)

declare void @contract_call(ptr, i64)

declare void @prophet_printf(i64, i64)

define ptr @heap_malloc(i64 %0) {
entry:
  %size_alloca = alloca i64, align 8
  store i64 %0, ptr %size_alloca, align 4
  %size = load i64, ptr %size_alloca, align 4
  %current_address = load i64, ptr @heap_address, align 4
  %updated_address = add i64 %current_address, %size
  store i64 %updated_address, ptr @heap_address, align 4
  %1 = inttoptr i64 %current_address to ptr
  ret ptr %1
}

define ptr @vector_new(i64 %0) {
entry:
  %size_alloca = alloca i64, align 8
  store i64 %0, ptr %size_alloca, align 4
  %size = load i64, ptr %size_alloca, align 4
  %1 = add i64 %size, 1
  %current_address = load i64, ptr @heap_address, align 4
  %updated_address = add i64 %current_address, %1
  store i64 %updated_address, ptr @heap_address, align 4
  %2 = inttoptr i64 %current_address to ptr
  store i64 %size, ptr %2, align 4
  ret ptr %2
}

define void @memcpy(ptr %0, ptr %1, i64 %2) {
entry:
  %index_alloca = alloca i64, align 8
  %len_alloca = alloca i64, align 8
  %dest_ptr_alloca = alloca ptr, align 8
  %src_ptr_alloca = alloca ptr, align 8
  store ptr %0, ptr %src_ptr_alloca, align 8
  %src_ptr = load ptr, ptr %src_ptr_alloca, align 8
  store ptr %1, ptr %dest_ptr_alloca, align 8
  %dest_ptr = load ptr, ptr %dest_ptr_alloca, align 8
  store i64 %2, ptr %len_alloca, align 4
  %len = load i64, ptr %len_alloca, align 4
  store i64 0, ptr %index_alloca, align 4
  br label %cond

cond:                                             ; preds = %body, %entry
  %index_value = load i64, ptr %index_alloca, align 4
  %loop_cond = icmp ult i64 %index_value, %len
  br i1 %loop_cond, label %body, label %done

body:                                             ; preds = %cond
  %src_index_access = getelementptr i64, ptr %src_ptr, i64 %index_value
  %3 = load i64, ptr %src_index_access, align 4
  %dest_index_access = getelementptr i64, ptr %dest_ptr, i64 %index_value
  store i64 %3, ptr %dest_index_access, align 4
  %next_index = add i64 %index_value, 1
  store i64 %next_index, ptr %index_alloca, align 4
  br label %cond

done:                                             ; preds = %cond
  ret void
}

define i64 @memcmp_eq(ptr %0, ptr %1, i64 %2) {
entry:
  %index_alloca = alloca i64, align 8
  %len_alloca = alloca i64, align 8
  %right_ptr_alloca = alloca ptr, align 8
  %left_ptr_alloca = alloca ptr, align 8
  store ptr %0, ptr %left_ptr_alloca, align 8
  %left_ptr = load ptr, ptr %left_ptr_alloca, align 8
  store ptr %1, ptr %right_ptr_alloca, align 8
  %right_ptr = load ptr, ptr %right_ptr_alloca, align 8
  store i64 %2, ptr %len_alloca, align 4
  %len = load i64, ptr %len_alloca, align 4
  store i64 0, ptr %index_alloca, align 4
  br label %cond

cond:                                             ; preds = %body, %entry
  %index_value = load i64, ptr %index_alloca, align 4
  %loop_check = icmp ult i64 %index_value, %len
  br i1 %loop_check, label %body, label %done

body:                                             ; preds = %cond
  %left_elem_ptr = getelementptr i64, ptr %left_ptr, i64 %index_value
  %left_elem = load i64, ptr %left_elem_ptr, align 4
  %right_elem_ptr = getelementptr i64, ptr %right_ptr, i64 %index_value
  %right_elem = load i64, ptr %right_elem_ptr, align 4
  %compare = icmp eq i64 %left_elem, %right_elem
  %next_index = add i64 %index_value, 1
  store i64 %next_index, ptr %index_alloca, align 4
  br i1 %compare, label %cond, label %done

done:                                             ; preds = %body, %cond
  %result_phi = phi i64 [ 1, %cond ], [ 0, %body ]
  ret i64 %result_phi
}

define i64 @memcmp_ugt(ptr %0, ptr %1, i64 %2) {
entry:
  %index_alloca = alloca i64, align 8
  %len_alloca = alloca i64, align 8
  %right_ptr_alloca = alloca ptr, align 8
  %left_ptr_alloca = alloca ptr, align 8
  store ptr %0, ptr %left_ptr_alloca, align 8
  %left_ptr = load ptr, ptr %left_ptr_alloca, align 8
  store ptr %1, ptr %right_ptr_alloca, align 8
  %right_ptr = load ptr, ptr %right_ptr_alloca, align 8
  store i64 %2, ptr %len_alloca, align 4
  %len = load i64, ptr %len_alloca, align 4
  store i64 0, ptr %index_alloca, align 4
  br label %cond

cond:                                             ; preds = %body, %entry
  %index_value = load i64, ptr %index_alloca, align 4
  %loop_check = icmp ult i64 %index_value, %len
  br i1 %loop_check, label %body, label %done

body:                                             ; preds = %cond
  %left_elem_ptr = getelementptr i64, ptr %left_ptr, i64 %index_value
  %left_elem = load i64, ptr %left_elem_ptr, align 4
  %right_elem_ptr = getelementptr i64, ptr %right_ptr, i64 %index_value
  %right_elem = load i64, ptr %right_elem_ptr, align 4
  %compare = icmp ugt i64 %left_elem, %right_elem
  %next_index = add i64 %index_value, 1
  store i64 %next_index, ptr %index_alloca, align 4
  br i1 %compare, label %cond, label %done

done:                                             ; preds = %body, %cond
  %result_phi = phi i64 [ 1, %cond ], [ 0, %body ]
  ret i64 %result_phi
}

define i64 @memcmp_uge(ptr %0, ptr %1, i64 %2) {
entry:
  %index_alloca = alloca i64, align 8
  %len_alloca = alloca i64, align 8
  %right_ptr_alloca = alloca ptr, align 8
  %left_ptr_alloca = alloca ptr, align 8
  store ptr %0, ptr %left_ptr_alloca, align 8
  %left_ptr = load ptr, ptr %left_ptr_alloca, align 8
  store ptr %1, ptr %right_ptr_alloca, align 8
  %right_ptr = load ptr, ptr %right_ptr_alloca, align 8
  store i64 %2, ptr %len_alloca, align 4
  %len = load i64, ptr %len_alloca, align 4
  store i64 0, ptr %index_alloca, align 4
  br label %cond

cond:                                             ; preds = %body, %entry
  %index_value = load i64, ptr %index_alloca, align 4
  %loop_check = icmp ult i64 %index_value, %len
  br i1 %loop_check, label %body, label %done

body:                                             ; preds = %cond
  %left_elem_ptr = getelementptr i64, ptr %left_ptr, i64 %index_value
  %left_elem = load i64, ptr %left_elem_ptr, align 4
  %right_elem_ptr = getelementptr i64, ptr %right_ptr, i64 %index_value
  %right_elem = load i64, ptr %right_elem_ptr, align 4
  %compare = icmp uge i64 %left_elem, %right_elem
  %next_index = add i64 %index_value, 1
  store i64 %next_index, ptr %index_alloca, align 4
  br i1 %compare, label %cond, label %done

done:                                             ; preds = %body, %cond
  %result_phi = phi i64 [ 1, %cond ], [ 0, %body ]
  ret i64 %result_phi
}

define void @u32_div_mod(i64 %0, i64 %1, ptr %2, ptr %3) {
entry:
  %remainder_alloca = alloca ptr, align 8
  %quotient_alloca = alloca ptr, align 8
  %divisor_alloca = alloca i64, align 8
  %dividend_alloca = alloca i64, align 8
  store i64 %0, ptr %dividend_alloca, align 4
  %dividend = load i64, ptr %dividend_alloca, align 4
  store i64 %1, ptr %divisor_alloca, align 4
  %divisor = load i64, ptr %divisor_alloca, align 4
  store ptr %2, ptr %quotient_alloca, align 8
  %quotient = load ptr, ptr %quotient_alloca, align 8
  store ptr %3, ptr %remainder_alloca, align 8
  %remainder = load ptr, ptr %remainder_alloca, align 8
  %4 = call i64 @prophet_u32_mod(i64 %dividend, i64 %divisor)
  call void @builtin_range_check(i64 %4)
  %5 = add i64 %4, 1
  %6 = sub i64 %divisor, %5
  call void @builtin_range_check(i64 %6)
  %7 = call i64 @prophet_u32_div(i64 %dividend, i64 %divisor)
  call void @builtin_range_check(ptr %quotient)
  %8 = mul i64 %7, %divisor
  %9 = add i64 %8, %4
  %10 = icmp eq i64 %9, %dividend
  %11 = zext i1 %10 to i64
  call void @builtin_assert(i64 %11)
  store i64 %7, ptr %quotient, align 4
  store i64 %4, ptr %remainder, align 4
  ret void
}

define i64 @u32_power(i64 %0, i64 %1) {
entry:
  %exponent_alloca = alloca i64, align 8
  %base_alloca = alloca i64, align 8
  store i64 %0, ptr %base_alloca, align 4
  %base = load i64, ptr %base_alloca, align 4
  store i64 %1, ptr %exponent_alloca, align 4
  %exponent = load i64, ptr %exponent_alloca, align 4
  br label %loop

loop:                                             ; preds = %loop, %entry
  %2 = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %3 = phi i64 [ 1, %entry ], [ %multmp, %loop ]
  %inc = add i64 %2, 1
  %multmp = mul i64 %3, %base
  %loopcond = icmp ule i64 %inc, %exponent
  br i1 %loopcond, label %loop, label %exit

exit:                                             ; preds = %loop
  call void @builtin_range_check(i64 %3)
  ret i64 %3
}

define i64 @extendedAccountVersion(ptr %0) {
entry:
  %ACCOUNT_CODE_STORAGE_SYSTEM_CONTRACT = alloca ptr, align 8
  %_address = alloca ptr, align 8
  store ptr %0, ptr %_address, align 8
  %1 = load ptr, ptr %_address, align 8
  %2 = call ptr @heap_malloc(i64 4)
  store i64 0, ptr %2, align 4
  %3 = getelementptr i64, ptr %2, i64 1
  store i64 0, ptr %3, align 4
  %4 = getelementptr i64, ptr %2, i64 2
  store i64 0, ptr %4, align 4
  %5 = getelementptr i64, ptr %2, i64 3
  store i64 0, ptr %5, align 4
  %6 = call ptr @heap_malloc(i64 8)
  call void @memcpy(ptr %2, ptr %6, i64 4)
  %7 = getelementptr i64, ptr %6, i64 4
  call void @memcpy(ptr %1, ptr %7, i64 4)
  %8 = getelementptr i64, ptr %7, i64 4
  %9 = call ptr @heap_malloc(i64 4)
  call void @poseidon_hash(ptr %6, ptr %9, i64 8)
  %10 = call ptr @heap_malloc(i64 2)
  %11 = call ptr @heap_malloc(i64 4)
  call void @get_storage(ptr %9, ptr %11)
  %storage_value = load i64, ptr %11, align 4
  %slot_value = load i64, ptr %9, align 4
  %slot_offset = add i64 %slot_value, 1
  store i64 %slot_offset, ptr %9, align 4
  %supportedAAVersion = getelementptr inbounds { i64, i64 }, ptr %10, i32 0, i32 0
  store i64 %storage_value, ptr %supportedAAVersion, align 4
  %12 = call ptr @heap_malloc(i64 4)
  call void @get_storage(ptr %9, ptr %12)
  %storage_value1 = load i64, ptr %12, align 4
  %slot_value2 = load i64, ptr %9, align 4
  %slot_offset3 = add i64 %slot_value2, 1
  store i64 %slot_offset3, ptr %9, align 4
  %nonceOrdering = getelementptr inbounds { i64, i64 }, ptr %10, i32 0, i32 1
  store i64 %storage_value1, ptr %nonceOrdering, align 4
  %struct_member = getelementptr inbounds { i64, i64 }, ptr %10, i32 0, i32 0
  %13 = load i64, ptr %struct_member, align 4
  %14 = icmp ne i64 %13, 0
  br i1 %14, label %then, label %endif

then:                                             ; preds = %entry
  %struct_member4 = getelementptr inbounds { i64, i64 }, ptr %10, i32 0, i32 0
  %15 = load i64, ptr %struct_member4, align 4
  ret i64 %15

endif:                                            ; preds = %entry
  %16 = call ptr @heap_malloc(i64 4)
  %index_access = getelementptr i64, ptr %16, i64 0
  store i64 0, ptr %index_access, align 4
  %index_access5 = getelementptr i64, ptr %16, i64 1
  store i64 0, ptr %index_access5, align 4
  %index_access6 = getelementptr i64, ptr %16, i64 2
  store i64 0, ptr %index_access6, align 4
  %index_access7 = getelementptr i64, ptr %16, i64 3
  store i64 32770, ptr %index_access7, align 4
  store ptr %16, ptr %ACCOUNT_CODE_STORAGE_SYSTEM_CONTRACT, align 8
  %17 = load ptr, ptr %ACCOUNT_CODE_STORAGE_SYSTEM_CONTRACT, align 8
  %address_start = ptrtoint ptr %17 to i64
  call void @prophet_printf(i64 %address_start, i64 2)
  %18 = load ptr, ptr %_address, align 8
  %19 = call ptr @vector_new(i64 6)
  %20 = getelementptr i64, ptr %18, i64 0
  %21 = load i64, ptr %20, align 4
  %encode_value_ptr = getelementptr i64, ptr %19, i64 1
  store i64 %21, ptr %encode_value_ptr, align 4
  %22 = getelementptr i64, ptr %18, i64 1
  %23 = load i64, ptr %22, align 4
  %encode_value_ptr8 = getelementptr i64, ptr %19, i64 2
  store i64 %23, ptr %encode_value_ptr8, align 4
  %24 = getelementptr i64, ptr %18, i64 2
  %25 = load i64, ptr %24, align 4
  %encode_value_ptr9 = getelementptr i64, ptr %19, i64 3
  store i64 %25, ptr %encode_value_ptr9, align 4
  %26 = getelementptr i64, ptr %18, i64 3
  %27 = load i64, ptr %26, align 4
  %encode_value_ptr10 = getelementptr i64, ptr %19, i64 4
  store i64 %27, ptr %encode_value_ptr10, align 4
  %encode_value_ptr11 = getelementptr i64, ptr %19, i64 5
  store i64 4, ptr %encode_value_ptr11, align 4
  %encode_value_ptr12 = getelementptr i64, ptr %19, i64 6
  store i64 2179613704, ptr %encode_value_ptr12, align 4
  %fields_start = ptrtoint ptr %19 to i64
  call void @prophet_printf(i64 %fields_start, i64 0)
  ret i64 0
}

define ptr @create2(ptr %0, ptr %1, ptr %2) {
entry:
  %_input = alloca ptr, align 8
  %_bytecodeHash = alloca ptr, align 8
  %_salt = alloca ptr, align 8
  store ptr %0, ptr %_salt, align 8
  store ptr %1, ptr %_bytecodeHash, align 8
  store ptr %2, ptr %_input, align 8
  %3 = load ptr, ptr %_input, align 8
  %4 = load ptr, ptr %_salt, align 8
  %5 = load ptr, ptr %_bytecodeHash, align 8
  %6 = call ptr @create2Account(ptr %4, ptr %5, ptr %3, i64 0)
  ret ptr %6
}

define ptr @create2Account(ptr %0, ptr %1, ptr %2, i64 %3) {
entry:
  %newAddress = alloca ptr, align 8
  %_aaVersion = alloca i64, align 8
  %_input = alloca ptr, align 8
  %_bytecodeHash = alloca ptr, align 8
  %_salt = alloca ptr, align 8
  store ptr %0, ptr %_salt, align 8
  store ptr %1, ptr %_bytecodeHash, align 8
  store ptr %2, ptr %_input, align 8
  %4 = load ptr, ptr %_input, align 8
  store i64 %3, ptr %_aaVersion, align 4
  %5 = call ptr @heap_malloc(i64 12)
  call void @get_tape_data(ptr %5, i64 12)
  %6 = load ptr, ptr %_bytecodeHash, align 8
  %7 = load ptr, ptr %_salt, align 8
  %8 = call ptr @getNewAddressCreate2(ptr %5, ptr %6, ptr %7, ptr %4)
  store ptr %8, ptr %newAddress, align 8
  %9 = load ptr, ptr %newAddress, align 8
  %address_start = ptrtoint ptr %9 to i64
  call void @prophet_printf(i64 %address_start, i64 2)
  %10 = load ptr, ptr %_bytecodeHash, align 8
  %11 = load ptr, ptr %newAddress, align 8
  %12 = load i64, ptr %_aaVersion, align 4
  call void @_nonSystemDeployOnAddress(ptr %10, ptr %11, i64 %12, ptr %4)
  %13 = load ptr, ptr %newAddress, align 8
  ret ptr %13
}

define ptr @getNewAddressCreate2(ptr %0, ptr %1, ptr %2, ptr %3) {
entry:
  %_hash = alloca ptr, align 8
  %constructorInputHash = alloca ptr, align 8
  %CREATE2_PREFIX = alloca ptr, align 8
  %_input = alloca ptr, align 8
  %_salt = alloca ptr, align 8
  %_bytecodeHash = alloca ptr, align 8
  %_sender = alloca ptr, align 8
  store ptr %0, ptr %_sender, align 8
  store ptr %1, ptr %_bytecodeHash, align 8
  store ptr %2, ptr %_salt, align 8
  store ptr %3, ptr %_input, align 8
  %4 = load ptr, ptr %_input, align 8
  %5 = call ptr @vector_new(i64 100)
  %vector_data = getelementptr i64, ptr %5, i64 1
  %index_access = getelementptr i64, ptr %vector_data, i64 0
  store i64 79, ptr %index_access, align 4
  %index_access1 = getelementptr i64, ptr %vector_data, i64 1
  store i64 108, ptr %index_access1, align 4
  %index_access2 = getelementptr i64, ptr %vector_data, i64 2
  store i64 97, ptr %index_access2, align 4
  %index_access3 = getelementptr i64, ptr %vector_data, i64 3
  store i64 67, ptr %index_access3, align 4
  %index_access4 = getelementptr i64, ptr %vector_data, i64 4
  store i64 114, ptr %index_access4, align 4
  %index_access5 = getelementptr i64, ptr %vector_data, i64 5
  store i64 101, ptr %index_access5, align 4
  %index_access6 = getelementptr i64, ptr %vector_data, i64 6
  store i64 97, ptr %index_access6, align 4
  %index_access7 = getelementptr i64, ptr %vector_data, i64 7
  store i64 116, ptr %index_access7, align 4
  %index_access8 = getelementptr i64, ptr %vector_data, i64 8
  store i64 101, ptr %index_access8, align 4
  %index_access9 = getelementptr i64, ptr %vector_data, i64 9
  store i64 50, ptr %index_access9, align 4
  %vector_length = load i64, ptr %5, align 4
  %vector_data10 = getelementptr i64, ptr %5, i64 1
  %6 = call ptr @heap_malloc(i64 4)
  call void @poseidon_hash(ptr %vector_data10, ptr %6, i64 %vector_length)
  store ptr %6, ptr %CREATE2_PREFIX, align 8
  %vector_length11 = load i64, ptr %4, align 4
  %vector_data12 = getelementptr i64, ptr %4, i64 1
  %7 = call ptr @heap_malloc(i64 4)
  call void @poseidon_hash(ptr %vector_data12, ptr %7, i64 %vector_length11)
  store ptr %7, ptr %constructorInputHash, align 8
  %8 = load ptr, ptr %CREATE2_PREFIX, align 8
  %9 = load ptr, ptr %_sender, align 8
  %10 = load ptr, ptr %_salt, align 8
  %11 = load ptr, ptr %_bytecodeHash, align 8
  %12 = load ptr, ptr %constructorInputHash, align 8
  %13 = call ptr @vector_new(i64 20)
  %14 = getelementptr i64, ptr %8, i64 0
  %15 = load i64, ptr %14, align 4
  %encode_value_ptr = getelementptr i64, ptr %13, i64 1
  store i64 %15, ptr %encode_value_ptr, align 4
  %16 = getelementptr i64, ptr %8, i64 1
  %17 = load i64, ptr %16, align 4
  %encode_value_ptr13 = getelementptr i64, ptr %13, i64 2
  store i64 %17, ptr %encode_value_ptr13, align 4
  %18 = getelementptr i64, ptr %8, i64 2
  %19 = load i64, ptr %18, align 4
  %encode_value_ptr14 = getelementptr i64, ptr %13, i64 3
  store i64 %19, ptr %encode_value_ptr14, align 4
  %20 = getelementptr i64, ptr %8, i64 3
  %21 = load i64, ptr %20, align 4
  %encode_value_ptr15 = getelementptr i64, ptr %13, i64 4
  store i64 %21, ptr %encode_value_ptr15, align 4
  %22 = getelementptr i64, ptr %9, i64 0
  %23 = load i64, ptr %22, align 4
  %encode_value_ptr16 = getelementptr i64, ptr %13, i64 5
  store i64 %23, ptr %encode_value_ptr16, align 4
  %24 = getelementptr i64, ptr %9, i64 1
  %25 = load i64, ptr %24, align 4
  %encode_value_ptr17 = getelementptr i64, ptr %13, i64 6
  store i64 %25, ptr %encode_value_ptr17, align 4
  %26 = getelementptr i64, ptr %9, i64 2
  %27 = load i64, ptr %26, align 4
  %encode_value_ptr18 = getelementptr i64, ptr %13, i64 7
  store i64 %27, ptr %encode_value_ptr18, align 4
  %28 = getelementptr i64, ptr %9, i64 3
  %29 = load i64, ptr %28, align 4
  %encode_value_ptr19 = getelementptr i64, ptr %13, i64 8
  store i64 %29, ptr %encode_value_ptr19, align 4
  %30 = getelementptr i64, ptr %10, i64 0
  %31 = load i64, ptr %30, align 4
  %encode_value_ptr20 = getelementptr i64, ptr %13, i64 9
  store i64 %31, ptr %encode_value_ptr20, align 4
  %32 = getelementptr i64, ptr %10, i64 1
  %33 = load i64, ptr %32, align 4
  %encode_value_ptr21 = getelementptr i64, ptr %13, i64 10
  store i64 %33, ptr %encode_value_ptr21, align 4
  %34 = getelementptr i64, ptr %10, i64 2
  %35 = load i64, ptr %34, align 4
  %encode_value_ptr22 = getelementptr i64, ptr %13, i64 11
  store i64 %35, ptr %encode_value_ptr22, align 4
  %36 = getelementptr i64, ptr %10, i64 3
  %37 = load i64, ptr %36, align 4
  %encode_value_ptr23 = getelementptr i64, ptr %13, i64 12
  store i64 %37, ptr %encode_value_ptr23, align 4
  %38 = getelementptr i64, ptr %11, i64 0
  %39 = load i64, ptr %38, align 4
  %encode_value_ptr24 = getelementptr i64, ptr %13, i64 13
  store i64 %39, ptr %encode_value_ptr24, align 4
  %40 = getelementptr i64, ptr %11, i64 1
  %41 = load i64, ptr %40, align 4
  %encode_value_ptr25 = getelementptr i64, ptr %13, i64 14
  store i64 %41, ptr %encode_value_ptr25, align 4
  %42 = getelementptr i64, ptr %11, i64 2
  %43 = load i64, ptr %42, align 4
  %encode_value_ptr26 = getelementptr i64, ptr %13, i64 15
  store i64 %43, ptr %encode_value_ptr26, align 4
  %44 = getelementptr i64, ptr %11, i64 3
  %45 = load i64, ptr %44, align 4
  %encode_value_ptr27 = getelementptr i64, ptr %13, i64 16
  store i64 %45, ptr %encode_value_ptr27, align 4
  %46 = getelementptr i64, ptr %12, i64 0
  %47 = load i64, ptr %46, align 4
  %encode_value_ptr28 = getelementptr i64, ptr %13, i64 17
  store i64 %47, ptr %encode_value_ptr28, align 4
  %48 = getelementptr i64, ptr %12, i64 1
  %49 = load i64, ptr %48, align 4
  %encode_value_ptr29 = getelementptr i64, ptr %13, i64 18
  store i64 %49, ptr %encode_value_ptr29, align 4
  %50 = getelementptr i64, ptr %12, i64 2
  %51 = load i64, ptr %50, align 4
  %encode_value_ptr30 = getelementptr i64, ptr %13, i64 19
  store i64 %51, ptr %encode_value_ptr30, align 4
  %52 = getelementptr i64, ptr %12, i64 3
  %53 = load i64, ptr %52, align 4
  %encode_value_ptr31 = getelementptr i64, ptr %13, i64 20
  store i64 %53, ptr %encode_value_ptr31, align 4
  %vector_length32 = load i64, ptr %13, align 4
  %vector_data33 = getelementptr i64, ptr %13, i64 1
  %54 = call ptr @heap_malloc(i64 4)
  call void @poseidon_hash(ptr %vector_data33, ptr %54, i64 %vector_length32)
  store ptr %54, ptr %_hash, align 8
  %55 = load ptr, ptr %_hash, align 8
  ret ptr %55
}

define void @_nonSystemDeployOnAddress(ptr %0, ptr %1, i64 %2, ptr %3) {
entry:
  %NONCE_HOLDER_ADDRESS = alloca ptr, align 8
  %ACCOUNT_CODE_STORAGE_SYSTEM_CONTRACT = alloca ptr, align 8
  %MAX_SYSTEM_CONTRACT_ADDRESS = alloca ptr, align 8
  %_input = alloca ptr, align 8
  %_aaVersion = alloca i64, align 8
  %_newAddress = alloca ptr, align 8
  %_bytecodeHash = alloca ptr, align 8
  store ptr %0, ptr %_bytecodeHash, align 8
  store ptr %1, ptr %_newAddress, align 8
  store i64 %2, ptr %_aaVersion, align 4
  store ptr %3, ptr %_input, align 8
  %4 = load ptr, ptr %_input, align 8
  %5 = load ptr, ptr %_bytecodeHash, align 8
  %6 = call ptr @heap_malloc(i64 4)
  %index_access = getelementptr i64, ptr %6, i64 0
  store i64 0, ptr %index_access, align 4
  %index_access1 = getelementptr i64, ptr %6, i64 1
  store i64 0, ptr %index_access1, align 4
  %index_access2 = getelementptr i64, ptr %6, i64 2
  store i64 0, ptr %index_access2, align 4
  %index_access3 = getelementptr i64, ptr %6, i64 3
  store i64 0, ptr %index_access3, align 4
  %7 = call i64 @memcmp_eq(ptr %5, ptr %6, i64 4)
  %8 = icmp eq i64 %7, 0
  %9 = zext i1 %8 to i64
  call void @builtin_assert(i64 %9)
  %10 = call ptr @heap_malloc(i64 4)
  %index_access4 = getelementptr i64, ptr %10, i64 0
  store i64 0, ptr %index_access4, align 4
  %index_access5 = getelementptr i64, ptr %10, i64 1
  store i64 0, ptr %index_access5, align 4
  %index_access6 = getelementptr i64, ptr %10, i64 2
  store i64 0, ptr %index_access6, align 4
  %index_access7 = getelementptr i64, ptr %10, i64 3
  store i64 65535, ptr %index_access7, align 4
  store ptr %10, ptr %MAX_SYSTEM_CONTRACT_ADDRESS, align 8
  %11 = load ptr, ptr %_newAddress, align 8
  %12 = load ptr, ptr %MAX_SYSTEM_CONTRACT_ADDRESS, align 8
  %13 = call i64 @memcmp_ugt(ptr %11, ptr %12, i64 4)
  call void @builtin_assert(i64 %13)
  %14 = call ptr @heap_malloc(i64 4)
  %index_access8 = getelementptr i64, ptr %14, i64 0
  store i64 0, ptr %index_access8, align 4
  %index_access9 = getelementptr i64, ptr %14, i64 1
  store i64 0, ptr %index_access9, align 4
  %index_access10 = getelementptr i64, ptr %14, i64 2
  store i64 0, ptr %index_access10, align 4
  %index_access11 = getelementptr i64, ptr %14, i64 3
  store i64 32770, ptr %index_access11, align 4
  store ptr %14, ptr %ACCOUNT_CODE_STORAGE_SYSTEM_CONTRACT, align 8
  %15 = load ptr, ptr %ACCOUNT_CODE_STORAGE_SYSTEM_CONTRACT, align 8
  %address_start = ptrtoint ptr %15 to i64
  call void @prophet_printf(i64 %address_start, i64 2)
  %16 = load ptr, ptr %_newAddress, align 8
  %17 = call ptr @vector_new(i64 6)
  %18 = getelementptr i64, ptr %16, i64 0
  %19 = load i64, ptr %18, align 4
  %encode_value_ptr = getelementptr i64, ptr %17, i64 1
  store i64 %19, ptr %encode_value_ptr, align 4
  %20 = getelementptr i64, ptr %16, i64 1
  %21 = load i64, ptr %20, align 4
  %encode_value_ptr12 = getelementptr i64, ptr %17, i64 2
  store i64 %21, ptr %encode_value_ptr12, align 4
  %22 = getelementptr i64, ptr %16, i64 2
  %23 = load i64, ptr %22, align 4
  %encode_value_ptr13 = getelementptr i64, ptr %17, i64 3
  store i64 %23, ptr %encode_value_ptr13, align 4
  %24 = getelementptr i64, ptr %16, i64 3
  %25 = load i64, ptr %24, align 4
  %encode_value_ptr14 = getelementptr i64, ptr %17, i64 4
  store i64 %25, ptr %encode_value_ptr14, align 4
  %encode_value_ptr15 = getelementptr i64, ptr %17, i64 5
  store i64 4, ptr %encode_value_ptr15, align 4
  %encode_value_ptr16 = getelementptr i64, ptr %17, i64 6
  store i64 2179613704, ptr %encode_value_ptr16, align 4
  %fields_start = ptrtoint ptr %17 to i64
  call void @prophet_printf(i64 %fields_start, i64 0)
  %26 = call ptr @heap_malloc(i64 4)
  %index_access17 = getelementptr i64, ptr %26, i64 0
  store i64 0, ptr %index_access17, align 4
  %index_access18 = getelementptr i64, ptr %26, i64 1
  store i64 0, ptr %index_access18, align 4
  %index_access19 = getelementptr i64, ptr %26, i64 2
  store i64 0, ptr %index_access19, align 4
  %index_access20 = getelementptr i64, ptr %26, i64 3
  store i64 32771, ptr %index_access20, align 4
  store ptr %26, ptr %NONCE_HOLDER_ADDRESS, align 8
  %27 = load ptr, ptr %_newAddress, align 8
  %28 = call ptr @vector_new(i64 6)
  %29 = getelementptr i64, ptr %27, i64 0
  %30 = load i64, ptr %29, align 4
  %encode_value_ptr21 = getelementptr i64, ptr %28, i64 1
  store i64 %30, ptr %encode_value_ptr21, align 4
  %31 = getelementptr i64, ptr %27, i64 1
  %32 = load i64, ptr %31, align 4
  %encode_value_ptr22 = getelementptr i64, ptr %28, i64 2
  store i64 %32, ptr %encode_value_ptr22, align 4
  %33 = getelementptr i64, ptr %27, i64 2
  %34 = load i64, ptr %33, align 4
  %encode_value_ptr23 = getelementptr i64, ptr %28, i64 3
  store i64 %34, ptr %encode_value_ptr23, align 4
  %35 = getelementptr i64, ptr %27, i64 3
  %36 = load i64, ptr %35, align 4
  %encode_value_ptr24 = getelementptr i64, ptr %28, i64 4
  store i64 %36, ptr %encode_value_ptr24, align 4
  %encode_value_ptr25 = getelementptr i64, ptr %28, i64 5
  store i64 4, ptr %encode_value_ptr25, align 4
  %encode_value_ptr26 = getelementptr i64, ptr %28, i64 6
  store i64 3868785611, ptr %encode_value_ptr26, align 4
  %fields_start27 = ptrtoint ptr %28 to i64
  call void @prophet_printf(i64 %fields_start27, i64 0)
  %37 = load ptr, ptr %_bytecodeHash, align 8
  %38 = load ptr, ptr %_newAddress, align 8
  %39 = load i64, ptr %_aaVersion, align 4
  call void @_performDeployOnAddress(ptr %37, ptr %38, i64 %39, ptr %4)
  ret void
}

define void @_performDeployOnAddress(ptr %0, ptr %1, i64 %2, ptr %3) {
entry:
  %KNOWN_CODE_STORAGE_CONTRACT = alloca ptr, align 8
  %_input = alloca ptr, align 8
  %_aaVersion = alloca i64, align 8
  %_newAddress = alloca ptr, align 8
  %_bytecodeHash = alloca ptr, align 8
  store ptr %0, ptr %_bytecodeHash, align 8
  store ptr %1, ptr %_newAddress, align 8
  store i64 %2, ptr %_aaVersion, align 4
  store ptr %3, ptr %_input, align 8
  %4 = load ptr, ptr %_input, align 8
  %5 = call ptr @heap_malloc(i64 4)
  %index_access = getelementptr i64, ptr %5, i64 0
  store i64 0, ptr %index_access, align 4
  %index_access1 = getelementptr i64, ptr %5, i64 1
  store i64 0, ptr %index_access1, align 4
  %index_access2 = getelementptr i64, ptr %5, i64 2
  store i64 0, ptr %index_access2, align 4
  %index_access3 = getelementptr i64, ptr %5, i64 3
  store i64 32772, ptr %index_access3, align 4
  store ptr %5, ptr %KNOWN_CODE_STORAGE_CONTRACT, align 8
  %6 = load ptr, ptr %_bytecodeHash, align 8
  %7 = call ptr @vector_new(i64 6)
  %8 = getelementptr i64, ptr %6, i64 0
  %9 = load i64, ptr %8, align 4
  %encode_value_ptr = getelementptr i64, ptr %7, i64 1
  store i64 %9, ptr %encode_value_ptr, align 4
  %10 = getelementptr i64, ptr %6, i64 1
  %11 = load i64, ptr %10, align 4
  %encode_value_ptr4 = getelementptr i64, ptr %7, i64 2
  store i64 %11, ptr %encode_value_ptr4, align 4
  %12 = getelementptr i64, ptr %6, i64 2
  %13 = load i64, ptr %12, align 4
  %encode_value_ptr5 = getelementptr i64, ptr %7, i64 3
  store i64 %13, ptr %encode_value_ptr5, align 4
  %14 = getelementptr i64, ptr %6, i64 3
  %15 = load i64, ptr %14, align 4
  %encode_value_ptr6 = getelementptr i64, ptr %7, i64 4
  store i64 %15, ptr %encode_value_ptr6, align 4
  %encode_value_ptr7 = getelementptr i64, ptr %7, i64 5
  store i64 4, ptr %encode_value_ptr7, align 4
  %encode_value_ptr8 = getelementptr i64, ptr %7, i64 6
  store i64 4199620571, ptr %encode_value_ptr8, align 4
  %fields_start = ptrtoint ptr %7 to i64
  call void @prophet_printf(i64 %fields_start, i64 0)
  %16 = call ptr @heap_malloc(i64 2)
  %struct_member = getelementptr inbounds { i64, i64 }, ptr %16, i32 0, i32 0
  %17 = load i64, ptr %_aaVersion, align 4
  store i64 %17, ptr %struct_member, align 4
  %struct_member9 = getelementptr inbounds { i64, i64 }, ptr %16, i32 0, i32 1
  store i64 0, ptr %struct_member9, align 4
  %18 = load ptr, ptr %_newAddress, align 8
  %19 = call ptr @heap_malloc(i64 4)
  store i64 0, ptr %19, align 4
  %20 = getelementptr i64, ptr %19, i64 1
  store i64 0, ptr %20, align 4
  %21 = getelementptr i64, ptr %19, i64 2
  store i64 0, ptr %21, align 4
  %22 = getelementptr i64, ptr %19, i64 3
  store i64 0, ptr %22, align 4
  %23 = call ptr @heap_malloc(i64 8)
  call void @memcpy(ptr %19, ptr %23, i64 4)
  %24 = getelementptr i64, ptr %23, i64 4
  call void @memcpy(ptr %18, ptr %24, i64 4)
  %25 = getelementptr i64, ptr %24, i64 4
  %26 = call ptr @heap_malloc(i64 4)
  call void @poseidon_hash(ptr %23, ptr %26, i64 8)
  %supportedAAVersion = getelementptr inbounds { i64, i64 }, ptr %16, i32 0, i32 0
  %27 = load i64, ptr %supportedAAVersion, align 4
  %28 = call ptr @heap_malloc(i64 4)
  store i64 %27, ptr %28, align 4
  %29 = getelementptr i64, ptr %28, i64 1
  store i64 0, ptr %29, align 4
  %30 = getelementptr i64, ptr %28, i64 2
  store i64 0, ptr %30, align 4
  %31 = getelementptr i64, ptr %28, i64 3
  store i64 0, ptr %31, align 4
  call void @set_storage(ptr %26, ptr %28)
  %slot_value = load i64, ptr %26, align 4
  %slot_offset = add i64 %slot_value, 1
  store i64 %slot_offset, ptr %26, align 4
  %nonceOrdering = getelementptr inbounds { i64, i64 }, ptr %16, i32 0, i32 1
  %32 = load i64, ptr %nonceOrdering, align 4
  %33 = call ptr @heap_malloc(i64 4)
  store i64 %32, ptr %33, align 4
  %34 = getelementptr i64, ptr %33, i64 1
  store i64 0, ptr %34, align 4
  %35 = getelementptr i64, ptr %33, i64 2
  store i64 0, ptr %35, align 4
  %36 = getelementptr i64, ptr %33, i64 3
  store i64 0, ptr %36, align 4
  call void @set_storage(ptr %26, ptr %33)
  %37 = call ptr @heap_malloc(i64 12)
  call void @get_tape_data(ptr %37, i64 12)
  %38 = load ptr, ptr %_newAddress, align 8
  %39 = load ptr, ptr %_bytecodeHash, align 8
  call void @_constructContract(ptr %37, ptr %38, ptr %39, ptr %4, i64 0, i64 1)
  ret void
}

define void @_constructContract(ptr %0, ptr %1, ptr %2, ptr %3, i64 %4, i64 %5) {
entry:
  %ACCOUNT_CODE_STORAGE_SYSTEM_CONTRACT = alloca ptr, align 8
  %_callConstructor = alloca i64, align 8
  %_isSystem = alloca i64, align 8
  %_input = alloca ptr, align 8
  %_bytecodeHash = alloca ptr, align 8
  %_newAddress = alloca ptr, align 8
  %_sender = alloca ptr, align 8
  store ptr %0, ptr %_sender, align 8
  store ptr %1, ptr %_newAddress, align 8
  store ptr %2, ptr %_bytecodeHash, align 8
  store ptr %3, ptr %_input, align 8
  %6 = load ptr, ptr %_input, align 8
  store i64 %4, ptr %_isSystem, align 4
  store i64 %5, ptr %_callConstructor, align 4
  %7 = call ptr @heap_malloc(i64 4)
  %index_access = getelementptr i64, ptr %7, i64 0
  store i64 0, ptr %index_access, align 4
  %index_access1 = getelementptr i64, ptr %7, i64 1
  store i64 0, ptr %index_access1, align 4
  %index_access2 = getelementptr i64, ptr %7, i64 2
  store i64 0, ptr %index_access2, align 4
  %index_access3 = getelementptr i64, ptr %7, i64 3
  store i64 32770, ptr %index_access3, align 4
  store ptr %7, ptr %ACCOUNT_CODE_STORAGE_SYSTEM_CONTRACT, align 8
  %8 = load ptr, ptr %_sender, align 8
  %address_start = ptrtoint ptr %8 to i64
  call void @prophet_printf(i64 %address_start, i64 2)
  %9 = load ptr, ptr %_newAddress, align 8
  %address_start4 = ptrtoint ptr %9 to i64
  call void @prophet_printf(i64 %address_start4, i64 2)
  %10 = load ptr, ptr %_bytecodeHash, align 8
  %hash_start = ptrtoint ptr %10 to i64
  call void @prophet_printf(i64 %hash_start, i64 2)
  %fields_start = ptrtoint ptr %6 to i64
  call void @prophet_printf(i64 %fields_start, i64 0)
  %11 = load i64, ptr %_isSystem, align 4
  call void @prophet_printf(i64 %11, i64 3)
  %12 = load i64, ptr %_callConstructor, align 4
  call void @prophet_printf(i64 %12, i64 3)
  %13 = load i64, ptr %_callConstructor, align 4
  %14 = trunc i64 %13 to i1
  br i1 %14, label %then, label %else

then:                                             ; preds = %entry
  %15 = load ptr, ptr %_newAddress, align 8
  %16 = load ptr, ptr %_bytecodeHash, align 8
  %17 = call ptr @vector_new(i64 10)
  %18 = getelementptr i64, ptr %15, i64 0
  %19 = load i64, ptr %18, align 4
  %encode_value_ptr = getelementptr i64, ptr %17, i64 1
  store i64 %19, ptr %encode_value_ptr, align 4
  %20 = getelementptr i64, ptr %15, i64 1
  %21 = load i64, ptr %20, align 4
  %encode_value_ptr5 = getelementptr i64, ptr %17, i64 2
  store i64 %21, ptr %encode_value_ptr5, align 4
  %22 = getelementptr i64, ptr %15, i64 2
  %23 = load i64, ptr %22, align 4
  %encode_value_ptr6 = getelementptr i64, ptr %17, i64 3
  store i64 %23, ptr %encode_value_ptr6, align 4
  %24 = getelementptr i64, ptr %15, i64 3
  %25 = load i64, ptr %24, align 4
  %encode_value_ptr7 = getelementptr i64, ptr %17, i64 4
  store i64 %25, ptr %encode_value_ptr7, align 4
  %26 = getelementptr i64, ptr %16, i64 0
  %27 = load i64, ptr %26, align 4
  %encode_value_ptr8 = getelementptr i64, ptr %17, i64 5
  store i64 %27, ptr %encode_value_ptr8, align 4
  %28 = getelementptr i64, ptr %16, i64 1
  %29 = load i64, ptr %28, align 4
  %encode_value_ptr9 = getelementptr i64, ptr %17, i64 6
  store i64 %29, ptr %encode_value_ptr9, align 4
  %30 = getelementptr i64, ptr %16, i64 2
  %31 = load i64, ptr %30, align 4
  %encode_value_ptr10 = getelementptr i64, ptr %17, i64 7
  store i64 %31, ptr %encode_value_ptr10, align 4
  %32 = getelementptr i64, ptr %16, i64 3
  %33 = load i64, ptr %32, align 4
  %encode_value_ptr11 = getelementptr i64, ptr %17, i64 8
  store i64 %33, ptr %encode_value_ptr11, align 4
  %encode_value_ptr12 = getelementptr i64, ptr %17, i64 9
  store i64 8, ptr %encode_value_ptr12, align 4
  %encode_value_ptr13 = getelementptr i64, ptr %17, i64 10
  store i64 3592270258, ptr %encode_value_ptr13, align 4
  %fields_start14 = ptrtoint ptr %17 to i64
  call void @prophet_printf(i64 %fields_start14, i64 0)
  br label %endif

else:                                             ; preds = %entry
  %34 = load ptr, ptr %_newAddress, align 8
  %35 = load ptr, ptr %_bytecodeHash, align 8
  %36 = call ptr @vector_new(i64 10)
  %37 = getelementptr i64, ptr %34, i64 0
  %38 = load i64, ptr %37, align 4
  %encode_value_ptr15 = getelementptr i64, ptr %36, i64 1
  store i64 %38, ptr %encode_value_ptr15, align 4
  %39 = getelementptr i64, ptr %34, i64 1
  %40 = load i64, ptr %39, align 4
  %encode_value_ptr16 = getelementptr i64, ptr %36, i64 2
  store i64 %40, ptr %encode_value_ptr16, align 4
  %41 = getelementptr i64, ptr %34, i64 2
  %42 = load i64, ptr %41, align 4
  %encode_value_ptr17 = getelementptr i64, ptr %36, i64 3
  store i64 %42, ptr %encode_value_ptr17, align 4
  %43 = getelementptr i64, ptr %34, i64 3
  %44 = load i64, ptr %43, align 4
  %encode_value_ptr18 = getelementptr i64, ptr %36, i64 4
  store i64 %44, ptr %encode_value_ptr18, align 4
  %45 = getelementptr i64, ptr %35, i64 0
  %46 = load i64, ptr %45, align 4
  %encode_value_ptr19 = getelementptr i64, ptr %36, i64 5
  store i64 %46, ptr %encode_value_ptr19, align 4
  %47 = getelementptr i64, ptr %35, i64 1
  %48 = load i64, ptr %47, align 4
  %encode_value_ptr20 = getelementptr i64, ptr %36, i64 6
  store i64 %48, ptr %encode_value_ptr20, align 4
  %49 = getelementptr i64, ptr %35, i64 2
  %50 = load i64, ptr %49, align 4
  %encode_value_ptr21 = getelementptr i64, ptr %36, i64 7
  store i64 %50, ptr %encode_value_ptr21, align 4
  %51 = getelementptr i64, ptr %35, i64 3
  %52 = load i64, ptr %51, align 4
  %encode_value_ptr22 = getelementptr i64, ptr %36, i64 8
  store i64 %52, ptr %encode_value_ptr22, align 4
  %encode_value_ptr23 = getelementptr i64, ptr %36, i64 9
  store i64 8, ptr %encode_value_ptr23, align 4
  %encode_value_ptr24 = getelementptr i64, ptr %36, i64 10
  store i64 3592270258, ptr %encode_value_ptr24, align 4
  %fields_start25 = ptrtoint ptr %36 to i64
  call void @prophet_printf(i64 %fields_start25, i64 0)
  br label %endif

endif:                                            ; preds = %else, %then
  ret void
}

define void @function_dispatch(i64 %0, i64 %1, ptr %2) {
entry:
  %input_alloca = alloca ptr, align 8
  store ptr %2, ptr %input_alloca, align 8
  %input = load ptr, ptr %input_alloca, align 8
  switch i64 %0, label %missing_function [
    i64 3138377232, label %func_0_dispatch
    i64 3535773298, label %func_1_dispatch
    i64 2002090733, label %func_2_dispatch
    i64 321189754, label %func_3_dispatch
    i64 2349874997, label %func_4_dispatch
    i64 400494576, label %func_5_dispatch
    i64 4098969507, label %func_6_dispatch
  ]

missing_function:                                 ; preds = %entry
  unreachable

func_0_dispatch:                                  ; preds = %entry
  %3 = getelementptr ptr, ptr %input, i64 0
  %4 = call i64 @extendedAccountVersion(ptr %3)
  %5 = call ptr @heap_malloc(i64 2)
  %encode_value_ptr = getelementptr i64, ptr %5, i64 0
  store i64 %4, ptr %encode_value_ptr, align 4
  %encode_value_ptr1 = getelementptr i64, ptr %5, i64 1
  store i64 1, ptr %encode_value_ptr1, align 4
  call void @set_tape_data(ptr %5, i64 2)
  ret void

func_1_dispatch:                                  ; preds = %entry
  %6 = getelementptr ptr, ptr %input, i64 0
  %7 = getelementptr ptr, ptr %6, i64 4
  %8 = getelementptr ptr, ptr %7, i64 4
  %vector_length = load i64, ptr %8, align 4
  %9 = add i64 %vector_length, 1
  %10 = call ptr @create2(ptr %6, ptr %7, ptr %8)
  %11 = call ptr @heap_malloc(i64 5)
  %12 = getelementptr i64, ptr %10, i64 0
  %13 = load i64, ptr %12, align 4
  %encode_value_ptr2 = getelementptr i64, ptr %11, i64 0
  store i64 %13, ptr %encode_value_ptr2, align 4
  %14 = getelementptr i64, ptr %10, i64 1
  %15 = load i64, ptr %14, align 4
  %encode_value_ptr3 = getelementptr i64, ptr %11, i64 1
  store i64 %15, ptr %encode_value_ptr3, align 4
  %16 = getelementptr i64, ptr %10, i64 2
  %17 = load i64, ptr %16, align 4
  %encode_value_ptr4 = getelementptr i64, ptr %11, i64 2
  store i64 %17, ptr %encode_value_ptr4, align 4
  %18 = getelementptr i64, ptr %10, i64 3
  %19 = load i64, ptr %18, align 4
  %encode_value_ptr5 = getelementptr i64, ptr %11, i64 3
  store i64 %19, ptr %encode_value_ptr5, align 4
  %encode_value_ptr6 = getelementptr i64, ptr %11, i64 4
  store i64 4, ptr %encode_value_ptr6, align 4
  call void @set_tape_data(ptr %11, i64 5)
  ret void

func_2_dispatch:                                  ; preds = %entry
  %20 = getelementptr ptr, ptr %input, i64 0
  %21 = getelementptr ptr, ptr %20, i64 4
  %22 = getelementptr ptr, ptr %21, i64 4
  %vector_length7 = load i64, ptr %22, align 4
  %23 = add i64 %vector_length7, 1
  %24 = getelementptr ptr, ptr %22, i64 %23
  %25 = load i64, ptr %24, align 4
  %26 = call ptr @create2Account(ptr %20, ptr %21, ptr %22, i64 %25)
  %27 = call ptr @heap_malloc(i64 5)
  %28 = getelementptr i64, ptr %26, i64 0
  %29 = load i64, ptr %28, align 4
  %encode_value_ptr8 = getelementptr i64, ptr %27, i64 0
  store i64 %29, ptr %encode_value_ptr8, align 4
  %30 = getelementptr i64, ptr %26, i64 1
  %31 = load i64, ptr %30, align 4
  %encode_value_ptr9 = getelementptr i64, ptr %27, i64 1
  store i64 %31, ptr %encode_value_ptr9, align 4
  %32 = getelementptr i64, ptr %26, i64 2
  %33 = load i64, ptr %32, align 4
  %encode_value_ptr10 = getelementptr i64, ptr %27, i64 2
  store i64 %33, ptr %encode_value_ptr10, align 4
  %34 = getelementptr i64, ptr %26, i64 3
  %35 = load i64, ptr %34, align 4
  %encode_value_ptr11 = getelementptr i64, ptr %27, i64 3
  store i64 %35, ptr %encode_value_ptr11, align 4
  %encode_value_ptr12 = getelementptr i64, ptr %27, i64 4
  store i64 4, ptr %encode_value_ptr12, align 4
  call void @set_tape_data(ptr %27, i64 5)
  ret void

func_3_dispatch:                                  ; preds = %entry
  %36 = getelementptr ptr, ptr %input, i64 0
  %37 = getelementptr ptr, ptr %36, i64 4
  %38 = getelementptr ptr, ptr %37, i64 4
  %39 = getelementptr ptr, ptr %38, i64 4
  %vector_length13 = load i64, ptr %39, align 4
  %40 = add i64 %vector_length13, 1
  %41 = call ptr @getNewAddressCreate2(ptr %36, ptr %37, ptr %38, ptr %39)
  %42 = call ptr @heap_malloc(i64 5)
  %43 = getelementptr i64, ptr %41, i64 0
  %44 = load i64, ptr %43, align 4
  %encode_value_ptr14 = getelementptr i64, ptr %42, i64 0
  store i64 %44, ptr %encode_value_ptr14, align 4
  %45 = getelementptr i64, ptr %41, i64 1
  %46 = load i64, ptr %45, align 4
  %encode_value_ptr15 = getelementptr i64, ptr %42, i64 1
  store i64 %46, ptr %encode_value_ptr15, align 4
  %47 = getelementptr i64, ptr %41, i64 2
  %48 = load i64, ptr %47, align 4
  %encode_value_ptr16 = getelementptr i64, ptr %42, i64 2
  store i64 %48, ptr %encode_value_ptr16, align 4
  %49 = getelementptr i64, ptr %41, i64 3
  %50 = load i64, ptr %49, align 4
  %encode_value_ptr17 = getelementptr i64, ptr %42, i64 3
  store i64 %50, ptr %encode_value_ptr17, align 4
  %encode_value_ptr18 = getelementptr i64, ptr %42, i64 4
  store i64 4, ptr %encode_value_ptr18, align 4
  call void @set_tape_data(ptr %42, i64 5)
  ret void

func_4_dispatch:                                  ; preds = %entry
  %51 = getelementptr ptr, ptr %input, i64 0
  %52 = getelementptr ptr, ptr %51, i64 4
  %53 = getelementptr ptr, ptr %52, i64 4
  %54 = load i64, ptr %53, align 4
  %55 = getelementptr ptr, ptr %53, i64 1
  %vector_length19 = load i64, ptr %55, align 4
  %56 = add i64 %vector_length19, 1
  call void @_nonSystemDeployOnAddress(ptr %51, ptr %52, i64 %54, ptr %55)
  %57 = call ptr @heap_malloc(i64 1)
  store i64 0, ptr %57, align 4
  call void @set_tape_data(ptr %57, i64 1)
  ret void

func_5_dispatch:                                  ; preds = %entry
  %58 = getelementptr ptr, ptr %input, i64 0
  %59 = getelementptr ptr, ptr %58, i64 4
  %60 = getelementptr ptr, ptr %59, i64 4
  %61 = load i64, ptr %60, align 4
  %62 = getelementptr ptr, ptr %60, i64 1
  %vector_length20 = load i64, ptr %62, align 4
  %63 = add i64 %vector_length20, 1
  call void @_performDeployOnAddress(ptr %58, ptr %59, i64 %61, ptr %62)
  %64 = call ptr @heap_malloc(i64 1)
  store i64 0, ptr %64, align 4
  call void @set_tape_data(ptr %64, i64 1)
  ret void

func_6_dispatch:                                  ; preds = %entry
  %65 = getelementptr ptr, ptr %input, i64 0
  %66 = getelementptr ptr, ptr %65, i64 4
  %67 = getelementptr ptr, ptr %66, i64 4
  %68 = getelementptr ptr, ptr %67, i64 4
  %vector_length21 = load i64, ptr %68, align 4
  %69 = add i64 %vector_length21, 1
  %70 = getelementptr ptr, ptr %68, i64 %69
  %71 = load i64, ptr %70, align 4
  %72 = getelementptr ptr, ptr %70, i64 1
  %73 = load i64, ptr %72, align 4
  call void @_constructContract(ptr %65, ptr %66, ptr %67, ptr %68, i64 %71, i64 %73)
  %74 = call ptr @heap_malloc(i64 1)
  store i64 0, ptr %74, align 4
  call void @set_tape_data(ptr %74, i64 1)
  ret void
}

define void @main() {
entry:
  %0 = call ptr @heap_malloc(i64 13)
  call void @get_tape_data(ptr %0, i64 13)
  %function_selector = load i64, ptr %0, align 4
  %1 = call ptr @heap_malloc(i64 14)
  call void @get_tape_data(ptr %1, i64 14)
  %input_length = load i64, ptr %1, align 4
  %2 = add i64 %input_length, 14
  %3 = call ptr @heap_malloc(i64 %2)
  call void @get_tape_data(ptr %3, i64 %2)
  call void @function_dispatch(i64 %function_selector, i64 %input_length, ptr %3)
  ret void
}
