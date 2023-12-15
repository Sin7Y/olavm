; ModuleID = 'DefaultAccount'
source_filename = "DefaultAccount"

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

define void @onlyEntrypointCall() {
entry:
  %ENTRY_POINT_ADDRESS = alloca ptr, align 8
  %0 = call ptr @heap_malloc(i64 4)
  %index_access = getelementptr i64, ptr %0, i64 0
  store i64 0, ptr %index_access, align 4
  %index_access1 = getelementptr i64, ptr %0, i64 1
  store i64 0, ptr %index_access1, align 4
  %index_access2 = getelementptr i64, ptr %0, i64 2
  store i64 0, ptr %index_access2, align 4
  %index_access3 = getelementptr i64, ptr %0, i64 3
  store i64 32769, ptr %index_access3, align 4
  store ptr %0, ptr %ENTRY_POINT_ADDRESS, align 8
  %1 = call ptr @heap_malloc(i64 12)
  call void @get_tape_data(ptr %1, i64 12)
  %2 = load ptr, ptr %ENTRY_POINT_ADDRESS, align 8
  %3 = call i64 @memcmp_eq(ptr %1, ptr %2, i64 4)
  call void @builtin_assert(i64 %3)
  ret void
}

define void @ignoreDelegateCall() {
entry:
  %0 = call ptr @heap_malloc(i64 4)
  call void @get_tape_data(ptr %0, i64 4)
  %1 = call ptr @heap_malloc(i64 8)
  call void @get_tape_data(ptr %1, i64 8)
  %2 = call i64 @memcmp_eq(ptr %0, ptr %1, i64 4)
  call void @builtin_assert(i64 %2)
  ret void
}

define i64 @validateTransaction(ptr %0, ptr %1, ptr %2) {
entry:
  %magic = alloca ptr, align 8
  %_tx = alloca ptr, align 8
  %_signedHash = alloca ptr, align 8
  %_txHash = alloca ptr, align 8
  store ptr %0, ptr %_txHash, align 8
  store ptr %1, ptr %_signedHash, align 8
  store ptr %2, ptr %_tx, align 8
  %3 = load ptr, ptr %_tx, align 8
  %4 = load ptr, ptr %_txHash, align 8
  %hash_start = ptrtoint ptr %4 to i64
  call void @prophet_printf(i64 %hash_start, i64 2)
  %5 = load ptr, ptr %_signedHash, align 8
  %hash_start1 = ptrtoint ptr %5 to i64
  call void @prophet_printf(i64 %hash_start1, i64 2)
  %struct_member = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %3, i32 0, i32 0
  %address_start = ptrtoint ptr %struct_member to i64
  call void @prophet_printf(i64 %address_start, i64 2)
  %struct_member2 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %3, i32 0, i32 1
  %6 = load i64, ptr %struct_member2, align 4
  call void @prophet_printf(i64 %6, i64 3)
  %struct_member3 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %3, i32 0, i32 2
  %7 = load i64, ptr %struct_member3, align 4
  call void @prophet_printf(i64 %7, i64 3)
  %struct_member4 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %3, i32 0, i32 3
  %8 = load i64, ptr %struct_member4, align 4
  call void @prophet_printf(i64 %8, i64 3)
  %struct_member5 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %3, i32 0, i32 4
  %fields_start = ptrtoint ptr %struct_member5 to i64
  call void @prophet_printf(i64 %fields_start, i64 0)
  %struct_member6 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %3, i32 0, i32 5
  %fields_start7 = ptrtoint ptr %struct_member6 to i64
  call void @prophet_printf(i64 %fields_start7, i64 0)
  %struct_member8 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %3, i32 0, i32 6
  %fields_start9 = ptrtoint ptr %struct_member8 to i64
  call void @prophet_printf(i64 %fields_start9, i64 0)
  %struct_member10 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %3, i32 0, i32 7
  %hash_start11 = ptrtoint ptr %struct_member10 to i64
  call void @prophet_printf(i64 %hash_start11, i64 2)
  call void @onlyEntrypointCall()
  call void @ignoreDelegateCall()
  %9 = call ptr @vector_new(i64 1764)
  %vector_data = getelementptr i64, ptr %9, i64 1
  %index_access = getelementptr i64, ptr %vector_data, i64 0
  store i64 118, ptr %index_access, align 4
  %index_access12 = getelementptr i64, ptr %vector_data, i64 1
  store i64 97, ptr %index_access12, align 4
  %index_access13 = getelementptr i64, ptr %vector_data, i64 2
  store i64 108, ptr %index_access13, align 4
  %index_access14 = getelementptr i64, ptr %vector_data, i64 3
  store i64 105, ptr %index_access14, align 4
  %index_access15 = getelementptr i64, ptr %vector_data, i64 4
  store i64 100, ptr %index_access15, align 4
  %index_access16 = getelementptr i64, ptr %vector_data, i64 5
  store i64 97, ptr %index_access16, align 4
  %index_access17 = getelementptr i64, ptr %vector_data, i64 6
  store i64 116, ptr %index_access17, align 4
  %index_access18 = getelementptr i64, ptr %vector_data, i64 7
  store i64 101, ptr %index_access18, align 4
  %index_access19 = getelementptr i64, ptr %vector_data, i64 8
  store i64 84, ptr %index_access19, align 4
  %index_access20 = getelementptr i64, ptr %vector_data, i64 9
  store i64 114, ptr %index_access20, align 4
  %index_access21 = getelementptr i64, ptr %vector_data, i64 10
  store i64 97, ptr %index_access21, align 4
  %index_access22 = getelementptr i64, ptr %vector_data, i64 11
  store i64 110, ptr %index_access22, align 4
  %index_access23 = getelementptr i64, ptr %vector_data, i64 12
  store i64 115, ptr %index_access23, align 4
  %index_access24 = getelementptr i64, ptr %vector_data, i64 13
  store i64 97, ptr %index_access24, align 4
  %index_access25 = getelementptr i64, ptr %vector_data, i64 14
  store i64 99, ptr %index_access25, align 4
  %index_access26 = getelementptr i64, ptr %vector_data, i64 15
  store i64 116, ptr %index_access26, align 4
  %index_access27 = getelementptr i64, ptr %vector_data, i64 16
  store i64 105, ptr %index_access27, align 4
  %index_access28 = getelementptr i64, ptr %vector_data, i64 17
  store i64 111, ptr %index_access28, align 4
  %index_access29 = getelementptr i64, ptr %vector_data, i64 18
  store i64 110, ptr %index_access29, align 4
  %index_access30 = getelementptr i64, ptr %vector_data, i64 19
  store i64 40, ptr %index_access30, align 4
  %index_access31 = getelementptr i64, ptr %vector_data, i64 20
  store i64 104, ptr %index_access31, align 4
  %index_access32 = getelementptr i64, ptr %vector_data, i64 21
  store i64 97, ptr %index_access32, align 4
  %index_access33 = getelementptr i64, ptr %vector_data, i64 22
  store i64 115, ptr %index_access33, align 4
  %index_access34 = getelementptr i64, ptr %vector_data, i64 23
  store i64 104, ptr %index_access34, align 4
  %index_access35 = getelementptr i64, ptr %vector_data, i64 24
  store i64 44, ptr %index_access35, align 4
  %index_access36 = getelementptr i64, ptr %vector_data, i64 25
  store i64 104, ptr %index_access36, align 4
  %index_access37 = getelementptr i64, ptr %vector_data, i64 26
  store i64 97, ptr %index_access37, align 4
  %index_access38 = getelementptr i64, ptr %vector_data, i64 27
  store i64 115, ptr %index_access38, align 4
  %index_access39 = getelementptr i64, ptr %vector_data, i64 28
  store i64 104, ptr %index_access39, align 4
  %index_access40 = getelementptr i64, ptr %vector_data, i64 29
  store i64 44, ptr %index_access40, align 4
  %index_access41 = getelementptr i64, ptr %vector_data, i64 30
  store i64 84, ptr %index_access41, align 4
  %index_access42 = getelementptr i64, ptr %vector_data, i64 31
  store i64 114, ptr %index_access42, align 4
  %index_access43 = getelementptr i64, ptr %vector_data, i64 32
  store i64 97, ptr %index_access43, align 4
  %index_access44 = getelementptr i64, ptr %vector_data, i64 33
  store i64 110, ptr %index_access44, align 4
  %index_access45 = getelementptr i64, ptr %vector_data, i64 34
  store i64 115, ptr %index_access45, align 4
  %index_access46 = getelementptr i64, ptr %vector_data, i64 35
  store i64 97, ptr %index_access46, align 4
  %index_access47 = getelementptr i64, ptr %vector_data, i64 36
  store i64 99, ptr %index_access47, align 4
  %index_access48 = getelementptr i64, ptr %vector_data, i64 37
  store i64 116, ptr %index_access48, align 4
  %index_access49 = getelementptr i64, ptr %vector_data, i64 38
  store i64 105, ptr %index_access49, align 4
  %index_access50 = getelementptr i64, ptr %vector_data, i64 39
  store i64 111, ptr %index_access50, align 4
  %index_access51 = getelementptr i64, ptr %vector_data, i64 40
  store i64 110, ptr %index_access51, align 4
  %index_access52 = getelementptr i64, ptr %vector_data, i64 41
  store i64 41, ptr %index_access52, align 4
  %vector_length = load i64, ptr %9, align 4
  %vector_data53 = getelementptr i64, ptr %9, i64 1
  %10 = call ptr @heap_malloc(i64 4)
  call void @poseidon_hash(ptr %vector_data53, ptr %10, i64 %vector_length)
  store ptr %10, ptr %magic, align 8
  %11 = load ptr, ptr %magic, align 8
  %12 = call ptr @vector_new(i64 4)
  %13 = getelementptr i64, ptr %11, i64 0
  %14 = load i64, ptr %13, align 4
  %15 = getelementptr i64, ptr %12, i64 0
  store i64 %14, ptr %15, align 4
  %16 = getelementptr i64, ptr %11, i64 1
  %17 = load i64, ptr %16, align 4
  %18 = getelementptr i64, ptr %12, i64 1
  store i64 %17, ptr %18, align 4
  %19 = getelementptr i64, ptr %11, i64 2
  %20 = load i64, ptr %19, align 4
  %21 = getelementptr i64, ptr %12, i64 2
  store i64 %20, ptr %21, align 4
  %22 = getelementptr i64, ptr %11, i64 3
  %23 = load i64, ptr %22, align 4
  %24 = getelementptr i64, ptr %12, i64 3
  store i64 %23, ptr %24, align 4
  %vector_length54 = load i64, ptr %12, align 4
  %25 = sub i64 %vector_length54, 1
  %26 = sub i64 %25, 0
  call void @builtin_range_check(i64 %26)
  %vector_data55 = getelementptr i64, ptr %12, i64 1
  %index_access56 = getelementptr i64, ptr %vector_data55, i64 0
  %27 = load i64, ptr %index_access56, align 4
  ret i64 %27
}

define ptr @executeTransaction(ptr %0, ptr %1, ptr %2) {
entry:
  %to = alloca ptr, align 8
  %_tx = alloca ptr, align 8
  %_signedHash = alloca ptr, align 8
  %_txHash = alloca ptr, align 8
  store ptr %0, ptr %_txHash, align 8
  store ptr %1, ptr %_signedHash, align 8
  store ptr %2, ptr %_tx, align 8
  %3 = load ptr, ptr %_tx, align 8
  %4 = load ptr, ptr %_txHash, align 8
  %hash_start = ptrtoint ptr %4 to i64
  call void @prophet_printf(i64 %hash_start, i64 2)
  %5 = load ptr, ptr %_signedHash, align 8
  %hash_start1 = ptrtoint ptr %5 to i64
  call void @prophet_printf(i64 %hash_start1, i64 2)
  %struct_member = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %3, i32 0, i32 0
  %address_start = ptrtoint ptr %struct_member to i64
  call void @prophet_printf(i64 %address_start, i64 2)
  %struct_member2 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %3, i32 0, i32 1
  %6 = load i64, ptr %struct_member2, align 4
  call void @prophet_printf(i64 %6, i64 3)
  %struct_member3 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %3, i32 0, i32 2
  %7 = load i64, ptr %struct_member3, align 4
  call void @prophet_printf(i64 %7, i64 3)
  %struct_member4 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %3, i32 0, i32 3
  %8 = load i64, ptr %struct_member4, align 4
  call void @prophet_printf(i64 %8, i64 3)
  %struct_member5 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %3, i32 0, i32 4
  %fields_start = ptrtoint ptr %struct_member5 to i64
  call void @prophet_printf(i64 %fields_start, i64 0)
  %struct_member6 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %3, i32 0, i32 5
  %fields_start7 = ptrtoint ptr %struct_member6 to i64
  call void @prophet_printf(i64 %fields_start7, i64 0)
  %struct_member8 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %3, i32 0, i32 6
  %fields_start9 = ptrtoint ptr %struct_member8 to i64
  call void @prophet_printf(i64 %fields_start9, i64 0)
  %struct_member10 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %3, i32 0, i32 7
  %hash_start11 = ptrtoint ptr %struct_member10 to i64
  call void @prophet_printf(i64 %hash_start11, i64 2)
  call void @onlyEntrypointCall()
  call void @ignoreDelegateCall()
  %struct_member12 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %3, i32 0, i32 4
  %9 = load ptr, ptr %struct_member12, align 8
  %vector_length = load i64, ptr %9, align 4
  %array_len_sub_one = sub i64 %vector_length, 1
  %10 = sub i64 %array_len_sub_one, 0
  call void @builtin_range_check(i64 %10)
  %11 = sub i64 %vector_length, 4
  call void @builtin_range_check(i64 %11)
  call void @builtin_range_check(i64 4)
  %12 = call ptr @vector_new(i64 4)
  %vector_data = getelementptr i64, ptr %12, i64 1
  %vector_data13 = getelementptr i64, ptr %9, i64 1
  call void @memcpy(ptr %vector_data13, ptr %vector_data, i64 4)
  %vector_length14 = load i64, ptr %12, align 4
  %vector_data15 = getelementptr i64, ptr %12, i64 1
  %13 = getelementptr ptr, ptr %vector_data15, i64 0
  store ptr %13, ptr %to, align 8
  %14 = load ptr, ptr %to, align 8
  %address_start16 = ptrtoint ptr %14 to i64
  call void @prophet_printf(i64 %address_start16, i64 2)
  %struct_member17 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %3, i32 0, i32 4
  %15 = load ptr, ptr %struct_member17, align 8
  %vector_length18 = load i64, ptr %15, align 4
  %array_len_sub_one19 = sub i64 %vector_length18, 1
  %16 = sub i64 %array_len_sub_one19, 4
  call void @builtin_range_check(i64 %16)
  %17 = sub i64 %vector_length18, %vector_length18
  call void @builtin_range_check(i64 %17)
  %slice_len = sub i64 %vector_length18, 4
  call void @builtin_range_check(i64 %slice_len)
  %18 = call ptr @vector_new(i64 %slice_len)
  %vector_data20 = getelementptr i64, ptr %18, i64 1
  %vector_data21 = getelementptr i64, ptr %15, i64 1
  call void @memcpy(ptr %vector_data21, ptr %vector_data20, i64 %slice_len)
  ret ptr %18
}

define void @function_dispatch(i64 %0, i64 %1, ptr %2) {
entry:
  %input_alloca = alloca ptr, align 8
  store ptr %2, ptr %input_alloca, align 8
  %input = load ptr, ptr %input_alloca, align 8
  switch i64 %0, label %missing_function [
    i64 3726813225, label %func_0_dispatch
    i64 3602345202, label %func_1_dispatch
    i64 4150653994, label %func_2_dispatch
    i64 998595747, label %func_3_dispatch
  ]

missing_function:                                 ; preds = %entry
  unreachable

func_0_dispatch:                                  ; preds = %entry
  call void @onlyEntrypointCall()
  %3 = call ptr @heap_malloc(i64 1)
  store i64 0, ptr %3, align 4
  call void @set_tape_data(ptr %3, i64 1)
  ret void

func_1_dispatch:                                  ; preds = %entry
  call void @ignoreDelegateCall()
  %4 = call ptr @heap_malloc(i64 1)
  store i64 0, ptr %4, align 4
  call void @set_tape_data(ptr %4, i64 1)
  ret void

func_2_dispatch:                                  ; preds = %entry
  %5 = getelementptr ptr, ptr %input, i64 0
  %6 = getelementptr ptr, ptr %5, i64 4
  %7 = getelementptr ptr, ptr %6, i64 4
  %decode_struct_field = getelementptr { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %7, i64 0
  %decode_struct_field1 = getelementptr { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %7, i64 4
  %8 = load i64, ptr %decode_struct_field1, align 4
  %decode_struct_field2 = getelementptr { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %7, i64 5
  %9 = load i64, ptr %decode_struct_field2, align 4
  %decode_struct_field3 = getelementptr { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %7, i64 6
  %10 = load i64, ptr %decode_struct_field3, align 4
  %decode_struct_field4 = getelementptr { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %7, i64 7
  %vector_length = load i64, ptr %decode_struct_field4, align 4
  %11 = add i64 %vector_length, 1
  %decode_struct_offset = add i64 7, %11
  %decode_struct_field5 = getelementptr { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %7, i64 %decode_struct_offset
  %vector_length6 = load i64, ptr %decode_struct_field5, align 4
  %12 = add i64 %vector_length6, 1
  %decode_struct_offset7 = add i64 %decode_struct_offset, %12
  %decode_struct_field8 = getelementptr { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %7, i64 %decode_struct_offset7
  %vector_length9 = load i64, ptr %decode_struct_field8, align 4
  %13 = add i64 %vector_length9, 1
  %decode_struct_offset10 = add i64 %decode_struct_offset7, %13
  %decode_struct_field11 = getelementptr { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %7, i64 %decode_struct_offset10
  %decode_struct_offset12 = add i64 %decode_struct_offset10, 4
  %14 = call ptr @heap_malloc(i64 14)
  %struct_member = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %14, i32 0, i32 0
  store ptr %decode_struct_field, ptr %struct_member, align 8
  %struct_member13 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %14, i32 0, i32 1
  store i64 %8, ptr %struct_member13, align 4
  %struct_member14 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %14, i32 0, i32 2
  store i64 %9, ptr %struct_member14, align 4
  %struct_member15 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %14, i32 0, i32 3
  store i64 %10, ptr %struct_member15, align 4
  %struct_member16 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %14, i32 0, i32 4
  store ptr %decode_struct_field4, ptr %struct_member16, align 8
  %struct_member17 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %14, i32 0, i32 5
  store ptr %decode_struct_field5, ptr %struct_member17, align 8
  %struct_member18 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %14, i32 0, i32 6
  store ptr %decode_struct_field8, ptr %struct_member18, align 8
  %struct_member19 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %14, i32 0, i32 7
  store ptr %decode_struct_field11, ptr %struct_member19, align 8
  %15 = call i64 @validateTransaction(ptr %5, ptr %6, ptr %14)
  %16 = call ptr @heap_malloc(i64 2)
  %encode_value_ptr = getelementptr i64, ptr %16, i64 0
  store i64 %15, ptr %encode_value_ptr, align 4
  %encode_value_ptr20 = getelementptr i64, ptr %16, i64 1
  store i64 1, ptr %encode_value_ptr20, align 4
  call void @set_tape_data(ptr %16, i64 2)
  ret void

func_3_dispatch:                                  ; preds = %entry
  %17 = getelementptr ptr, ptr %input, i64 0
  %18 = getelementptr ptr, ptr %17, i64 4
  %19 = getelementptr ptr, ptr %18, i64 4
  %decode_struct_field21 = getelementptr { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %19, i64 0
  %decode_struct_field22 = getelementptr { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %19, i64 4
  %20 = load i64, ptr %decode_struct_field22, align 4
  %decode_struct_field23 = getelementptr { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %19, i64 5
  %21 = load i64, ptr %decode_struct_field23, align 4
  %decode_struct_field24 = getelementptr { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %19, i64 6
  %22 = load i64, ptr %decode_struct_field24, align 4
  %decode_struct_field25 = getelementptr { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %19, i64 7
  %vector_length26 = load i64, ptr %decode_struct_field25, align 4
  %23 = add i64 %vector_length26, 1
  %decode_struct_offset27 = add i64 7, %23
  %decode_struct_field28 = getelementptr { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %19, i64 %decode_struct_offset27
  %vector_length29 = load i64, ptr %decode_struct_field28, align 4
  %24 = add i64 %vector_length29, 1
  %decode_struct_offset30 = add i64 %decode_struct_offset27, %24
  %decode_struct_field31 = getelementptr { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %19, i64 %decode_struct_offset30
  %vector_length32 = load i64, ptr %decode_struct_field31, align 4
  %25 = add i64 %vector_length32, 1
  %decode_struct_offset33 = add i64 %decode_struct_offset30, %25
  %decode_struct_field34 = getelementptr { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %19, i64 %decode_struct_offset33
  %decode_struct_offset35 = add i64 %decode_struct_offset33, 4
  %26 = call ptr @heap_malloc(i64 14)
  %struct_member36 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %26, i32 0, i32 0
  store ptr %decode_struct_field21, ptr %struct_member36, align 8
  %struct_member37 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %26, i32 0, i32 1
  store i64 %20, ptr %struct_member37, align 4
  %struct_member38 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %26, i32 0, i32 2
  store i64 %21, ptr %struct_member38, align 4
  %struct_member39 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %26, i32 0, i32 3
  store i64 %22, ptr %struct_member39, align 4
  %struct_member40 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %26, i32 0, i32 4
  store ptr %decode_struct_field25, ptr %struct_member40, align 8
  %struct_member41 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %26, i32 0, i32 5
  store ptr %decode_struct_field28, ptr %struct_member41, align 8
  %struct_member42 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %26, i32 0, i32 6
  store ptr %decode_struct_field31, ptr %struct_member42, align 8
  %struct_member43 = getelementptr inbounds { ptr, i64, i64, i64, ptr, ptr, ptr, ptr }, ptr %26, i32 0, i32 7
  store ptr %decode_struct_field34, ptr %struct_member43, align 8
  %27 = call ptr @executeTransaction(ptr %17, ptr %18, ptr %26)
  %vector_length44 = load i64, ptr %27, align 4
  %28 = add i64 %vector_length44, 1
  %heap_size = add i64 %28, 1
  %29 = call ptr @heap_malloc(i64 %heap_size)
  %vector_length45 = load i64, ptr %27, align 4
  %vector_data = getelementptr i64, ptr %27, i64 1
  %30 = add i64 %vector_length45, 1
  call void @memcpy(ptr %vector_data, ptr %29, i64 %30)
  %31 = add i64 %30, 0
  %encode_value_ptr46 = getelementptr i64, ptr %29, i64 %31
  store i64 %28, ptr %encode_value_ptr46, align 4
  call void @set_tape_data(ptr %29, i64 %heap_size)
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
