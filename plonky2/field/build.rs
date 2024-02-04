#[cfg(not(feature = "cuda"))]
fn main() {}

#[cfg(feature = "cuda")]
fn main() {
    let profile = std::env::var("PROFILE").unwrap();
    let profile = match profile.as_str() {
        "debug" => "Debug",
        "release" => "Release",
        _ => "Release",
    };
    // let mut dst = Config::new("cuda")
    //     .define("CMAKE_BUILD_TYPE", profile)
    //     .build();
    // dst.push("lib");
    // println!("cargo:rustc-link-search=native={}", dst.display());
    let ex_cuda_lib_path: &str = "plonky2/field/cuda/build";
    println!("cargo:rustc-link-search=native={}", ex_cuda_lib_path);
    println!("cargo:rustc-link-lib=static=cuda_lib");

    let default_cuda_lib_path: &str = "/usr/local/cuda/lib64";
    println!("cargo:rustc-link-search=native={}", default_cuda_lib_path);
    //println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

    println!("cargo:rustc-link-lib=stdc++");
}
