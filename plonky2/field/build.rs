extern crate cc;

fn main() {
    cc::Build::new().file("src/cfft/ntt/ntt.c").compile("libntt.a");
}