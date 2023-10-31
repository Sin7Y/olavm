use std::{io::{Write, self, BufRead}, fs::File, path::Path, time::Instant, fmt};

use plonky2_util::log2_strict;

use crate::{goldilocks_field::GoldilocksField, types::{Field, PrimeField64}, extension::{quadratic::QuadraticExtension, FieldExtension}};

type F = GoldilocksField;

#[test]
fn fft_in_place() {
    // degree 3
    let n = 4;
    // let mut p = F::rand_vec(n);
    let mut p = vec![
        F::from_canonical_u64(11),
        F::from_canonical_u64(22),
        F::from_canonical_u64(33),
        F::from_canonical_u64(44),
    ];
    let twiddles = super::get_twiddles::<F>(n);
    super::serial::fft_in_place(&mut p, &twiddles, 1, 1, 0);
    super::permute(&mut p);

    let mut values_file = std::fs::File::create("./tst.txt").unwrap();
    for point in p {
        values_file.write_all(point.0.to_string().as_bytes()).unwrap();
        writeln!(values_file).unwrap();
    }
}

#[test]
#[ignore]
fn evaluate_poly_2_20() {
    type F = GoldilocksField;
    type FE = QuadraticExtension<F>;
    let degree: usize = 1 << 20;
    let degree_padded = degree.next_power_of_two();

    // [1] sample points
    // Create a vector of coeffs; the first degree of them are
    // "random", the last degree_padded-degree of them are zero.
    // let mut points = (0..degree)
    //     .map(|i| F::from_canonical_usize(i * 2443 % 257))
    //     .chain(std::iter::repeat(F::ZERO).take(degree_padded - degree))
    //     .collect::<Vec<_>>();
    let mut points = (0..degree)
        .map(|i| {
            let base = F::from_canonical_usize(i * 2443 % 257);
            FE::from_basefield_array([base, base])
        })
        .chain(std::iter::repeat(FE::ZERO).take(degree_padded - degree))
        .collect::<Vec<_>>();

    // [2] read points from file
    // let mut points = vec![F::ZERO; degree_padded];
    // if let Ok(lines) = read_lines("./coeffs_20.txt") {
    //     for (idx, line) in lines.enumerate() {
    //         if let Ok(v) = line {
    //             let val = u64::from_str_radix(v.as_str(), 10);
    //             if let Ok(val) = val {
    //                 points[idx] = F::from_canonical_u64(val);
    //             }
    //         }
    //     }
    // }

    let coeffs = points.clone();
    let twiddles = super::get_twiddles::<FE>(degree_padded);

    let start = Instant::now();

    super::evaluate_poly(&mut points, &twiddles);

    println!("evaluate_poly cost time = {:?}", start.elapsed());

    let mut coeffs_file = std::fs::File::create("./coeffs_20.txt").unwrap();
    let mut values_file = std::fs::File::create("./values_20.txt").unwrap();
    let mut twiddles_file = std::fs::File::create("./twiddles_20.txt").unwrap();
    for (coeff, point) in coeffs.into_iter().zip(points) {
        // coeffs_file.write_all(coeff.0[0].to_string().as_bytes()).unwrap();
        coeffs_file.write_all(format!("{} {}", coeff.0[0].to_string(), coeff.0[1].to_string()).as_bytes()).unwrap();
        writeln!(coeffs_file).unwrap();
        // values_file.write_all(point.0.to_string().as_bytes()).unwrap();
        values_file.write_all(format!("{} {}", point.0[0].to_string(), point.0[1].to_string()).as_bytes()).unwrap();
        writeln!(values_file).unwrap();
    }
    for twiddle in twiddles {
        twiddles_file.write_all(format!("{} {}", twiddle.0[0].to_string(), twiddle.0[1].to_string()).as_bytes()).unwrap();
        writeln!(twiddles_file).unwrap();
    }
}

#[test]
#[ignore]
fn interpolate_poly_2_20() {
    type F = GoldilocksField;
    type FE = QuadraticExtension<F>;
    let degree: usize = 1 << 20;
    let degree_padded = degree.next_power_of_two();

    // [1] sample points
    // Create a vector of coeffs; the first degree of them are
    // "random", the last degree_padded-degree of them are zero.
    // let mut points = (0..degree)
    //     .map(|i| F::from_canonical_usize(i * 1337 % 100))
    //     .chain(std::iter::repeat(F::ZERO).take(degree_padded - degree))
    //     .collect::<Vec<_>>();

    // [2] read points from file
    let mut points = vec![FE::ZERO; degree_padded];
    if let Ok(lines) = read_lines("./values_20.txt") {
        for (idx, line) in lines.enumerate() {
            // if let Ok(v) = line {
            //     let val = u64::from_str_radix(v.as_str(), 10);
            //     if let Ok(val) = val {
            //         coeffs[idx] = F::from_canonical_u64(val);
            //     }
            // }
            if let Ok(v) = line {
                let arr: Vec<_> = v.split(' ').collect();
                let val0 = u64::from_str_radix(arr[0], 10);
                let val1 = u64::from_str_radix(arr[1], 10);
                // if let Ok(val0) = val0, let Ok(val1) = val1 {
                //     points[idx] = F::from_canonical_u64(val);
                // }
                match (val0, val1) {
                    (Ok(val0), Ok(val1)) => {
                        let val0 = F::from_canonical_u64(val0);
                        let val1 = F::from_canonical_u64(val1);
                        points[idx] = FE::from_basefield_array([val0, val1]);
                    }
                    (_, _) => {}
                }
            }
        }
    }

    let mut coeffs: Vec<_> = points.clone();
    let twiddles = super::get_inv_twiddles::<FE>(degree_padded);

    let start = Instant::now();

    super::interpolate_poly(&mut coeffs, &twiddles);

    println!("interpolate_poly cost time = {:?}", start.elapsed());

    let mut coeffs_file = std::fs::File::create("./ifft_coeffs_20.txt").unwrap();
    let mut values_file = std::fs::File::create("./ifft_values_20.txt").unwrap();
    // for (coeff, point) in coeffs.into_iter().zip(points) {
    //     coeffs_file.write_all(coeff.to_canonical_u64().to_string().as_bytes()).unwrap();
    //     writeln!(coeffs_file).unwrap();
    //     values_file.write_all(point.to_canonical_u64().to_string().as_bytes()).unwrap();
    //     writeln!(values_file).unwrap();
    // }
    for (coeff, point) in coeffs.into_iter().zip(points) {
        coeffs_file.write_all(format!("{} {}", coeff.0[0].to_string(), coeff.0[1].to_string()).as_bytes()).unwrap();
        writeln!(coeffs_file).unwrap();
        values_file.write_all(format!("{} {}", point.0[0].to_string(), point.0[1].to_string()).as_bytes()).unwrap();
        writeln!(values_file).unwrap();
    }
}

fn build_domain(size: usize) -> Vec<F> {
    let g = F::primitive_root_of_unity(log2_strict(size));
    g.powers().take(size).collect()
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename).unwrap();
    let lines = io::BufReader::new(file).lines();
    Ok(lines)
}