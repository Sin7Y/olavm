use std::{io::{Write, self, BufRead}, fs::File, path::Path};

use plonky2_util::log2_strict;

use crate::{goldilocks_field::GoldilocksField, types::Field};

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
fn fft_2_24() {
    type F = GoldilocksField;
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
    let mut points = vec![F::ZERO; degree_padded];
    if let Ok(lines) = read_lines("./coeffs_20.txt") {
        for (idx, line) in lines.enumerate() {
            if let Ok(v) = line {
                let val = u64::from_str_radix(v.as_str(), 10);
                if let Ok(val) = val {
                    points[idx] = F::from_canonical_u64(val);
                }
            }
        }
    }

    let coeffs = points.clone();
    let twiddles = super::get_twiddles::<F>(degree_padded);
    super::evaluate_poly(&mut points, &twiddles);

    let mut coeffs_file = std::fs::File::create("./coeffs_20_copy.txt").unwrap();
    let mut values_file = std::fs::File::create("./values_20_copy.txt").unwrap();
    for (coeff, point) in coeffs.into_iter().zip(points) {
        coeffs_file.write_all(coeff.0.to_string().as_bytes()).unwrap();
        writeln!(coeffs_file).unwrap();
        values_file.write_all(point.0.to_string().as_bytes()).unwrap();
        writeln!(values_file).unwrap();
    }
}

#[test]
fn ifft_2_24() {
    type F = GoldilocksField;
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
    let mut coeffs = vec![F::ZERO; degree_padded];
    if let Ok(lines) = read_lines("./values_20.txt") {
        for (idx, line) in lines.enumerate() {
            if let Ok(v) = line {
                let val = u64::from_str_radix(v.as_str(), 10);
                if let Ok(val) = val {
                    coeffs[idx] = F::from_canonical_u64(val);
                }
            }
        }
    }

    let points: Vec<GoldilocksField> = coeffs.clone();
    let twiddles = super::get_inv_twiddles::<F>(degree_padded);
    super::interpolate_poly(&mut coeffs, &twiddles);

    let mut coeffs_file = std::fs::File::create("./coeffs_20_copy_copy.txt").unwrap();
    let mut values_file = std::fs::File::create("./values_20_copy_copy.txt").unwrap();
    for (coeff, point) in coeffs.into_iter().zip(points) {
        coeffs_file.write_all(coeff.0.to_string().as_bytes()).unwrap();
        writeln!(coeffs_file).unwrap();
        values_file.write_all(point.0.to_string().as_bytes()).unwrap();
        writeln!(values_file).unwrap();
    }
}

fn build_domain(size: usize) -> Vec<F> {
    let g = F::primitive_root_of_unity(log2_strict(size));
    g.powers().take(size).collect()
}

fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where P: AsRef<Path>, {
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}