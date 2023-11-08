use enum_iterator::Sequence;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Sequence)]
pub enum OlaRegister {
    R0,
    R1,
    R2,
    R3,
    R4,
    R5,
    R6,
    R7,
    R8,
    R9,
}

impl OlaRegister {
    pub fn index(&self) -> u8 {
        match self {
            OlaRegister::R9 => 9,
            OlaRegister::R8 => 8,
            OlaRegister::R7 => 7,
            OlaRegister::R6 => 6,
            OlaRegister::R5 => 5,
            OlaRegister::R4 => 4,
            OlaRegister::R3 => 3,
            OlaRegister::R2 => 2,
            OlaRegister::R1 => 1,
            OlaRegister::R0 => 0,
        }
    }

    fn binary_bit_shift_as_op0(&self) -> u8 {
        match self {
            OlaRegister::R9 => 61,
            OlaRegister::R8 => 60,
            OlaRegister::R7 => 59,
            OlaRegister::R6 => 58,
            OlaRegister::R5 => 57,
            OlaRegister::R4 => 56,
            OlaRegister::R3 => 55,
            OlaRegister::R2 => 54,
            OlaRegister::R1 => 53,
            OlaRegister::R0 => 52,
        }
    }

    fn binary_bit_shift_as_op1(&self) -> u8 {
        match self {
            OlaRegister::R9 => 51,
            OlaRegister::R8 => 50,
            OlaRegister::R7 => 49,
            OlaRegister::R6 => 48,
            OlaRegister::R5 => 47,
            OlaRegister::R4 => 46,
            OlaRegister::R3 => 45,
            OlaRegister::R2 => 44,
            OlaRegister::R1 => 43,
            OlaRegister::R0 => 42,
        }
    }

    fn binary_bit_shift_as_dst(&self) -> u8 {
        match self {
            OlaRegister::R9 => 41,
            OlaRegister::R8 => 40,
            OlaRegister::R7 => 39,
            OlaRegister::R6 => 38,
            OlaRegister::R5 => 37,
            OlaRegister::R4 => 36,
            OlaRegister::R3 => 35,
            OlaRegister::R2 => 34,
            OlaRegister::R1 => 33,
            OlaRegister::R0 => 32,
        }
    }

    pub fn binary_bit_mask_as_op0(&self) -> u64 {
        1 << self.binary_bit_shift_as_op0()
    }

    pub fn binary_bit_mask_as_op1(&self) -> u64 {
        1 << self.binary_bit_shift_as_op1()
    }

    pub fn binary_bit_mask_as_dst(&self) -> u64 {
        1 << self.binary_bit_shift_as_dst()
    }
}

impl FromStr for OlaRegister {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "r0" => Ok(OlaRegister::R0),
            "r1" => Ok(OlaRegister::R1),
            "r2" => Ok(OlaRegister::R2),
            "r3" => Ok(OlaRegister::R3),
            "r4" => Ok(OlaRegister::R4),
            "r5" => Ok(OlaRegister::R5),
            "r6" => Ok(OlaRegister::R6),
            "r7" => Ok(OlaRegister::R7),
            "r8" => Ok(OlaRegister::R8),
            "r9" => Ok(OlaRegister::R9),
            _ => Err(format!("invalid reg identifier: {}", s)),
        }
    }
}

impl Display for OlaRegister {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let token = match self {
            OlaRegister::R0 => "r0".to_string(),
            OlaRegister::R1 => "r1".to_string(),
            OlaRegister::R2 => "r2".to_string(),
            OlaRegister::R3 => "r3".to_string(),
            OlaRegister::R4 => "r4".to_string(),
            OlaRegister::R5 => "r5".to_string(),
            OlaRegister::R6 => "r6".to_string(),
            OlaRegister::R7 => "r7".to_string(),
            OlaRegister::R8 => "r8".to_string(),
            OlaRegister::R9 => "r9".to_string(),
        };
        write!(f, "{}", token)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum OlaSpecialRegister {
    PC,
    PSP,
}

impl Display for OlaSpecialRegister {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let token = match self {
            OlaSpecialRegister::PC => "pc",
            OlaSpecialRegister::PSP => "psp",
        };
        write!(f, "{}", token)
    }
}

impl FromStr for OlaSpecialRegister {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pc" => Ok(OlaSpecialRegister::PC),
            "psp" => Ok(OlaSpecialRegister::PSP),
            _ => Err(format!("invalid special reg identifier: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::vm::hardware::{OlaRegister, OlaSpecialRegister};
    use std::str::FromStr;

    #[test]
    fn test_hardware_parse() {
        let r8 = OlaRegister::from_str("r7").unwrap();
        assert_eq!(r8, OlaRegister::R7);

        let psp = OlaSpecialRegister::from_str("psp").unwrap();
        assert_eq!(psp, OlaSpecialRegister::PSP);
    }
}
