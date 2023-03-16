use std::fmt::{Display, Formatter};
use std::str::FromStr;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
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
}

impl OlaRegister {
    fn binary_bit_shift_as_op0(&self) -> u8 {
        match self {
            OlaRegister::R8 => 61,
            OlaRegister::R7 => 60,
            OlaRegister::R6 => 59,
            OlaRegister::R5 => 58,
            OlaRegister::R4 => 57,
            OlaRegister::R3 => 56,
            OlaRegister::R2 => 55,
            OlaRegister::R1 => 54,
            OlaRegister::R0 => 53,
        }
    }

    fn binary_bit_shift_as_op1(&self) -> u8 {
        match self {
            OlaRegister::R8 => 52,
            OlaRegister::R7 => 51,
            OlaRegister::R6 => 50,
            OlaRegister::R5 => 49,
            OlaRegister::R4 => 48,
            OlaRegister::R3 => 47,
            OlaRegister::R2 => 46,
            OlaRegister::R1 => 45,
            OlaRegister::R0 => 44,
        }
    }

    fn binary_bit_shift_as_dst(&self) -> u8 {
        match self {
            OlaRegister::R8 => 43,
            OlaRegister::R7 => 42,
            OlaRegister::R6 => 41,
            OlaRegister::R5 => 40,
            OlaRegister::R4 => 39,
            OlaRegister::R3 => 38,
            OlaRegister::R2 => 37,
            OlaRegister::R1 => 36,
            OlaRegister::R0 => 35,
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
    use crate::hardware::{OlaRegister, OlaSpecialRegister};
    use std::str::FromStr;

    #[test]
    fn test_hardware_parse() {
        let r8 = OlaRegister::from_str("r7").unwrap();
        assert_eq!(r8, OlaRegister::R7);

        let psp = OlaSpecialRegister::from_str("psp").unwrap();
        assert_eq!(psp, OlaSpecialRegister::PSP);
    }
}
