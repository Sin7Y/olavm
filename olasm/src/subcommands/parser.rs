use anyhow::{bail, Ok, Result};
use ola_lang_abi::{FixedArray4, FixedArray8, Param, Type, Value};

use crate::utils::{
    h256_from_hex_be, h256_to_u32_array, h256_to_u64_array, u32_array_to_h256, u64_array_to_h256,
    OLA_FIELD_ORDER,
};

pub struct ToValue;
impl ToValue {
    pub fn parse_input(param: Param, input: String) -> Value {
        let parse_result = match param.type_ {
            ola_lang_abi::Type::U32 => Self::parse_u32(input),
            ola_lang_abi::Type::Field => Self::parse_field(input),
            ola_lang_abi::Type::Hash => Self::parse_hash(input),
            ola_lang_abi::Type::Address => Self::parse_address(input),
            ola_lang_abi::Type::Bool => Self::parse_bool(input),
            ola_lang_abi::Type::FixedArray(t, size) => Self::parse_fixed_array(*t, size, input),
            ola_lang_abi::Type::String => Self::parse_string(input),
            ola_lang_abi::Type::Fields => Self::parse_fields(input),
            ola_lang_abi::Type::Array(t) => Self::parse_array(*t, input),
            ola_lang_abi::Type::Tuple(attrs) => Self::parse_tuple(attrs, input),
            ola_lang_abi::Type::U256 => Self::parse_u256(input),
        };
        parse_result.unwrap()
    }

    fn parse_u32(input: String) -> Result<Value> {
        let value = input.parse::<u32>().expect("invalid u32 input");
        Ok(Value::U32(value as u64))
    }

    fn parse_field(input: String) -> Result<Value> {
        let value = input.parse::<u64>().expect("invalid field element input");
        if value > OLA_FIELD_ORDER {
            bail!("invalid field element input")
        }
        Ok(Value::Field(value))
    }

    fn parse_hash(input: String) -> Result<Value> {
        let hash = h256_from_hex_be(input.as_str()).expect("invalid contract address");
        let u256 = h256_to_u64_array(&hash);
        Ok(Value::Hash(FixedArray4(u256)))
    }

    fn parse_address(input: String) -> Result<Value> {
        Self::parse_hash(input)
    }

    fn parse_bool(input: String) -> Result<Value> {
        let value = input.parse::<bool>().expect("invalid bool input");
        Ok(Value::Bool(value))
    }

    fn parse_fixed_array(t: Type, size: u64, input: String) -> Result<Value> {
        match t {
            Type::U32
            | Type::Field
            | Type::Hash
            | Type::Address
            | Type::Bool
            | Type::String
            | Type::Fields => {
                let s = input.as_str();
                if !s.starts_with('[') || !s.ends_with(']') {
                    bail!("invalid fixed array format.")
                }
                let content = &s[1..s.len() - 1];
                let split_content: Vec<String> =
                    content.split(',').map(|s| s.to_string()).collect();
                if split_content.len() as u64 != size {
                    bail!("invalid fixed array size")
                }
                let items: Vec<Value> = split_content
                    .iter()
                    .map(|i| {
                        Self::parse_input(
                            Param {
                                name: "tmp".to_string(),
                                type_: t.clone(),
                                indexed: None,
                            },
                            i.clone(),
                        )
                    })
                    .collect();
                Ok(Value::FixedArray(items, t))
            }
            Type::FixedArray(_, _) | Type::Array(_) | Type::Tuple(_) | Type::U256 => {
                bail!("Composite types in FixedArray has not been supported for cli tools.")
            }
        }
    }

    fn parse_string(input: String) -> Result<Value> {
        Ok(Value::String(input))
    }

    fn parse_fields(input: String) -> Result<Value> {
        let s = input.as_str();
        if !s.starts_with('[') || !s.ends_with(']') {
            bail!("invalid fixed array format.")
        }
        let content = &s[1..s.len() - 1];
        let split_content: Vec<String> = content.split(',').map(|s| s.to_string()).collect();
        let items: Vec<u64> = split_content
            .iter()
            .map(|i| {
                let value = i.parse::<u64>().expect("invalid field element input");
                if value > OLA_FIELD_ORDER {
                    panic!("invalid field element input")
                }
                value
            })
            .collect();
        Ok(Value::Fields(items))
    }

    fn parse_array(t: Type, input: String) -> Result<Value> {
        match t {
            Type::U32
            | Type::Field
            | Type::Hash
            | Type::Address
            | Type::Bool
            | Type::String
            | Type::Fields => {
                let s = input.as_str();
                if !s.starts_with('[') || !s.ends_with(']') {
                    bail!("invalid array format.")
                }
                let content = &s[1..s.len() - 1];
                let split_content: Vec<String> =
                    content.split(',').map(|s| s.to_string()).collect();
                let items: Vec<Value> = split_content
                    .iter()
                    .map(|i| {
                        Self::parse_input(
                            Param {
                                name: "tmp".to_string(),
                                type_: t.clone(),
                                indexed: None,
                            },
                            i.clone(),
                        )
                    })
                    .collect();
                Ok(Value::Array(items, t))
            }
            Type::FixedArray(_, _) | Type::Array(_) | Type::Tuple(_) | Type::U256 => {
                bail!("Composite types in Array has not been supported for cli tools.")
            }
        }
    }

    fn parse_tuple(attrs: Vec<(String, Type)>, input: String) -> Result<Value> {
        let s = input.as_str();
        if !s.starts_with('{') || !s.ends_with('}') {
            bail!("invalid tuple format.")
        }
        let content = &s[1..s.len() - 1];
        let split_content: Vec<String> = content.split(',').map(|s| s.to_string()).collect();
        if split_content.len() != attrs.len() {
            bail!("invalid tuple size")
        }
        let items: Vec<(String, Value)> = split_content
            .iter()
            .zip(attrs.iter())
            .map(|(i, (name, t))| {
                match t {
                    Type::U32
                    | Type::Field
                    | Type::Hash
                    | Type::Address
                    | Type::Bool
                    | Type::String
                    | Type::Fields => {}
                    Type::FixedArray(_, _) | Type::Array(_) | Type::Tuple(_) | Type::U256 => {
                        panic!("Composite types in Tuple has not been supported for cli tools.")
                    }
                }
                let v = Self::parse_input(
                    Param {
                        name: name.clone(),
                        type_: t.clone(),
                        indexed: None,
                    },
                    i.clone(),
                );
                (name.clone(), v)
            })
            .collect();
        Ok(Value::Tuple(items))
    }

    fn parse_u256(input: String) -> Result<Value> {
        let hash = h256_from_hex_be(input.as_str()).expect("invalid contract address");
        let u256 = h256_to_u32_array(&hash);
        Ok(Value::U256(FixedArray8(u256)))
    }
}

pub struct FromValue;
impl FromValue {
    pub fn parse_input(input: Value) -> String {
        let parse_result = match input {
            Value::U32(input) => Self::parse_u32(input),
            Value::Field(input) => Self::parse_field(input),
            Value::Address(input) => Self::parse_address(input),
            Value::Hash(input) => Self::parse_hash(input),
            Value::Bool(input) => Self::parse_bool(input),
            Value::FixedArray(input, t) => Self::parse_fixed_array(input, t),
            Value::String(input) => Self::parse_string(input),
            Value::Fields(input) => Self::parse_fields(input),
            Value::Array(input, t) => Self::parse_array(input, t),
            Value::Tuple(input) => Self::parse_tuple(input),
            Value::U256(input) => Self::parse_u256(input),
        };
        parse_result.unwrap()
    }

    fn parse_u32(input: u64) -> Result<String> {
        Ok((input as u32).to_string())
    }

    fn parse_field(input: u64) -> Result<String> {
        if input > OLA_FIELD_ORDER {
            bail!("invalid field element input")
        }
        Ok(input.to_string())
    }

    fn parse_hash(input: FixedArray4) -> Result<String> {
        let hash = u64_array_to_h256(&input.0);
        Ok(hex::encode(hash.0))
    }

    fn parse_address(input: FixedArray4) -> Result<String> {
        Self::parse_hash(input)
    }

    fn parse_bool(input: bool) -> Result<String> {
        Ok(input.to_string())
    }

    fn parse_fixed_array(input: Vec<Value>, t: Type) -> Result<String> {
        match t {
            Type::U32
            | Type::Field
            | Type::Hash
            | Type::Address
            | Type::Bool
            | Type::String
            | Type::Fields => {
                let mut ret = String::from("[");
                input.iter().for_each(|i| {
                    let s = Self::parse_input(i.clone());

                    ret += &s;
                    ret += ",";
                });
                ret.pop();
                ret += "]";
                Ok(ret)
            }
            Type::FixedArray(_, _) | Type::Array(_) | Type::Tuple(_) | Type::U256 => {
                bail!("Composite types in FixedArray has not been supported for cli tools.")
            }
        }
    }

    fn parse_string(input: String) -> Result<String> {
        Ok(input)
    }

    fn parse_fields(input: Vec<u64>) -> Result<String> {
        let mut ret = String::from("0x");
        input.iter().for_each(|i| {
            if *i > OLA_FIELD_ORDER {
                panic!("invalid field element input")
            }

            ret += &format!("{:x}", i);
        });
        Ok(ret)
    }

    fn parse_array(input: Vec<Value>, t: Type) -> Result<String> {
        match t {
            Type::U32
            | Type::Field
            | Type::Hash
            | Type::Address
            | Type::Bool
            | Type::String
            | Type::Fields => {
                let mut ret = String::from("[");
                input.iter().for_each(|i| {
                    let s = Self::parse_input(i.clone());

                    ret += &s;
                    ret += ",";
                });
                ret.pop();
                ret += "]";
                Ok(ret)
            }
            Type::FixedArray(_, _) | Type::Array(_) | Type::Tuple(_) | Type::U256 => {
                bail!("Composite types in Array has not been supported for cli tools.")
            }
        }
    }

    fn parse_tuple(input: Vec<(String, Value)>) -> Result<String> {
        let mut ret = String::from("{");
        input.iter().for_each(|i| {
            match i.1 {
                Value::FixedArray(_, _) | Value::Array(_, _) | Value::Tuple(_) => {
                    panic!("Composite types in Tuple has not been supported for cli tools.")
                }
                _ => {}
            }

            let v = Self::parse_input(i.1.clone());
            ret += format!("{}: {},", i.0, v).as_str();
        });
        ret.pop();
        ret += "}";

        Ok(ret)
    }

    fn parse_u256(input: FixedArray8) -> Result<String> {
        let u256 = u32_array_to_h256(&input.0);
        Ok(hex::encode(u256.0))
    }
}
