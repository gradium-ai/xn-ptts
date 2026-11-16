//! The part of Apple's Core ML model schema this crate writes, as `prost` messages.
//!
//! Hand-written from coremltools' `Model.proto`, `FeatureTypes.proto` and `MIL.proto` (BSD
//! 3-clause, Apple), keeping only the messages and fields an ML Program uses here. Field numbers
//! and wire types are the schema's, so what this encodes is an ordinary Core ML model; enum
//! fields travel as their `int32` values, which is how protobuf encodes them.

pub mod core_ml {
    pub mod specification {
        use std::collections::HashMap;

        #[derive(Clone, PartialEq, prost::Message)]
        pub struct Model {
            #[prost(int32, tag = "1")]
            pub specification_version: i32,
            #[prost(message, optional, tag = "2")]
            pub description: Option<ModelDescription>,
            #[prost(bool, tag = "10")]
            pub is_updatable: bool,
            #[prost(oneof = "model::Type", tags = "502")]
            pub r#type: Option<model::Type>,
        }

        pub mod model {
            #[derive(Clone, PartialEq, prost::Oneof)]
            pub enum Type {
                #[prost(message, tag = "502")]
                MlProgram(super::mil_spec::Program),
            }
        }

        /// A program with one function describes it in `input` and `output`; one with several
        /// leaves those empty and describes each in `functions` instead.
        #[derive(Clone, PartialEq, prost::Message)]
        pub struct ModelDescription {
            #[prost(message, repeated, tag = "1")]
            pub input: Vec<FeatureDescription>,
            #[prost(message, repeated, tag = "10")]
            pub output: Vec<FeatureDescription>,
            #[prost(message, repeated, tag = "20")]
            pub functions: Vec<FunctionDescription>,
            #[prost(string, tag = "21")]
            pub default_function_name: String,
        }

        #[derive(Clone, PartialEq, prost::Message)]
        pub struct FunctionDescription {
            #[prost(string, tag = "1")]
            pub name: String,
            #[prost(message, repeated, tag = "2")]
            pub input: Vec<FeatureDescription>,
            #[prost(message, repeated, tag = "3")]
            pub output: Vec<FeatureDescription>,
        }

        #[derive(Clone, PartialEq, prost::Message)]
        pub struct FeatureDescription {
            #[prost(string, tag = "1")]
            pub name: String,
            #[prost(string, tag = "2")]
            pub short_description: String,
            #[prost(message, optional, tag = "3")]
            pub r#type: Option<FeatureType>,
        }

        #[derive(Clone, PartialEq, prost::Message)]
        pub struct FeatureType {
            #[prost(bool, tag = "1000")]
            pub is_optional: bool,
            #[prost(oneof = "feature_type::Type", tags = "5")]
            pub r#type: Option<feature_type::Type>,
        }

        pub mod feature_type {
            #[derive(Clone, PartialEq, prost::Oneof)]
            pub enum Type {
                #[prost(message, tag = "5")]
                MultiArrayType(super::ArrayFeatureType),
            }
        }

        /// `data_type` is `ArrayFeatureType.ArrayDataType`: 65552 fp16, 65568 fp32, 131104 int32.
        #[derive(Clone, PartialEq, prost::Message)]
        pub struct ArrayFeatureType {
            #[prost(int64, repeated, tag = "1")]
            pub shape: Vec<i64>,
            #[prost(int32, tag = "2")]
            pub data_type: i32,
        }

        pub mod mil_spec {
            use super::HashMap;

            #[derive(Clone, PartialEq, prost::Message)]
            pub struct Program {
                #[prost(int64, tag = "1")]
                pub version: i64,
                #[prost(map = "string, message", tag = "2")]
                pub functions: HashMap<String, Function>,
                #[prost(string, tag = "3")]
                pub doc_string: String,
                #[prost(map = "string, message", tag = "4")]
                pub attributes: HashMap<String, Value>,
            }

            #[derive(Clone, PartialEq, prost::Message)]
            pub struct Function {
                #[prost(message, repeated, tag = "1")]
                pub inputs: Vec<NamedValueType>,
                #[prost(string, tag = "2")]
                pub opset: String,
                #[prost(map = "string, message", tag = "3")]
                pub block_specializations: HashMap<String, Block>,
                #[prost(map = "string, message", tag = "4")]
                pub attributes: HashMap<String, Value>,
            }

            #[derive(Clone, PartialEq, prost::Message)]
            pub struct Block {
                #[prost(message, repeated, tag = "1")]
                pub inputs: Vec<NamedValueType>,
                #[prost(string, repeated, tag = "2")]
                pub outputs: Vec<String>,
                #[prost(message, repeated, tag = "3")]
                pub operations: Vec<Operation>,
                #[prost(map = "string, message", tag = "4")]
                pub attributes: HashMap<String, Value>,
            }

            #[derive(Clone, PartialEq, prost::Message)]
            pub struct Argument {
                #[prost(message, repeated, tag = "1")]
                pub arguments: Vec<argument::Binding>,
            }

            pub mod argument {
                #[derive(Clone, PartialEq, prost::Message)]
                pub struct Binding {
                    #[prost(oneof = "binding::Binding", tags = "1")]
                    pub binding: Option<binding::Binding>,
                }

                pub mod binding {
                    #[derive(Clone, PartialEq, prost::Oneof)]
                    pub enum Binding {
                        #[prost(string, tag = "1")]
                        Name(String),
                    }
                }
            }

            #[derive(Clone, PartialEq, prost::Message)]
            pub struct Operation {
                #[prost(string, tag = "1")]
                pub r#type: String,
                #[prost(map = "string, message", tag = "2")]
                pub inputs: HashMap<String, Argument>,
                #[prost(message, repeated, tag = "3")]
                pub outputs: Vec<NamedValueType>,
                #[prost(message, repeated, tag = "4")]
                pub blocks: Vec<Block>,
                #[prost(map = "string, message", tag = "5")]
                pub attributes: HashMap<String, Value>,
            }

            #[derive(Clone, PartialEq, prost::Message)]
            pub struct NamedValueType {
                #[prost(string, tag = "1")]
                pub name: String,
                #[prost(message, optional, tag = "2")]
                pub r#type: Option<ValueType>,
            }

            #[derive(Clone, PartialEq, prost::Message)]
            pub struct ValueType {
                #[prost(oneof = "value_type::Type", tags = "1")]
                pub r#type: Option<value_type::Type>,
            }

            pub mod value_type {
                #[derive(Clone, PartialEq, prost::Oneof)]
                pub enum Type {
                    #[prost(message, tag = "1")]
                    TensorType(super::TensorType),
                }
            }

            /// `data_type` is MIL's `DataType`: 1 bool, 2 string, 10 fp16, 11 fp32, 23 int32.
            #[derive(Clone, PartialEq, prost::Message)]
            pub struct TensorType {
                #[prost(int32, tag = "1")]
                pub data_type: i32,
                #[prost(int64, tag = "2")]
                pub rank: i64,
                #[prost(message, repeated, tag = "3")]
                pub dimensions: Vec<Dimension>,
                #[prost(map = "string, message", tag = "4")]
                pub attributes: HashMap<String, Value>,
            }

            #[derive(Clone, PartialEq, prost::Message)]
            pub struct Dimension {
                #[prost(oneof = "dimension::Dimension", tags = "1")]
                pub dimension: Option<dimension::Dimension>,
            }

            pub mod dimension {
                #[derive(Clone, PartialEq, prost::Oneof)]
                pub enum Dimension {
                    #[prost(message, tag = "1")]
                    Constant(ConstantDimension),
                }

                #[derive(Clone, PartialEq, prost::Message)]
                pub struct ConstantDimension {
                    #[prost(uint64, tag = "1")]
                    pub size: u64,
                }
            }

            #[derive(Clone, PartialEq, prost::Message)]
            pub struct Value {
                #[prost(string, tag = "1")]
                pub doc_string: String,
                #[prost(message, optional, tag = "2")]
                pub r#type: Option<ValueType>,
                #[prost(oneof = "value::Value", tags = "3, 5")]
                pub value: Option<value::Value>,
            }

            pub mod value {
                #[derive(Clone, PartialEq, prost::Oneof)]
                pub enum Value {
                    #[prost(message, tag = "3")]
                    ImmediateValue(ImmediateValue),
                    #[prost(message, tag = "5")]
                    BlobFileValue(BlobFileValue),
                }

                #[derive(Clone, PartialEq, prost::Message)]
                pub struct ImmediateValue {
                    #[prost(oneof = "immediate_value::Value", tags = "1")]
                    pub value: Option<immediate_value::Value>,
                }

                pub mod immediate_value {
                    #[derive(Clone, PartialEq, prost::Oneof)]
                    pub enum Value {
                        #[prost(message, tag = "1")]
                        Tensor(super::super::TensorValue),
                    }
                }

                #[derive(Clone, PartialEq, prost::Message)]
                pub struct BlobFileValue {
                    #[prost(string, tag = "1")]
                    pub file_name: String,
                    #[prost(uint64, tag = "2")]
                    pub offset: u64,
                }
            }

            #[derive(Clone, PartialEq, prost::Message)]
            pub struct TensorValue {
                #[prost(oneof = "tensor_value::Value", tags = "1, 2, 3, 4, 7")]
                pub value: Option<tensor_value::Value>,
            }

            pub mod tensor_value {
                #[derive(Clone, PartialEq, prost::Oneof)]
                pub enum Value {
                    #[prost(message, tag = "1")]
                    Floats(RepeatedFloats),
                    #[prost(message, tag = "2")]
                    Ints(RepeatedInts),
                    #[prost(message, tag = "3")]
                    Bools(RepeatedBools),
                    #[prost(message, tag = "4")]
                    Strings(RepeatedStrings),
                    #[prost(message, tag = "7")]
                    Bytes(RepeatedBytes),
                }

                #[derive(Clone, PartialEq, prost::Message)]
                pub struct RepeatedFloats {
                    #[prost(float, repeated, tag = "1")]
                    pub values: Vec<f32>,
                }

                #[derive(Clone, PartialEq, prost::Message)]
                pub struct RepeatedInts {
                    #[prost(int32, repeated, tag = "1")]
                    pub values: Vec<i32>,
                }

                #[derive(Clone, PartialEq, prost::Message)]
                pub struct RepeatedBools {
                    #[prost(bool, repeated, tag = "1")]
                    pub values: Vec<bool>,
                }

                #[derive(Clone, PartialEq, prost::Message)]
                pub struct RepeatedStrings {
                    #[prost(string, repeated, tag = "1")]
                    pub values: Vec<String>,
                }

                #[derive(Clone, PartialEq, prost::Message)]
                pub struct RepeatedBytes {
                    #[prost(bytes = "vec", tag = "1")]
                    pub values: Vec<u8>,
                }
            }
        }
    }
}
