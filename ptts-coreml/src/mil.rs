//! A small builder for CoreML ML Program graphs.
//!
//! Covers the ops this model needs rather than all of MIL. Adding one is a few lines: give it a
//! type string, bind its inputs, declare the output shape. Every shape is concrete: the Neural
//! Engine will not take a graph with a symbolic dimension anywhere in it.

use crate::blob::{BlobDType, BlobWriter};
use crate::proto::core_ml::specification as spec;
use crate::proto::core_ml::specification::mil_spec as m;
use std::collections::HashMap;

pub const OPSET: &str = "CoreML8";
pub const SPEC_VERSION: i32 = 9;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    Fp32,
    Fp16,
    Int32,
    Str,
    Bool,
}

impl DType {
    /// MIL `DataType` enum value.
    fn mil(self) -> i32 {
        match self {
            DType::Fp16 => 10,
            DType::Fp32 => 11,
            DType::Int32 => 23,
            DType::Str => 2,
            DType::Bool => 1,
        }
    }
    /// `ArrayFeatureType.ArrayDataType`, which uses a different numbering to MIL's.
    fn array(self) -> i32 {
        match self {
            DType::Fp16 => 65552,
            DType::Fp32 => 65568,
            DType::Int32 => 131104,
            DType::Str | DType::Bool => {
                panic!("strings and bools are op parameters, never model inputs or outputs")
            }
        }
    }
}

/// The GELU a checkpoint was trained with: the exact one, or its tanh approximation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Gelu {
    Tanh,
    #[default]
    Erf,
}

fn tensor_type(dt: DType, shape: &[usize]) -> m::ValueType {
    m::ValueType {
        r#type: Some(m::value_type::Type::TensorType(m::TensorType {
            data_type: dt.mil(),
            rank: shape.len() as i64,
            dimensions: shape
                .iter()
                .map(|&d| m::Dimension {
                    dimension: Some(m::dimension::Dimension::Constant(
                        m::dimension::ConstantDimension { size: d as u64 },
                    )),
                })
                .collect(),
            attributes: HashMap::new(),
        })),
    }
}

fn str_value(s: &str) -> m::Value {
    m::Value {
        doc_string: String::new(),
        // Every attribute Value needs a type or the compiler rejects the op with
        // "Cannot parse a null value type". STRING is MIL DataType 2, rank 0.
        r#type: Some(m::ValueType {
            r#type: Some(m::value_type::Type::TensorType(m::TensorType {
                data_type: 2,
                rank: 0,
                dimensions: Vec::new(),
                attributes: HashMap::new(),
            })),
        }),
        value: Some(m::value::Value::ImmediateValue(m::value::ImmediateValue {
            value: Some(m::value::immediate_value::Value::Tensor(m::TensorValue {
                value: Some(m::tensor_value::Value::Strings(m::tensor_value::RepeatedStrings {
                    values: vec![s.to_string()],
                })),
            })),
        })),
    }
}

/// A value in the graph: its name plus enough type information to declare consumers.
#[derive(Clone, Debug)]
pub struct Var {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

pub struct Builder {
    ops: Vec<m::Operation>,
    inputs: Vec<m::NamedValueType>,
    /// The same inputs as `(name, type, shape)`, for the model description.
    described: Vec<(String, DType, Vec<usize>)>,
    /// Element type every float constant is emitted in. CoreML requires an op's inputs and
    /// output to agree exactly, so this has to match whatever the graph's tensors are.
    float: DType,
    /// Weight payloads, referenced from the graph by offset.
    blob: BlobWriter,
    next: usize,
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl Builder {
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            inputs: Vec::new(),
            described: Vec::new(),
            float: DType::Fp32,
            blob: BlobWriter::new(),
            next: 0,
        }
    }

    /// Emit float constants in `dt`. Set this before building anything.
    pub fn with_float(mut self, dt: DType) -> Self {
        self.float = dt;
        self
    }

    /// A weight: in the blob unless it is tiny, in the graph's float type.
    pub fn weight(&mut self, data: &[f32], shape: &[usize]) -> Var {
        if data.len() >= 4096 { self.const_blob(data, shape) } else { self.const_f32(data, shape) }
    }

    fn value_from_blob(
        &mut self,
        dt: DType,
        bdt: BlobDType,
        bytes: &[u8],
        shape: &[usize],
    ) -> m::Value {
        let off = self.blob.add(bdt, bytes);
        m::Value {
            doc_string: String::new(),
            r#type: Some(tensor_type(dt, shape)),
            value: Some(m::value::Value::BlobFileValue(m::value::BlobFileValue {
                file_name: "@model_path/weights/weight.bin".to_string(),
                offset: off,
            })),
        }
    }

    /// A float constant held in the weights blob.
    pub fn const_blob(&mut self, data: &[f32], shape: &[usize]) -> Var {
        let dt = self.float;
        let (bdt, bytes) = match dt {
            DType::Fp16 => (
                BlobDType::Fp16,
                data.iter()
                    .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                    .collect::<Vec<u8>>(),
            ),
            _ => (BlobDType::Fp32, data.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()),
        };
        let val = self.value_from_blob(dt, bdt, &bytes, shape);
        let name = self.fresh("c");
        self.emit_const(&name, val, dt, shape)
    }

    fn fresh(&mut self, prefix: &str) -> String {
        self.next += 1;
        format!("{prefix}_{}", self.next)
    }

    /// Declare a model input.
    pub fn input(&mut self, name: &str, dtype: DType, shape: &[usize]) -> Var {
        self.inputs.push(m::NamedValueType {
            name: name.to_string(),
            r#type: Some(tensor_type(dtype, shape)),
        });
        self.described.push((name.to_string(), dtype, shape.to_vec()));
        Var { name: name.to_string(), dtype, shape: shape.to_vec() }
    }

    /// An inline constant. Large tensors should go in the weights blob instead; this puts the
    /// bytes straight into the protobuf.
    pub fn const_f32(&mut self, data: &[f32], shape: &[usize]) -> Var {
        let name = self.fresh("c");
        let dt = self.float;
        // fp16 has no repeated field in TensorValue, so it travels as raw little-endian bytes.
        let payload = match dt {
            DType::Fp16 => {
                let mut bytes = Vec::with_capacity(data.len() * 2);
                for &v in data {
                    bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
                }
                m::tensor_value::Value::Bytes(m::tensor_value::RepeatedBytes { values: bytes })
            }
            _ => m::tensor_value::Value::Floats(m::tensor_value::RepeatedFloats {
                values: data.to_vec(),
            }),
        };
        let val = m::Value {
            doc_string: String::new(),
            r#type: Some(tensor_type(dt, shape)),
            value: Some(m::value::Value::ImmediateValue(m::value::ImmediateValue {
                value: Some(m::value::immediate_value::Value::Tensor(m::TensorValue {
                    value: Some(payload),
                })),
            })),
        };
        self.emit_const(&name, val, dt, shape)
    }

    pub fn const_i32(&mut self, data: &[i32], shape: &[usize]) -> Var {
        let name = self.fresh("c");
        let val = m::Value {
            doc_string: String::new(),
            r#type: Some(tensor_type(DType::Int32, shape)),
            value: Some(m::value::Value::ImmediateValue(m::value::ImmediateValue {
                value: Some(m::value::immediate_value::Value::Tensor(m::TensorValue {
                    value: Some(m::tensor_value::Value::Ints(m::tensor_value::RepeatedInts {
                        values: data.to_vec(),
                    })),
                })),
            })),
        };
        self.emit_const(&name, val, DType::Int32, shape)
    }

    /// A string parameter, e.g. `gelu`'s `mode` or `conv`'s `pad_type`.
    pub fn const_str(&mut self, v: &str) -> Var {
        let name = self.fresh("s");
        let mut val = str_value(v);
        val.r#type = Some(tensor_type(DType::Str, &[]));
        self.emit_const(&name, val, DType::Str, &[])
    }

    pub fn scalar_f32(&mut self, v: f32) -> Var {
        self.const_f32(&[v], &[])
    }

    pub fn scalar_i32(&mut self, v: i32) -> Var {
        self.const_i32(&[v], &[])
    }

    fn emit_const(&mut self, name: &str, val: m::Value, dt: DType, shape: &[usize]) -> Var {
        // `const` takes no input bindings: coremltools puts the payload in attributes["val"].
        let mut attributes = HashMap::new();
        attributes.insert("name".to_string(), str_value(name));
        attributes.insert("val".to_string(), val);
        self.ops.push(m::Operation {
            r#type: "const".to_string(),
            inputs: HashMap::new(),
            outputs: vec![m::NamedValueType {
                name: name.to_string(),
                r#type: Some(tensor_type(dt, shape)),
            }],
            blocks: Vec::new(),
            attributes,
        });
        Var { name: name.to_string(), dtype: dt, shape: shape.to_vec() }
    }

    /// `concat`, whose `values` parameter takes several bindings under one key.
    pub fn concat(&mut self, values: &[&Var], axis: i32, out_shape: &[usize]) -> Var {
        let dtype = values[0].dtype;
        let ax = self.scalar_i32(axis);
        let no = self.const_bool(false);
        let name = self.fresh("v");
        let mut inputs = HashMap::new();
        inputs.insert(
            "values".to_string(),
            m::Argument {
                arguments: values
                    .iter()
                    .map(|v| m::argument::Binding {
                        binding: Some(m::argument::binding::Binding::Name(v.name.clone())),
                    })
                    .collect(),
            },
        );
        for (k, v) in [("axis", &ax), ("interleave", &no)] {
            inputs.insert(
                k.to_string(),
                m::Argument {
                    arguments: vec![m::argument::Binding {
                        binding: Some(m::argument::binding::Binding::Name(v.name.clone())),
                    }],
                },
            );
        }
        let mut attributes = HashMap::new();
        attributes.insert("name".to_string(), str_value(&name));
        self.ops.push(m::Operation {
            r#type: "concat".to_string(),
            inputs,
            outputs: vec![m::NamedValueType {
                name: name.clone(),
                r#type: Some(tensor_type(dtype, out_shape)),
            }],
            blocks: Vec::new(),
            attributes,
        });
        Var { name, dtype, shape: out_shape.to_vec() }
    }

    pub fn const_bool(&mut self, v: bool) -> Var {
        let name = self.fresh("b");
        let val = m::Value {
            doc_string: String::new(),
            r#type: Some(tensor_type(DType::Bool, &[])),
            value: Some(m::value::Value::ImmediateValue(m::value::ImmediateValue {
                value: Some(m::value::immediate_value::Value::Tensor(m::TensorValue {
                    value: Some(m::tensor_value::Value::Bools(m::tensor_value::RepeatedBools {
                        values: vec![v],
                    })),
                })),
            })),
        };
        self.emit_const(&name, val, DType::Bool, &[])
    }

    pub fn reshape(&mut self, x: &Var, shape: &[usize]) -> Var {
        let sv =
            self.const_i32(&shape.iter().map(|&d| d as i32).collect::<Vec<_>>(), &[shape.len()]);
        let dt = x.dtype;
        self.op("reshape", &[("x", x), ("shape", &sv)], dt, shape)
    }

    pub fn transpose(&mut self, x: &Var, perm: &[i32], out_shape: &[usize]) -> Var {
        let p = self.const_i32(perm, &[perm.len()]);
        let dt = x.dtype;
        self.op("transpose", &[("x", x), ("perm", &p)], dt, out_shape)
    }

    /// `matmul` with optional transposes, as MIL spells them.
    pub fn matmul(&mut self, a: &Var, b: &Var, tb: bool, out_shape: &[usize]) -> Var {
        let fa = self.const_bool(false);
        let fb = self.const_bool(tb);
        let dt = a.dtype;
        self.op(
            "matmul",
            &[("x", a), ("y", b), ("transpose_x", &fa), ("transpose_y", &fb)],
            dt,
            out_shape,
        )
    }

    pub fn add(&mut self, a: &Var, b: &Var, out_shape: &[usize]) -> Var {
        let dt = a.dtype;
        self.op("add", &[("x", a), ("y", b)], dt, out_shape)
    }

    pub fn sub(&mut self, a: &Var, b: &Var, out_shape: &[usize]) -> Var {
        let dt = a.dtype;
        self.op("sub", &[("x", a), ("y", b)], dt, out_shape)
    }

    pub fn mul(&mut self, a: &Var, b: &Var, out_shape: &[usize]) -> Var {
        let dt = a.dtype;
        self.op("mul", &[("x", a), ("y", b)], dt, out_shape)
    }

    pub fn neg(&mut self, x: &Var) -> Var {
        let m1 = match x.dtype {
            DType::Fp16 | DType::Fp32 => self.scalar_f32(-1.0),
            _ => self.scalar_i32(-1),
        };
        let sh = x.shape.clone();
        self.mul(x, &m1, &sh)
    }

    pub fn layer_norm(&mut self, x: &Var, gamma: &Var, beta: &Var, eps: f32) -> Var {
        let axes = self.const_i32(&[-1], &[1]);
        let e = self.scalar_f32(eps);
        let dt = x.dtype;
        let sh = x.shape.clone();
        self.op(
            "layer_norm",
            &[("x", x), ("axes", &axes), ("gamma", gamma), ("beta", beta), ("epsilon", &e)],
            dt,
            &sh,
        )
    }

    pub fn gelu(&mut self, x: &Var, kind: Gelu) -> Var {
        let mode = self.const_str(match kind {
            Gelu::Erf => "EXACT",
            Gelu::Tanh => "TANH_APPROXIMATION",
        });
        let dt = x.dtype;
        let sh = x.shape.clone();
        self.op("gelu", &[("x", x), ("mode", &mode)], dt, &sh)
    }

    pub fn silu(&mut self, x: &Var) -> Var {
        let dt = x.dtype;
        let sh = x.shape.clone();
        self.op("silu", &[("x", x)], dt, &sh)
    }

    pub fn elu(&mut self, x: &Var) -> Var {
        let a = self.scalar_f32(1.0);
        let dt = x.dtype;
        let sh = x.shape.clone();
        self.op("elu", &[("x", x), ("alpha", &a)], dt, &sh)
    }

    pub fn softmax(&mut self, x: &Var, axis: i32) -> Var {
        let ax = self.scalar_i32(axis);
        let dt = x.dtype;
        let sh = x.shape.clone();
        self.op("softmax", &[("x", x), ("axis", &ax)], dt, &sh)
    }

    pub fn linear(&mut self, x: &Var, w: &Var, bias: Option<&Var>, out_shape: &[usize]) -> Var {
        let dt = x.dtype;
        match bias {
            Some(b) => self.op("linear", &[("x", x), ("weight", w), ("bias", b)], dt, out_shape),
            None => {
                // `linear` requires a bias; a zero vector is cheaper than a separate matmul path.
                let n = *out_shape.last().unwrap();
                let z = self.const_f32(&vec![0f32; n], &[n]);
                self.op("linear", &[("x", x), ("weight", w), ("bias", &z)], dt, out_shape)
            }
        }
    }

    /// 1-D convolution, no padding: streaming layers carry their left context in a buffer.
    pub fn conv1d(
        &mut self,
        x: &Var,
        w: &Var,
        bias: Option<&Var>,
        groups: usize,
        out_shape: &[usize],
    ) -> Var {
        let strides = self.const_i32(&[1], &[1]);
        let dil = self.const_i32(&[1], &[1]);
        let pad = self.const_i32(&[0, 0], &[2]);
        let pt = self.const_str("custom");
        let g = self.scalar_i32(groups as i32);
        let dt = x.dtype;
        let mut args: Vec<(&str, &Var)> = vec![
            ("x", x),
            ("weight", w),
            ("strides", &strides),
            ("dilations", &dil),
            ("pad", &pad),
            ("pad_type", &pt),
            ("groups", &g),
        ];
        if let Some(b) = bias {
            args.push(("bias", b));
        }
        self.op("conv", &args, dt, out_shape)
    }

    /// 1-D transposed convolution. `out_shape` is passed explicitly, as MIL wants.
    pub fn conv_transpose1d(
        &mut self,
        x: &Var,
        w: &Var,
        bias: Option<&Var>,
        stride: usize,
        groups: usize,
        out_shape: &[usize],
    ) -> Var {
        let strides = self.const_i32(&[stride as i32], &[1]);
        let dil = self.const_i32(&[1], &[1]);
        let pad = self.const_i32(&[0, 0], &[2]);
        let pt = self.const_str("custom");
        let g = self.scalar_i32(groups as i32);
        let os = self.const_i32(
            &out_shape.iter().map(|&d| d as i32).collect::<Vec<_>>(),
            &[out_shape.len()],
        );
        let dt = x.dtype;
        let mut args: Vec<(&str, &Var)> = vec![
            ("x", x),
            ("weight", w),
            ("strides", &strides),
            ("dilations", &dil),
            ("pad", &pad),
            ("pad_type", &pt),
            ("groups", &g),
            ("output_shape", &os),
        ];
        if let Some(b) = bias {
            args.push(("bias", b));
        }
        self.op("conv_transpose", &args, dt, out_shape)
    }

    /// `x[..., -n.., ...]` along `axis`: a tail whose start is measured from the end, so it
    /// does not depend on the (possibly symbolic) length of the axis.
    pub fn slice_tail(&mut self, x: &Var, axis: usize, n: usize, out_shape: &[usize]) -> Var {
        let rank = x.shape.len();
        let mut begin = vec![0i32; rank];
        begin[axis] = -(n as i32);
        let bmask: Vec<bool> = (0..rank).map(|i| i != axis).collect();
        let emask = vec![true; rank];
        self.slice_masked(x, &begin, &vec![0i32; rank], &bmask, &emask, out_shape)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn slice_masked(
        &mut self,
        x: &Var,
        begin: &[i32],
        end: &[i32],
        begin_mask: &[bool],
        end_mask: &[bool],
        out_shape: &[usize],
    ) -> Var {
        let b = self.const_i32(begin, &[begin.len()]);
        let e = self.const_i32(end, &[end.len()]);
        let bm = self.const_bools(begin_mask);
        let em = self.const_bools(end_mask);
        let dt = x.dtype;
        self.op(
            "slice_by_index",
            &[("x", x), ("begin", &b), ("end", &e), ("begin_mask", &bm), ("end_mask", &em)],
            dt,
            out_shape,
        )
    }

    pub fn const_bools(&mut self, vs: &[bool]) -> Var {
        let name = self.fresh("b");
        let val = m::Value {
            doc_string: String::new(),
            r#type: Some(tensor_type(DType::Bool, &[vs.len()])),
            value: Some(m::value::Value::ImmediateValue(m::value::ImmediateValue {
                value: Some(m::value::immediate_value::Value::Tensor(m::TensorValue {
                    value: Some(m::tensor_value::Value::Bools(m::tensor_value::RepeatedBools {
                        values: vs.to_vec(),
                    })),
                })),
            })),
        };
        self.emit_const(&name, val, DType::Bool, &[vs.len()])
    }

    /// A contiguous slice: `x[begin..end]` per axis, with masks selecting which bounds apply.
    pub fn slice(&mut self, x: &Var, begin: &[i32], end: &[i32], out_shape: &[usize]) -> Var {
        let b = self.const_i32(begin, &[begin.len()]);
        let e = self.const_i32(end, &[end.len()]);
        let dtype = x.dtype;
        self.op("slice_by_index", &[("x", x), ("begin", &b), ("end", &e)], dtype, out_shape)
    }

    /// Give a value a stable name, so callers can find it in the output map.
    ///
    /// CoreML returns outputs keyed by name and `featureNames()` is unordered, so anything
    /// positional -- carrying streaming state between calls -- needs the names pinned.
    pub fn alias(&mut self, v: &Var, name: &str) -> Var {
        let mut attributes = HashMap::new();
        attributes.insert("name".to_string(), str_value(name));
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            m::Argument {
                arguments: vec![m::argument::Binding {
                    binding: Some(m::argument::binding::Binding::Name(v.name.clone())),
                }],
            },
        );
        self.ops.push(m::Operation {
            r#type: "identity".to_string(),
            inputs,
            outputs: vec![m::NamedValueType {
                name: name.to_string(),
                r#type: Some(tensor_type(v.dtype, &v.shape)),
            }],
            blocks: Vec::new(),
            attributes,
        });
        Var { name: name.to_string(), dtype: v.dtype, shape: v.shape.clone() }
    }

    /// Emit an op whose inputs are all previously-defined values.
    pub fn op(
        &mut self,
        ty: &str,
        args: &[(&str, &Var)],
        out_dtype: DType,
        out_shape: &[usize],
    ) -> Var {
        let name = self.fresh("v");
        let inputs = args
            .iter()
            .map(|(k, v)| {
                (
                    k.to_string(),
                    m::Argument {
                        arguments: vec![m::argument::Binding {
                            binding: Some(m::argument::binding::Binding::Name(v.name.clone())),
                        }],
                    },
                )
            })
            .collect();
        let mut attributes = HashMap::new();
        attributes.insert("name".to_string(), str_value(&name));
        self.ops.push(m::Operation {
            r#type: ty.to_string(),
            inputs,
            outputs: vec![m::NamedValueType {
                name: name.clone(),
                r#type: Some(tensor_type(out_dtype, out_shape)),
            }],
            blocks: Vec::new(),
            attributes,
        });
        Var { name, dtype: out_dtype, shape: out_shape.to_vec() }
    }

    /// Finish, producing the model and the weights blob it refers to.
    pub fn finish_with_weights(self, outputs: &[&Var]) -> (spec::Model, Option<Vec<u8>>) {
        let weights = (!self.blob.is_empty()).then(|| self.blob.finish());
        let block = m::Block {
            inputs: Vec::new(),
            outputs: outputs.iter().map(|v| v.name.clone()).collect(),
            operations: self.ops,
            attributes: HashMap::new(),
        };
        let function = m::Function {
            inputs: self.inputs.clone(),
            opset: OPSET.to_string(),
            block_specializations: HashMap::from([(OPSET.to_string(), block)]),
            attributes: HashMap::new(),
        };
        let feature = |name: &str, dt: DType, shape: &[usize]| spec::FeatureDescription {
            name: name.to_string(),
            short_description: String::new(),
            r#type: Some(spec::FeatureType {
                is_optional: false,
                r#type: Some(spec::feature_type::Type::MultiArrayType(spec::ArrayFeatureType {
                    shape: shape.iter().map(|&d| d as i64).collect(),
                    data_type: dt.array(),
                })),
            }),
        };
        let model = spec::Model {
            specification_version: SPEC_VERSION,
            description: Some(spec::ModelDescription {
                input: self
                    .described
                    .iter()
                    .map(|(name, dt, shape)| feature(name, *dt, shape))
                    .collect(),
                output: outputs.iter().map(|v| feature(&v.name, v.dtype, &v.shape)).collect(),
            }),
            is_updatable: false,
            r#type: Some(spec::model::Type::MlProgram(m::Program {
                version: 1,
                functions: HashMap::from([("main".to_string(), function)]),
                doc_string: String::new(),
                attributes: HashMap::new(),
            })),
        };
        (model, weights)
    }
}
