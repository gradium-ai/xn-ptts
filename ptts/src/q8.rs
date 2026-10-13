//! A q8_0 matmul for weights that are otherwise kept in f32, used by the opt-in Mimi
//! quantization ([`crate::tts_model::TTSModel::quantize_mimi`]).
//!
//! xn's quantized linears exist on the CPU only, while the layers that hold these weights are
//! generic over the dtype and the backend. Both conversions below are checked downcasts that
//! succeed exactly when the layer is an f32 layer on the CPU, so no data is copied.

use std::any::Any;
use xn::nn::Linear;
use xn::quantized::{GgmlDType, QLinear};
use xn::{Backend, CpuDevice, Result, Tensor, WithDTypeF};

pub(crate) struct Q8Linear(Box<QLinear>);

fn as_cpu<T: WithDTypeF, B: Backend>(x: &Tensor<T, B>) -> Result<&Tensor<f32, CpuDevice>> {
    match (x as &dyn Any).downcast_ref::<Tensor<f32, CpuDevice>>() {
        Some(x) => Ok(x),
        None => xn::bail!("Mimi quantization needs f32 weights on the CPU"),
    }
}

impl Q8Linear {
    /// `weight` is `[n, k]`, `k` a multiple of 32. No bias.
    pub(crate) fn new<T: WithDTypeF, B: Backend>(weight: &Tensor<T, B>) -> Result<Self> {
        Self::from_linear(&Linear::new(weight.clone()))
    }

    /// A q8_0 copy of `linear`, its bias included.
    pub(crate) fn from_linear<T: WithDTypeF, B: Backend>(linear: &Linear<T, B>) -> Result<Self> {
        let mut cpu = Linear::new(as_cpu(linear.weight())?.clone());
        if let Some(bias) = linear.bias() {
            cpu = cpu.with_bias(as_cpu(bias)?.clone());
        }
        Ok(Self(Box::new(QLinear::from_linear(cpu, GgmlDType::Q8_0)?)))
    }

    /// `x` is `[..., k]`; returns `[..., n]`.
    pub(crate) fn forward<T: WithDTypeF, B: Backend>(
        &self,
        x: &Tensor<T, B>,
    ) -> Result<Tensor<T, B>> {
        use xn::ModuleT;
        let y: Box<dyn Any> = Box::new(self.0.forward(as_cpu(x)?)?);
        match y.downcast::<Tensor<T, B>>() {
            Ok(y) => Ok(*y),
            Err(_) => xn::bail!("Mimi quantization needs f32 weights on the CPU"),
        }
    }
}
