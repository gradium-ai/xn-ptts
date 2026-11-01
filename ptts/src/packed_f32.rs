//! Opt-in adapters to xn's explicit, model-owned packed f32 CPU operator.
use std::any::{Any, TypeId};
use std::cell::RefCell;
use xn::cpu_backend::packed::{PackedF32 as Matrix, PackedF32Scratch};

#[derive(Default)]
struct Workspace {
    pack: PackedF32Scratch,
    lowered: Vec<f32>,
    product: Vec<f32>,
}

thread_local! { static WORK: RefCell<Workspace> = RefCell::new(Workspace::default()); }

pub(crate) struct PackedF32 {
    matrix: Matrix,
    n: usize,
    k: usize,
}

impl PackedF32 {
    pub(crate) fn load<Q: xn::BackendQ>(
        vb: &xn::nn::var_builder::Path<Q::B>,
        n: usize,
        k: usize,
    ) -> xn::Result<Option<Box<Self>>> {
        if std::env::var("PTTS_EXPERIMENT_PACKED_MIMI").as_deref() != Ok("1")
            || TypeId::of::<Q>() != TypeId::of::<xn::Unquantized<f32, xn::CpuDevice>>()
        {
            return Ok(None);
        }
        let weights: xn::Tensor<f32, Q::B> = vb.tensor("weight", (n, k))?;
        Self::from_weight(&weights, n, k)
    }

    pub(crate) fn from_weight<T: xn::WithDTypeF, B: xn::Backend>(
        weight: &xn::Tensor<T, B>,
        n: usize,
        k: usize,
    ) -> xn::Result<Option<Box<Self>>> {
        if TypeId::of::<T>() != TypeId::of::<f32>()
            || TypeId::of::<B>() != TypeId::of::<xn::CpuDevice>()
        {
            return Ok(None);
        }
        let storage = weight.storage()?;
        let values = (&*storage as &dyn Any)
            .downcast_ref::<Vec<f32>>()
            .ok_or_else(|| xn::Error::msg("expected CPU f32 weights"))?;
        let matrix = match Matrix::try_new(&values[..weight.elem_count()], n, k, None)? {
            Some(m) => m,
            None => return Ok(None),
        };
        tracing::debug!(n, k, bytes = matrix.packed_bytes(), "packed Mimi weights");
        Ok(Some(Box::new(Self { matrix, n, k })))
    }

    #[tracing::instrument(level="trace",target="ptts_profile",name="packed_mimi_linear",skip_all,fields(n=self.n,k=self.k))]
    pub(crate) fn forward<T: xn::WithDTypeF, B: xn::Backend>(
        &self,
        x: &xn::Tensor<T, B>,
    ) -> xn::Result<xn::Tensor<T, B>> {
        if TypeId::of::<T>() != TypeId::of::<f32>()
            || TypeId::of::<B>() != TypeId::of::<xn::CpuDevice>()
            || x.dims().last().copied() != Some(self.k)
        {
            xn::bail!("incompatible packed input");
        }
        let m = x.elem_count() / self.k;
        let mut shape = x.dims().to_vec();
        *shape.last_mut().unwrap() = self.n;
        let len =
            m.checked_mul(self.n).ok_or_else(|| xn::Error::msg("packed output size overflow"))?;
        // CPU-only adapter: initialize directly rather than dispatching a tensor fill
        // to the shared worker pool just before the matrix kernel overwrites it.
        let output = xn::Tensor::from_vec(vec![T::zero(); len], shape, x.device())?;
        {
            let input = x.storage()?;
            let input = (&*input as &dyn Any)
                .downcast_ref::<Vec<f32>>()
                .ok_or_else(|| xn::Error::msg("expected CPU f32 input"))?;
            let mut dst = output.storage_mut()?;
            let dst = (&mut *dst as &mut dyn Any)
                .downcast_mut::<Vec<f32>>()
                .ok_or_else(|| xn::Error::msg("expected CPU f32 output"))?;
            WORK.with_borrow_mut(|w| {
                self.matrix.matmul(&input[..x.elem_count()], m, &mut dst[..m * self.n], &mut w.pack)
            })?;
        }
        Ok(output)
    }

    /// Lower a dense, unpadded, single-group NCL convolution to matrix rows.
    pub(crate) fn conv1d<T: xn::WithDTypeF, B: xn::Backend>(
        &self,
        x: &xn::Tensor<T, B>,
        kernel: usize,
        stride: usize,
        dilation: usize,
    ) -> xn::Result<xn::Tensor<T, B>> {
        if TypeId::of::<T>() != TypeId::of::<f32>()
            || TypeId::of::<B>() != TypeId::of::<xn::CpuDevice>()
        {
            xn::bail!("incompatible packed convolution input");
        }
        let (batch, channels, length) = x.dims3()?;
        if stride == 0
            || dilation == 0
            || kernel == 0
            || channels.checked_mul(kernel) != Some(self.k)
        {
            xn::bail!("invalid packed convolution dimensions");
        }
        let effective = (kernel - 1)
            .checked_mul(dilation)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| xn::Error::msg("convolution extent overflow"))?;
        if length < effective {
            xn::bail!("convolution input is shorter than its kernel");
        }
        let width = (length - effective) / stride + 1;
        let rows =
            batch.checked_mul(width).ok_or_else(|| xn::Error::msg("convolution size overflow"))?;
        let lower_len =
            rows.checked_mul(self.k).ok_or_else(|| xn::Error::msg("convolution size overflow"))?;
        let product_len =
            rows.checked_mul(self.n).ok_or_else(|| xn::Error::msg("convolution size overflow"))?;
        let output =
            xn::Tensor::from_vec(vec![T::zero(); product_len], (batch, self.n, width), x.device())?;
        {
            let storage = x.storage()?;
            let input = (&*storage as &dyn Any)
                .downcast_ref::<Vec<f32>>()
                .ok_or_else(|| xn::Error::msg("expected CPU f32 input"))?;
            let mut dst = output.storage_mut()?;
            let dst = (&mut *dst as &mut dyn Any)
                .downcast_mut::<Vec<f32>>()
                .ok_or_else(|| xn::Error::msg("expected CPU f32 output"))?;
            WORK.with_borrow_mut(|w| -> xn::Result<()> {
                tracing::trace_span!(target:"ptts_profile","packed_conv_lower").in_scope(|| {
                    w.lowered.resize(lower_len, 0.);
                    for b in 0..batch {
                        for t in 0..width {
                            for c in 0..channels {
                                let src = (b * channels + c) * length + t * stride;
                                let dest = ((b * width + t) * channels + c) * kernel;
                                for tap in 0..kernel {
                                    w.lowered[dest + tap] = input[src + tap * dilation];
                                }
                            }
                        }
                    }
                });
                w.product.resize(product_len, 0.);
                tracing::trace_span!(target:"ptts_profile","packed_conv_gemm").in_scope(|| {
                    self.matrix.matmul(&w.lowered, rows, &mut w.product, &mut w.pack)
                })?;
                tracing::trace_span!(target:"ptts_profile","packed_conv_output").in_scope(|| {
                    for b in 0..batch {
                        for c in 0..self.n {
                            for t in 0..width {
                                dst[(b * self.n + c) * width + t] =
                                    w.product[(b * width + t) * self.n + c];
                            }
                        }
                    }
                });
                Ok(())
            })?;
        }
        Ok(output)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use xn::{CpuDevice, Tensor};
    #[test]
    fn lowered_convolution_matches_dense_backend_for_strides_and_dilations() -> xn::Result<()> {
        for (batch, channels, n, length, kernel, stride, dilation) in [
            (1, 3, 35, 23, 7, 1, 1),
            (2, 5, 37, 31, 3, 2, 2),
            (1, 3, 33, 1, 1, 1, 1),
            (2, 7, 32, 17, 1, 2, 1),
        ] {
            let weight = Tensor::from_vec(
                (0..n * channels * kernel).map(|i| (i as f32 * 0.13).sin() * 0.1).collect(),
                (n, channels, kernel),
                &CpuDevice,
            )?;
            let Some(packed) = PackedF32::from_weight(&weight, n, channels * kernel)? else {
                return Ok(());
            };
            let input = Tensor::from_vec(
                (0..batch * channels * length).map(|i| (i as f32 * 0.07).cos()).collect(),
                (batch, channels, length),
                &CpuDevice,
            )?;
            let expected = input.conv1d(&weight, None, stride, 0, dilation, 1)?;
            let actual = packed.conv1d(&input, kernel, stride, dilation)?;
            assert_eq!(expected.dims(), actual.dims());
            for (a, b) in expected.to_vec()?.into_iter().zip(actual.to_vec()?) {
                assert!(b.is_finite() && (a - b).abs() < 1e-5, "{a} != {b}");
            }
        }
        Ok(())
    }
    #[test]
    fn linear_matches_reference_across_row_counts_and_rejects_other_dtypes() -> xn::Result<()> {
        let weight = Tensor::from_vec(
            (0..19 * 37).map(|i| (i as f32 * 0.13).sin() * 0.1).collect(),
            (19, 37),
            &CpuDevice,
        )?;
        assert!(PackedF32::from_weight(&weight.to::<half::f16>()?, 19, 37)?.is_none());
        let Some(packed) = PackedF32::from_weight(&weight, 19, 37)? else {
            return Ok(());
        };
        for rows in [1, 2, 8, 17, 1] {
            let x = Tensor::from_vec(
                (0..rows * 37).map(|i| (i as f32 * 0.07).cos()).collect(),
                (rows, 37),
                &CpuDevice,
            )?;
            let expected = x.matmul_t(&weight)?.to_vec()?;
            let actual = packed.forward(&x)?.to_vec()?;
            for (a, b) in expected.into_iter().zip(actual) {
                assert!(b.is_finite() && (a - b).abs() < 1e-5, "{a} != {b}");
            }
        }
        assert!(packed.forward(&Tensor::<f32, _>::zeros((2, 36), &CpuDevice)?).is_err());
        Ok(())
    }
}
