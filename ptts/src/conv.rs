use crate::q8::Q8Linear;
use xn::nn::var_builder::Path;
use xn::{Backend, Result, Tensor, WithDTypeF};

/// Pad input so that a convolution covers the full input.
pub fn pad_for_conv1d<T: WithDTypeF, B: Backend>(
    x: &Tensor<T, B>,
    kernel_size: usize,
    stride: usize,
) -> Result<Tensor<T, B>> {
    let length = x.dim(2usize)?;
    let n_frames = (length as f64 - kernel_size as f64) / stride as f64 + 1.0;
    let ideal_length = (n_frames.ceil() as usize - 1) * stride + kernel_size;
    let extra = ideal_length.saturating_sub(length);
    if extra > 0 { x.pad_with_zeros(2usize, 0, extra) } else { Ok(x.clone()) }
}

/// Conv1d wrapper with weight and optional bias.
pub struct Conv1d<T: WithDTypeF, B: Backend> {
    weight: Tensor<T, B>,
    /// The weight as `[out, in * kernel]` in q8_0, used instead when set.
    pub(crate) q8: Option<Q8Linear>,
    bias: Option<Tensor<T, B>>,
    pub stride: usize,
    pub dilation: usize,
    pub kernel_size: usize,
    pub in_channels: usize,
    pub groups: usize,
}

impl<T: WithDTypeF, B: Backend> Conv1d<T, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Path<B>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let weight = vb.tensor("weight", (out_channels, in_channels / groups, kernel_size))?;
        let bias = if bias { Some(vb.tensor("bias", (out_channels,))?) } else { None };
        Ok(Self { weight, q8: None, bias, stride, dilation, kernel_size, in_channels, groups })
    }

    /// The q8_0 weight that runs this convolution as a matmul over its input windows, for
    /// `self.q8`. CPU only.
    pub(crate) fn prepare_q8(&self) -> Result<Q8Linear> {
        let k = self.in_channels * self.kernel_size;
        if self.stride != 1 || self.groups != 1 || !k.is_multiple_of(32) {
            xn::bail!(
                "cannot quantize a conv with stride {}, {} groups and {k} inputs per output",
                self.stride,
                self.groups
            )
        }
        Q8Linear::new(&self.weight.reshape(((), k))?)
    }

    pub fn forward(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        if let Some(q8) = &self.q8 {
            let (b, c, l) = x.dims3()?;
            let (k, dil) = (self.kernel_size, self.dilation);
            let Some(t) = l.checked_sub((k - 1) * dil).filter(|t| *t > 0) else {
                xn::bail!("conv input of length {l} is shorter than its kernel")
            };
            let taps = (0..k)
                .map(|j| x.narrow(2, j * dil..j * dil + t)?.contiguous())
                .collect::<Result<Vec<_>>>()?;
            let taps: Vec<_> = taps.iter().collect();
            // [b, c, t, k] -> [b, t, c, k]: one row per output step, in the weight's (c, k) order.
            let windows =
                Tensor::stack(&taps, 3)?.transpose(1, 2)?.contiguous()?.reshape((b * t, c * k))?;
            let y = q8.forward(&windows)?.reshape((b, t, ()))?.transpose(1, 2)?.contiguous()?;
            return match &self.bias {
                Some(bias) => y.broadcast_add(&bias.reshape((1, (), 1))?),
                None => Ok(y),
            };
        }
        x.conv1d(&self.weight, self.bias.as_ref(), self.stride, 0, self.dilation, self.groups)
    }
}

/// ConvTranspose1d wrapper.
pub struct ConvTranspose1d<T: WithDTypeF, B: Backend> {
    weight: Tensor<T, B>,
    /// The weight as `[out * kernel, in]` in q8_0, used instead when set.
    pub(crate) q8: Option<Q8Linear>,
    pub bias: Option<Tensor<T, B>>,
    pub stride: usize,
    pub kernel_size: usize,
    pub out_channels: usize,
    pub groups: usize,
}

impl<T: WithDTypeF, B: Backend> ConvTranspose1d<T, B> {
    pub fn load(
        vb: &Path<B>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let weight = vb.tensor("weight", (in_channels, out_channels / groups, kernel_size))?;
        let bias = if bias { Some(vb.tensor("bias", (out_channels,))?) } else { None };
        Ok(Self { weight, q8: None, bias, stride, kernel_size, out_channels, groups })
    }

    /// The q8_0 weight that runs this transposed convolution as a matmul and an overlap-add,
    /// for `self.q8`. Only the `kernel == 2 * stride` shape SEANet uses is supported. CPU only.
    pub(crate) fn prepare_q8(&self) -> Result<Q8Linear> {
        let c = self.weight.dims3()?.0;
        if self.groups != 1 || self.kernel_size != 2 * self.stride || !c.is_multiple_of(32) {
            xn::bail!(
                "cannot quantize a transposed conv with kernel {}, stride {}, {} groups and {c} inputs",
                self.kernel_size,
                self.stride,
                self.groups
            )
        }
        Q8Linear::new(&self.weight.reshape((c, ()))?.transpose(0, 1)?.contiguous()?)
    }

    pub fn forward(&self, x: &Tensor<T, B>) -> Result<Tensor<T, B>> {
        if let Some(q8) = &self.q8 {
            let (b, c, t) = x.dims3()?;
            let (out, s) = (self.out_channels, self.stride);
            let z = q8.forward(&x.transpose(1, 2)?.contiguous()?.reshape((b * t, c))?)?;
            // Each input step yields 2 * stride outputs: the first half lands on its own
            // stride-long block, the second half on the next one.
            let z = z.reshape((b, t, out, 2, s))?;
            let half = |h: usize| {
                z.narrow(3, h..h + 1)?.permute(&[0, 2, 1, 3, 4])?.contiguous()?.reshape((
                    b,
                    out,
                    t * s,
                ))
            };
            let y = half(0)?.pad_with_zeros(2, 0, s)?.add(&half(1)?.pad_with_zeros(2, s, 0)?)?;
            return match &self.bias {
                Some(bias) => y.broadcast_add(&bias.reshape((1, (), 1))?),
                None => Ok(y),
            };
        }
        x.conv_transpose1d(&self.weight, self.bias.as_ref(), self.stride, 0, 0, self.groups)
    }
}

/// Streaming state for StreamingConv1d.
#[derive(Debug, Clone)]
pub struct StreamingConv1dState<T: WithDTypeF, B: Backend> {
    pub previous: Tensor<T, B>,
    pub first: bool,
}

/// Streaming Conv1d with causal padding.
pub struct StreamingConv1d<T: WithDTypeF, B: Backend> {
    pub conv: Conv1d<T, B>,
    pad_mode: PadMode,
}

#[derive(Clone, Copy)]
pub enum PadMode {
    Constant,
    Replicate,
}

impl<T: WithDTypeF, B: Backend> StreamingConv1d<T, B> {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &Path<B>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        pad_mode: PadMode,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let conv = Conv1d::load(
            &vb.pp("conv"),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation,
            groups,
            bias,
        )?;
        Ok(Self { conv, pad_mode })
    }

    fn effective_kernel_size(&self) -> usize {
        (self.conv.kernel_size - 1) * self.conv.dilation + 1
    }

    pub fn init_state(&self, batch_size: usize) -> Result<StreamingConv1dState<T, B>> {
        let kernel = self.effective_kernel_size();
        let prev_len = kernel - self.conv.stride;
        let previous = Tensor::zeros(
            (batch_size, self.conv.in_channels, prev_len),
            self.conv.weight.device(),
        )?;
        Ok(StreamingConv1dState { previous, first: true })
    }

    pub fn forward(
        &self,
        x: &Tensor<T, B>,
        state: &mut StreamingConv1dState<T, B>,
    ) -> Result<Tensor<T, B>> {
        let tp = state.previous.dim(2usize)?;

        // On first call with replicate padding, fill previous with first sample
        if tp > 0 && matches!(self.pad_mode, PadMode::Replicate) && state.first {
            let init = x.narrow(2, 0..1)?.contiguous()?;
            // Broadcast init to fill previous
            let mut fills = Vec::with_capacity(tp);
            for _ in 0..tp {
                fills.push(&init);
            }
            // Actually just repeat using cat
            state.previous = if tp == 1 {
                init
            } else {
                let refs: Vec<&Tensor<T, B>> = (0..tp).map(|_| &init).collect();
                Tensor::cat(&refs, 2)?
            };
        }

        // Prepend previous state
        let x = if tp > 0 { Tensor::cat(&[&state.previous, x], 2)? } else { x.clone() };

        // Run convolution
        let y = self.conv.forward(&x)?;

        // Update state
        if tp > 0 {
            let xlen = x.dim(2usize)?;
            state.previous = x.narrow(2, xlen - tp..xlen)?.contiguous()?;
            if matches!(self.pad_mode, PadMode::Replicate) {
                state.first = false;
            }
        }

        Ok(y)
    }
}

/// Streaming state for StreamingConvTranspose1d.
#[derive(Debug, Clone)]
pub struct StreamingConvTr1dState<T: WithDTypeF, B: Backend> {
    pub partial: Tensor<T, B>,
}

/// Streaming ConvTranspose1d.
pub struct StreamingConvTranspose1d<T: WithDTypeF, B: Backend> {
    pub convtr: ConvTranspose1d<T, B>,
}

impl<T: WithDTypeF, B: Backend> StreamingConvTranspose1d<T, B> {
    pub fn load(
        vb: &Path<B>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let convtr = ConvTranspose1d::load(
            &vb.pp("convtr"),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups,
            bias,
        )?;
        Ok(Self { convtr })
    }

    pub fn init_state(&self, batch_size: usize) -> Result<StreamingConvTr1dState<T, B>> {
        let pt = self.convtr.kernel_size - self.convtr.stride;
        let partial =
            Tensor::zeros((batch_size, self.convtr.out_channels, pt), self.convtr.weight.device())?;
        Ok(StreamingConvTr1dState { partial })
    }

    pub fn forward(
        &self,
        x: &Tensor<T, B>,
        state: &mut StreamingConvTr1dState<T, B>,
    ) -> Result<Tensor<T, B>> {
        let mut y = self.convtr.forward(x)?;
        let pt = state.partial.dim(2usize)?;

        if pt > 0 {
            // Add overlap from previous
            let y_start = y.narrow(2, 0..pt)?.contiguous()?;
            let y_start = y_start.add(&state.partial)?;
            y.slice_set(&y_start, 2usize, 0)?;

            // Save new partial (subtract bias if present)
            let y_len = y.dim(2usize)?;
            let mut for_partial = y.narrow(2, y_len - pt..y_len)?.contiguous()?;
            if let Some(bias) = &self.convtr.bias {
                let bias = bias.reshape((1, bias.elem_count(), 1))?;
                for_partial = for_partial.broadcast_sub(&bias)?;
            }
            state.partial = for_partial;

            // Return without the partial tail
            y = y.narrow(2, 0..y_len - pt)?.contiguous()?;
        }

        Ok(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xn::CpuDevice;

    fn tensor(shape: &[usize], seed: u64) -> Tensor<f32, CpuDevice> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n = shape.iter().product();
        let v: Vec<f32> = (0..n).map(|_| rng.random_range(-1.0f32..1.0)).collect();
        Tensor::from_vec(v, shape.to_vec(), &CpuDevice).unwrap()
    }

    fn rel_err(a: &Tensor<f32, CpuDevice>, b: &Tensor<f32, CpuDevice>) -> f32 {
        assert_eq!(a.dims(), b.dims());
        let (a, b) = (a.to_vec().unwrap(), b.to_vec().unwrap());
        let num: f32 = a.iter().zip(&b).map(|(x, y)| (x - y) * (x - y)).sum();
        let den: f32 = b.iter().map(|y| y * y).sum();
        (num / den).sqrt()
    }

    #[test]
    fn a_q8_conv_matches_the_f32_one() {
        for (c_in, c_out, k, dil) in [(64, 48, 7, 1), (32, 16, 3, 2), (32, 8, 1, 1)] {
            let mut conv = Conv1d {
                weight: tensor(&[c_out, c_in, k], 1),
                q8: None,
                bias: Some(tensor(&[c_out], 2)),
                stride: 1,
                dilation: dil,
                kernel_size: k,
                in_channels: c_in,
                groups: 1,
            };
            let x = tensor(&[1, c_in, 20 + (k - 1) * dil], 3);
            let want = conv.forward(&x).unwrap();
            conv.q8 = Some(conv.prepare_q8().unwrap());
            let got = conv.forward(&x).unwrap();
            let err = rel_err(&got, &want);
            assert!(err < 0.02, "conv {c_in}->{c_out} k{k} d{dil}: {err}");
        }
    }

    #[test]
    fn a_q8_transposed_conv_matches_the_f32_one() {
        for (c_in, c_out, stride) in [(64, 32, 4), (32, 16, 6)] {
            let k = 2 * stride;
            let mut conv = ConvTranspose1d {
                weight: tensor(&[c_in, c_out, k], 4),
                q8: None,
                bias: Some(tensor(&[c_out], 5)),
                stride,
                kernel_size: k,
                out_channels: c_out,
                groups: 1,
            };
            let x = tensor(&[1, c_in, 9], 6);
            let want = conv.forward(&x).unwrap();
            conv.q8 = Some(conv.prepare_q8().unwrap());
            let got = conv.forward(&x).unwrap();
            let err = rel_err(&got, &want);
            assert!(err < 0.02, "convtr {c_in}->{c_out} s{stride}: {err}");
        }
    }

    #[test]
    fn a_streamed_q8_transposed_conv_matches_the_f32_one() {
        // The overlap carried between calls is stored without the bias, so check that the q8
        // output, whose bias is added at the end, splits across calls the same way.
        let (c_in, c_out, stride) = (32, 16, 4);
        let streaming = || StreamingConvTranspose1d {
            convtr: ConvTranspose1d {
                weight: tensor(&[c_in, c_out, 2 * stride], 8),
                q8: None,
                bias: Some(tensor(&[c_out], 9)),
                stride,
                kernel_size: 2 * stride,
                out_channels: c_out,
                groups: 1,
            },
        };
        let (f32_conv, mut q8_conv) = (streaming(), streaming());
        q8_conv.convtr.q8 = Some(q8_conv.convtr.prepare_q8().unwrap());
        let run = |conv: &StreamingConvTranspose1d<f32, CpuDevice>| {
            let mut state = conv.init_state(1).unwrap();
            let chunks: Vec<_> = (0..3)
                .map(|i| {
                    conv.forward(&tensor(&[1, c_in, 2 + i], 10 + i as u64), &mut state).unwrap()
                })
                .collect();
            Tensor::cat(&chunks.iter().collect::<Vec<_>>(), 2).unwrap()
        };
        let err = rel_err(&run(&q8_conv), &run(&f32_conv));
        assert!(err < 0.02, "{err}");
    }

    #[test]
    fn shapes_the_q8_path_cannot_run_are_refused() {
        let conv = ConvTranspose1d {
            weight: tensor(&[32, 16, 6], 7),
            q8: None,
            bias: None,
            stride: 2,
            kernel_size: 6,
            out_channels: 16,
            groups: 1,
        };
        assert!(conv.prepare_q8().is_err());
    }
}
