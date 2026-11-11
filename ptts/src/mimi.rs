//! The Mimi neural audio codec.
//!
//! [`MimiDecoder`] turns the flow LM's latents into PCM at the checkpoint's sample rate, one
//! frame at a time, keeping its streaming state in [`MimiDecoderState`]. [`MimiEncoder`] goes
//! the other way, to embed a voice prompt from audio or, frame by frame with its state in
//! [`MimiEncoderState`], a speech-to-speech checkpoint's input.

use crate::conv::pad_for_conv1d;
use crate::conv::{StreamingConv1dState, StreamingConvTr1dState};
use crate::dummy_quantizer::DummyQuantizer;
use crate::resample::{ConvDownsample1d, ConvTrUpsample1d};
use crate::seanet::{SEANetDecoder, SEANetDecoderState, SEANetEncoder, SEANetEncoderState};
use crate::transformer::{ProjectedTransformer, StreamingTransformerState};
use xn::nn::var_builder::Path;
use xn::{Backend, BackendQ, Result, Tensor, Unquantized, WithDTypeF};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MimiConfig {
    pub channels: usize,
    pub sample_rate: usize,
    pub frame_rate: f64,
    pub dimension: usize,
    pub quantizer_dimension: usize,
    pub quantizer_output_dimension: usize,
    pub n_filters: usize,
    pub n_residual_layers: usize,
    pub ratios: Vec<usize>,
    pub kernel_size: usize,
    pub last_kernel_size: usize,
    pub residual_kernel_size: usize,
    pub dilation_base: usize,
    pub compress: usize,
    // Transformer params
    pub transformer_d_model: usize,
    pub transformer_num_heads: usize,
    pub transformer_num_layers: usize,
    pub transformer_layer_scale: f64,
    pub transformer_context: usize,
    pub transformer_max_period: f32,
    pub transformer_dim_feedforward: usize,
    #[serde(default)]
    pub downsample_channel_wise: bool,
    /// Width of the SEANet decoder and of the decoder transformer's output, when it differs from
    /// `dimension` (small decoders distilled from a larger codec). Unset: `dimension`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoder_dimension: Option<usize>,
    /// Width of the encoder's latents, which its downsample projects to; `dimension` when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inner_dim: Option<usize>,
    /// FFN activation of the encoder and decoder transformers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gelu: Option<crate::transformer::Gelu>,
}

impl MimiConfig {
    /// Width of the latents [`MimiEncoder::encode_to_latent`] returns.
    pub fn encoder_latent_dim(&self) -> usize {
        self.inner_dim.unwrap_or(self.dimension)
    }
}

pub struct MimiEncoder<Q: BackendQ> {
    encoder: SEANetEncoder<Q::T, Q::B>,
    encoder_transformer: ProjectedTransformer<Q>,
    downsample: Option<ConvDownsample1d<Q::T, Q::B>>,
    frame_size: usize,
}

pub struct MimiDecoder<Q: BackendQ> {
    decoder: SEANetDecoder<Q::T, Q::B>,
    decoder_transformer: ProjectedTransformer<Q>,
    upsample: Option<ConvTrUpsample1d<Q::T, Q::B>>,
    pub(crate) quantizer: DummyQuantizer<Q::T, Q::B>,
    pub sample_rate: usize,
    pub frame_rate: f64,
}

#[derive(Debug, Clone)]
pub struct MimiEncoderState<T: WithDTypeF, B: Backend> {
    encoder_state: SEANetEncoderState<T, B>,
    encoder_transformer_state: StreamingTransformerState<T, B>,
    downsample_state: Option<StreamingConv1dState<T, B>>,
}

#[derive(Debug, Clone)]
pub struct MimiDecoderState<T: WithDTypeF, B: Backend> {
    decoder_state: SEANetDecoderState<T, B>,
    decoder_transformer_state: StreamingTransformerState<T, B>,
    upsample_state: Option<StreamingConvTr1dState<T, B>>,
}

impl<Q: BackendQ> MimiEncoder<Q> {
    pub fn load(vb: &Path<Q::B>, cfg: &MimiConfig) -> Result<Self> {
        let pad_mode = crate::conv::PadMode::Constant;

        let encoder = SEANetEncoder::load(
            &vb.pp("encoder"),
            cfg.channels,
            cfg.dimension,
            cfg.n_filters,
            cfg.n_residual_layers,
            &cfg.ratios,
            cfg.kernel_size,
            cfg.last_kernel_size,
            cfg.residual_kernel_size,
            cfg.dilation_base,
            pad_mode,
            cfg.compress,
        )?;

        let output_dimensions = vec![cfg.dimension];
        let mut encoder_transformer = ProjectedTransformer::load(
            &vb.pp("encoder_transformer"),
            cfg.dimension,
            &output_dimensions,
            cfg.transformer_d_model,
            cfg.transformer_num_heads,
            cfg.transformer_num_layers,
            Some(cfg.transformer_layer_scale),
            cfg.transformer_context,
            cfg.transformer_max_period,
            cfg.transformer_dim_feedforward,
        )?;
        encoder_transformer.transformer.set_gelu(cfg.gelu.unwrap_or_default());

        let hop_length: usize = cfg.ratios.iter().product();
        let encoder_frame_rate = cfg.sample_rate as f64 / hop_length as f64;

        let downsample = if (encoder_frame_rate - cfg.frame_rate).abs() > 0.01 {
            let downsample_stride = (encoder_frame_rate / cfg.frame_rate) as usize;
            let ds = ConvDownsample1d::load(
                &vb.pp("downsample"),
                downsample_stride,
                cfg.dimension,
                cfg.encoder_latent_dim(),
                cfg.downsample_channel_wise,
            )?;
            Some(ds)
        } else {
            None
        };
        let frame_size = (cfg.sample_rate as f64 / cfg.frame_rate).round() as usize;
        Ok(Self { encoder, encoder_transformer, downsample, frame_size })
    }

    /// Audio samples per latent frame.
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    pub fn init_state(&self, batch_size: usize) -> Result<MimiEncoderState<Q::T, Q::B>> {
        let downsample_state = self.downsample.as_ref().map(|ds| ds.init_state(batch_size));
        Ok(MimiEncoderState {
            encoder_state: self.encoder.init_state(batch_size)?,
            // `sequence_length` reaches only the flow-LM attention kind, see
            // `MimiDecoder::init_state`.
            encoder_transformer_state: self.encoder_transformer.init_state(batch_size, 0)?,
            downsample_state: downsample_state.transpose()?,
        })
    }

    /// Encode audio to latent (non-streaming). Returns [B, C, T'].
    pub fn encode_to_latent(&self, x: &Tensor<Q::T, Q::B>) -> Result<Tensor<Q::T, Q::B>> {
        let x = pad_for_conv1d(x, self.frame_size, self.frame_size)?;
        let mut state = self.init_state(x.dim(0usize)?)?;
        self.encode_step(&x, &mut state)
    }

    /// Encode audio to latent (streaming). Input: [B, C, T], with T a multiple of
    /// [`Self::frame_size`]; returns [B, C', T / frame_size].
    pub fn encode_step(
        &self,
        x: &Tensor<Q::T, Q::B>,
        state: &mut MimiEncoderState<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        let emb = self.encoder.forward(x, &mut state.encoder_state)?;
        let mut outs =
            self.encoder_transformer.forward(&emb, &mut state.encoder_transformer_state)?;
        let emb = outs.swap_remove(0);
        // Downsample to frame rate
        match (&self.downsample, &mut state.downsample_state) {
            (Some(ds), Some(ds_state)) => ds.forward(&emb, ds_state),
            _ => Ok(emb),
        }
    }
}

impl<B: Backend> MimiDecoder<Unquantized<f32, B>> {
    /// Run the decoder transformer and the two widest SEANet convolutions in q8_0; see
    /// [`crate::tts_model::TTSModel::quantize_mimi`]. Every weight is quantized before any is
    /// switched over, so on error the decoder is left as it was. CPU only.
    pub fn quantize_q8(&mut self) -> Result<()> {
        let transformer = self.decoder_transformer.transformer.prepare_q8()?;
        let convs = self.decoder.prepare_q8()?;
        self.decoder_transformer.transformer.apply_q8(transformer);
        self.decoder.apply_q8(convs);
        Ok(())
    }
}

impl<Q: BackendQ> MimiDecoder<Q> {
    pub fn load(vb: &Path<Q::B>, cfg: &MimiConfig) -> Result<Self> {
        let pad_mode = crate::conv::PadMode::Constant;
        let decoder_dimension = cfg.decoder_dimension.unwrap_or(cfg.dimension);

        let decoder = SEANetDecoder::load(
            &vb.pp("decoder"),
            cfg.channels,
            decoder_dimension,
            cfg.n_filters,
            cfg.n_residual_layers,
            &cfg.ratios,
            cfg.kernel_size,
            cfg.last_kernel_size,
            cfg.residual_kernel_size,
            cfg.dilation_base,
            pad_mode,
            cfg.compress,
        )?;

        let quantizer = DummyQuantizer::load(
            &vb.pp("quantizer"),
            cfg.quantizer_dimension,
            cfg.quantizer_output_dimension,
        )?;
        let output_dimensions = vec![decoder_dimension];
        let mut decoder_transformer = ProjectedTransformer::load(
            &vb.pp("decoder_transformer"),
            cfg.dimension,
            &output_dimensions,
            cfg.transformer_d_model,
            cfg.transformer_num_heads,
            cfg.transformer_num_layers,
            Some(cfg.transformer_layer_scale),
            cfg.transformer_context,
            cfg.transformer_max_period,
            cfg.transformer_dim_feedforward,
        )?;
        decoder_transformer.transformer.set_gelu(cfg.gelu.unwrap_or_default());
        let hop_length: usize = cfg.ratios.iter().product();
        let encoder_frame_rate = cfg.sample_rate as f64 / hop_length as f64;

        let upsample = if (encoder_frame_rate - cfg.frame_rate).abs() > 0.01 {
            let downsample_stride = (encoder_frame_rate / cfg.frame_rate) as usize;
            let us = ConvTrUpsample1d::load(&vb.pp("upsample"), downsample_stride, cfg.dimension)?;
            Some(us)
        } else {
            None
        };
        Ok(Self {
            decoder,
            decoder_transformer,
            upsample,
            quantizer,
            sample_rate: cfg.sample_rate,
            frame_rate: cfg.frame_rate,
        })
    }

    pub fn init_state(
        &self,
        batch_size: usize,
        sequence_length: usize,
    ) -> Result<MimiDecoderState<Q::T, Q::B>> {
        let upsample_state = match &self.upsample {
            Some(us) => Some(us.init_state(batch_size)?),
            None => None,
        };
        let s = MimiDecoderState {
            decoder_state: self.decoder.init_state(batch_size)?,
            decoder_transformer_state: self
                .decoder_transformer
                .init_state(batch_size, sequence_length)?,
            upsample_state,
        };
        Ok(s)
    }

    /// Decode from latent to audio (streaming). Input: [B, C, T'].
    pub fn decode_from_latent_step(
        &self,
        latent: &Tensor<Q::T, Q::B>,
        state: &mut MimiDecoderState<Q::T, Q::B>,
    ) -> Result<Tensor<Q::T, Q::B>> {
        // Upsample to encoder frame rate
        let emb = match (&self.upsample, &mut state.upsample_state) {
            (Some(us), Some(us_state)) => us.forward(latent, us_state)?,
            _ => latent.clone(),
        };

        let outs = self.decoder_transformer.forward(&emb, &mut state.decoder_transformer_state)?;
        self.decoder.forward(&outs[0], &mut state.decoder_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xn::CpuDevice;
    use xn::nn::VB;

    type F32 = Unquantized<f32, CpuDevice>;

    /// An encoder with random weights: 8 samples per frame, two SEANet steps per frame, a
    /// transformer context of 6 steps, and 6-wide latents.
    fn encoder() -> MimiEncoder<F32> {
        use rand::{Rng, SeedableRng};
        let cfg = MimiConfig {
            channels: 1,
            sample_rate: 16,
            frame_rate: 2.0,
            dimension: 16,
            quantizer_dimension: 6,
            quantizer_output_dimension: 16,
            n_filters: 4,
            n_residual_layers: 1,
            ratios: vec![2, 2],
            kernel_size: 3,
            last_kernel_size: 3,
            residual_kernel_size: 3,
            dilation_base: 2,
            compress: 2,
            transformer_d_model: 16,
            transformer_num_heads: 2,
            transformer_num_layers: 1,
            transformer_layer_scale: 0.01,
            transformer_context: 6,
            transformer_max_period: 10_000.,
            transformer_dim_feedforward: 32,
            downsample_channel_wise: false,
            decoder_dimension: None,
            inner_dim: Some(6),
            gelu: None,
        };
        let conv = |name: &str, out: usize, inp: usize, k: usize| {
            vec![(format!("{name}.weight"), vec![out, inp, k]), (format!("{name}.bias"), vec![out])]
        };
        let t = "encoder_transformer.transformer.layers.0";
        let mut tensors = [
            conv("encoder.model.0.conv", 4, 1, 3),
            conv("encoder.model.1.block.1.conv", 2, 4, 3),
            conv("encoder.model.1.block.3.conv", 4, 2, 1),
            conv("encoder.model.3.conv", 8, 4, 4),
            conv("encoder.model.4.block.1.conv", 4, 8, 3),
            conv("encoder.model.4.block.3.conv", 8, 4, 1),
            conv("encoder.model.6.conv", 16, 8, 4),
            conv("encoder.model.8.conv", 16, 16, 3),
        ]
        .concat();
        for (name, shape) in [
            ("self_attn.in_proj.weight", vec![48, 16]),
            ("self_attn.out_proj.weight", vec![16, 16]),
            ("linear1.weight", vec![32, 16]),
            ("linear2.weight", vec![16, 32]),
            ("norm1.weight", vec![16]),
            ("norm1.bias", vec![16]),
            ("norm2.weight", vec![16]),
            ("norm2.bias", vec![16]),
            ("layer_scale_1.scale", vec![16]),
            ("layer_scale_2.scale", vec![16]),
        ] {
            tensors.push((format!("{t}.{name}"), shape));
        }
        tensors.push(("downsample.conv.conv.weight".into(), vec![6, 16, 4]));

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let (mut header, mut data) = (serde_json::Map::new(), Vec::new());
        for (name, shape) in tensors {
            let start = data.len();
            for _ in 0..shape.iter().product::<usize>() {
                data.extend(rng.random_range(-0.5f32..0.5).to_le_bytes());
            }
            header.insert(
                name,
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        let header = serde_json::to_vec(&header).unwrap();
        let bytes = [(header.len() as u64).to_le_bytes().as_slice(), &header, &data].concat();
        let vb = VB::from_bytes(vec![bytes], CpuDevice).unwrap().root();
        MimiEncoder::load(&vb, &cfg).unwrap()
    }

    #[test]
    fn encoding_frame_by_frame_matches_encoding_the_whole_input() {
        let enc = encoder();
        let frames = 20;
        let pcm: Vec<f32> = (0..frames * 8).map(|i| (i as f32 * 0.37).sin()).collect();
        let whole = Tensor::from_vec(pcm.clone(), (1, 1, pcm.len()), &CpuDevice).unwrap();
        let want: Vec<f32> = enc.encode_to_latent(&whole).unwrap().to_vec().unwrap();
        assert_eq!(want.len(), 6 * frames);
        for per in [1, 3] {
            let mut state = enc.init_state(1).unwrap();
            let pieces: Vec<Tensor<f32, CpuDevice>> = pcm
                .chunks(8 * per)
                .map(|c| {
                    let c = Tensor::from_vec(c.to_vec(), (1, 1, c.len()), &CpuDevice).unwrap();
                    enc.encode_step(&c, &mut state).unwrap()
                })
                .collect();
            let got: Vec<f32> =
                Tensor::cat(&pieces.iter().collect::<Vec<_>>(), 2).unwrap().to_vec().unwrap();
            let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(err < 1e-4, "{per} frames at a time: {err}");
        }
    }
}
