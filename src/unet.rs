use candle_core::{Result, Tensor};
use candle_nn::{
    batch_norm, conv2d, conv2d_no_bias, conv_transpose2d_no_bias, ops, BatchNorm, BatchNormConfig,
    Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig, Module, VarBuilder,
};

/// ConvBlock is composed of:
/// two 3x3 conv2d -> BatchNorm -> ReLU
struct ConvBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    bnorm1: BatchNorm,
    bnorm2: BatchNorm,
}

impl ConvBlock {
    fn new(in_chans: usize, out_chans: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let mut conv_cfg = Conv2dConfig::default();
        // We use padding, contrary to the original paper, for simplicity.
        conv_cfg.padding = 1;
        let conv1 = conv2d_no_bias(
            in_chans,
            out_chans,
            kernel_size,
            conv_cfg.clone(),
            vb.pp("conv1"),
        )?;
        let bnorm1 = batch_norm(out_chans, BatchNormConfig::default(), vb.pp("bnorm1"))?;
        let conv2 = conv2d_no_bias(out_chans, out_chans, kernel_size, conv_cfg, vb.pp("conv2"))?;
        let bnorm2 = batch_norm(out_chans, BatchNormConfig::default(), vb.pp("bnorm2"))?;
        Ok(Self {
            conv1,
            conv2,
            bnorm1,
            bnorm2,
        })
    }

    fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let xs = self.conv1.forward(&xs)?;
        let xs = xs.apply_t(&self.bnorm1, train)?.relu()?;
        let xs = self.conv2.forward(&xs)?;
        xs.apply_t(&self.bnorm2, train)?.relu()
    }
}

/// UpBlock is composed of:
/// - two 3x3 conv, each followed by ReLU
/// - a 2x2 up-conv
struct UpBlock {
    upconv: ConvTranspose2d,
    convblk: ConvBlock,
}

impl UpBlock {
    fn new(
        in_chans: usize,
        out_chans: usize,
        upconv_kern: usize,
        upconv_stride: usize,
        conv_kern: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut upconv_cfg = ConvTranspose2dConfig::default();
        upconv_cfg.stride = upconv_stride;
        let upconv = conv_transpose2d_no_bias(
            in_chans,
            out_chans,
            upconv_kern,
            upconv_cfg,
            vb.pp("upconv"),
        )?;
        let convblk = ConvBlock::new(out_chans * 2, out_chans, conv_kern, vb.pp("convblk"))?;
        Ok(Self { upconv, convblk })
    }

    fn forward(&self, xs: &Tensor, skip_con: &Tensor, train: bool) -> Result<Tensor> {
        let xs = self.upconv.forward(xs)?;
        let xs_cat = Tensor::cat(&[&xs, skip_con], 1)?;
        self.convblk.forward(&xs_cat, train)
    }
}

pub struct Unet {
    contractive_path: Vec<ConvBlock>,
    bottleneck: ConvBlock,
    expansive_path: Vec<UpBlock>,
    final_conv: Conv2d,
}

impl Unet {
    /// Create a new Unet model
    /// * `in_dim`: number of channels of the input, eg 1 for grayscale, 3 for RGB image.
    /// * `blk1_feat`: number of features (ie channels) of the first conv block, following blocks
    /// will double the feature size of the previous block.
    /// * `out_dim`: number of channels of the model's output, eg 1 for a segmentaiton mask.
    /// * `vb`: VarBuilder
    pub fn new(in_dim: usize, blk1_feat: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let mut cont_path: Vec<ConvBlock> = Vec::new();
        let mut curr_in = in_dim;
        let mut curr_feat = blk1_feat;
        for i in 0..4 {
            let convblk = ConvBlock::new(curr_in, curr_feat, 3, vb.pp(format!("downblk{}", i)))?;
            cont_path.push(convblk);
            curr_in = curr_feat;
            curr_feat *= 2;
        }

        let bottleneck = ConvBlock::new(curr_in, curr_feat, 3, vb.pp("bottleneck"))?;
        curr_in = curr_feat;
        curr_feat /= 2;

        let mut exp_path: Vec<UpBlock> = Vec::new();
        for i in 0..4 {
            let upblk = UpBlock::new(curr_in, curr_feat, 2, 2, 3, vb.pp(format!("upblk{}", i)))?;
            exp_path.push(upblk);
            curr_in = curr_feat;
            curr_feat /= 2;
        }

        let conv_cfg = Conv2dConfig::default();
        let final_conv = conv2d(curr_in, out_dim, 1, conv_cfg, vb.pp("final_conv"))?;

        Ok(Self {
            contractive_path: cont_path,
            bottleneck,
            expansive_path: exp_path,
            final_conv,
        })
    }
    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let mut block_output: Vec<Tensor> = Vec::new();
        let mut pool_output: Vec<Tensor> = Vec::new();
        pool_output.push(xs.clone());

        for (i, block) in self.contractive_path.iter().enumerate() {
            let out = block.forward(&pool_output[i], train)?;
            pool_output.push(out.avg_pool2d_with_stride(2, 2)?);
            block_output.push(out);
        }

        let num_pool = pool_output.len();
        let num_blk = block_output.len();

        let mut out = self.bottleneck.forward(&pool_output[num_pool - 1], train)?;

        for (i, block) in self.expansive_path.iter().enumerate() {
            out = block.forward(&out, &block_output[num_blk - 1 - i], train)?;
        }
        ops::sigmoid(&self.final_conv.forward(&out)?)
    }
}
