mod unet;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{VarBuilder, VarMap};
use unet::Unet;

fn main() -> Result<()> {
    let dev = Device::cuda_if_available(0)?;
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);
    let unet = Unet::new(1, 64, 1, vb.pp("unet"))?;
    let xs = Tensor::randn(0., 0.1, (1, 1, 512, 512), &dev)?.to_dtype(DType::F32)?;
    let out = unet.forward(&xs, false)?;
    println!("output shape: {:?}", out.dims());
    Ok(())
}
