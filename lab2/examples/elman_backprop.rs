use burn::backend::{Autodiff, ndarray::NdArray};
use burn::optim::AdamConfig;
use burn::optim::decay::WeightDecayConfig;
use lab2::config::TrainingConfig;
use lab2::models::{self, Model};

fn main() -> Result<(), &'static str> {
    type MyBackend = Autodiff<NdArray>;

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let config = TrainingConfig {
        optimizer: AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1e-4))),
        num_epochs: 130,
        batch_size: 164,
        num_workers: 6,
        seed: 42,
        learning_rate: 1.0e-3,
    };

    let model = models::ElmanBackprop::try_new(&device, [2, 15, 1]).unwrap();
    model.train::<MyBackend>("./tmp/elman1/burn-regression", &config, false)?;

    let model = models::ElmanBackprop::try_new(&device, [2, 5, 5, 5, 1]).unwrap();
    model.train::<MyBackend>("./tmp/elman2/burn-regression", &config, false)?;

    Ok(())
}
