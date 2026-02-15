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
        num_epochs: 100,
        batch_size: 128,
        num_workers: 6,
        seed: 42,
        learning_rate: 1.0e-3,
    };

    let model = models::CascadeForward::try_new(&device, [2, 20, 1]).unwrap();
    model.train::<MyBackend>("./tmp/cascade1/burn-regression", &config, true)?;

    let model = models::CascadeForward::try_new(&device, [2, 10, 10, 1]).unwrap();
    model.train::<MyBackend>("./tmp/cascade2/burn-regression", &config, true)?;

    Ok(())
}
