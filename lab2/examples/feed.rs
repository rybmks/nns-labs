use burn::backend::{Autodiff, ndarray::NdArray};
use burn::optim::AdamConfig;
use burn::optim::decay::WeightDecayConfig;
use lab2::models::Model;
use lab2::{config::TrainingConfig, models::FeedForward};

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

    let model = FeedForward::new(&device, 10);
    model.train::<MyBackend>("./tmp/feed1/burn-regression", &config, true)?;

    let model = FeedForward::new(&device, 20);
    model.train::<MyBackend>("./tmp/feed2/burn-regression", &config, true)?;

    Ok(())
}
