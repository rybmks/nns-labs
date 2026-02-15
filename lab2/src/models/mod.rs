mod cascade_forward;
mod elman_backprop;
mod feed_forward;

pub use cascade_forward::*;
pub use elman_backprop::*;
pub use feed_forward::*;

pub use std::fmt::Display;

use burn::{
    Tensor,
    config::Config,
    data::dataloader::DataLoaderBuilder,
    lr_scheduler::linear::LinearLrSchedulerConfig,
    module::{AutodiffModule, Module},
    nn,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        InferenceStep, Learner, RegressionOutput, SupervisedTraining, TrainStep, metric::LossMetric,
    },
};

use crate::{
    config::TrainingConfig,
    create_artifact_dir,
    data::{self, RegressionBatch, RegressionBatcher},
    generate_data,
};

pub const TRAIN_START: f32 = 0.0;
pub const TRAIN_END: f32 = 20.0;
pub const TRAIN_POINTS: usize = 1000;

pub const TEST_START: f32 = 20.0;
pub const TEST_END: f32 = 30.0;
pub const TEST_POINTS: usize = 300;

pub trait Model<B: Backend> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2>;

    fn forward_regression(
        &self,
        input: Tensor<B, 2>,
        targets: Tensor<B, 2>,
    ) -> RegressionOutput<B> {
        let output = self.forward(input);
        let loss = nn::loss::MseLoss::new().forward(
            output.clone(),
            targets.clone(),
            nn::loss::Reduction::Mean,
        );

        RegressionOutput::new(loss, output, targets)
    }

    fn train<BA>(self, artifact_dir: &str, config: &TrainingConfig, shuffle: bool)
    where
        BA: AutodiffBackend,
        Self: AutodiffModule<BA> + TrainStep<Input = data::RegressionBatch<BA>> + Display + 'static,
        <Self as AutodiffModule<BA>>::InnerModule: InferenceStep<
                Input = RegressionBatch<BA::InnerBackend>,
                Output = RegressionOutput<BA::InnerBackend>,
            >,
        <<Self as burn::train::TrainStep>::Output as burn::train::ItemLazy>::ItemSync:
            burn::train::metric::Adaptor<burn::train::metric::LossInput<burn::backend::NdArray>>,
    {
        create_artifact_dir(artifact_dir);
        config
            .save(format!("{artifact_dir}/config.json"))
            .expect("Config should be saved successfully");

        let batcher_train = RegressionBatcher::default();
        let batcher_valid = RegressionBatcher::default();

        let mut dataloader_train = DataLoaderBuilder::new(batcher_train)
            .batch_size(config.batch_size)
            .num_workers(config.num_workers);

        let mut dataloader_test = DataLoaderBuilder::new(batcher_valid)
            .batch_size(config.batch_size)
            .num_workers(config.num_workers);

        if shuffle {
            dataloader_test = dataloader_test.shuffle(config.seed);
            dataloader_train = dataloader_train.shuffle(config.seed)
        }

        let dataloader_train =
            dataloader_train.build(generate_data(TRAIN_START, TRAIN_END, TRAIN_POINTS));
        let dataloader_test =
            dataloader_test.build(generate_data(TEST_START, TEST_END, TEST_POINTS));

        let iterations_per_epoch = TRAIN_POINTS.div_ceil(config.batch_size);
        let total_iterations = iterations_per_epoch * config.num_epochs;

        let lr_scheduler =
            LinearLrSchedulerConfig::new(config.learning_rate, 1.0e-5, total_iterations)
                .init()
                .expect("failed to create scheduler");

        let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
            .metrics((LossMetric::new(),))
            .with_file_checkpointer(CompactRecorder::new())
            .num_epochs(config.num_epochs)
            .summary();

        let result = training.launch(Learner::new(self, config.optimizer.init(), lr_scheduler));

        result
            .model
            .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
            .expect("Trained model should be saved successfully");
    }
}

#[macro_export]
macro_rules! impl_burn_steps {
    ($structname:ident) => {
        impl<B: $crate::models::AutodiffBackend> $crate::models::TrainStep for $structname<B> {
            type Input = $crate::models::RegressionBatch<B>;
            type Output = $crate::models::RegressionOutput<B>;

            fn step(
                &self,
                batch: $crate::models::RegressionBatch<B>,
            ) -> burn::train::TrainOutput<$crate::models::RegressionOutput<B>> {
                let item = self.forward_regression(batch.inputs, batch.targets);

                burn::train::TrainOutput::new(self, item.loss.backward(), item)
            }
        }

        impl<B: Backend> $crate::models::InferenceStep for $structname<B> {
            type Input = $crate::models::RegressionBatch<B>;
            type Output = $crate::models::RegressionOutput<B>;

            fn step(
                &self,
                batch: $crate::models::RegressionBatch<B>,
            ) -> $crate::models::RegressionOutput<B> {
                self.forward_regression(batch.inputs, batch.targets)
            }
        }
    };
}
