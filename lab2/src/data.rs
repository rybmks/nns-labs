use burn::{
    Tensor,
    data::dataloader::batcher::Batcher,
    prelude::Backend,
    tensor::{Shape, TensorData},
};

#[derive(Clone, Debug)]
pub struct RegressionItem {
    pub input: [f32; 2],
    pub target: [f32; 1],
}

#[derive(Clone, Debug)]
pub struct RegressionBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
}

#[derive(Clone, Default)]
pub struct RegressionBatcher {}

impl<B: Backend> Batcher<B, RegressionItem, RegressionBatch<B>> for RegressionBatcher {
    fn batch(
        &self,
        items: Vec<RegressionItem>,
        device: &<B as Backend>::Device,
    ) -> RegressionBatch<B> {
        let inputs_iter = items.iter().map(|item| item.input);
        let targets_iter = items.iter().map(|item| item.target);

        let inputs_data = TensorData::new(
            inputs_iter.flatten().collect(),
            Shape::new([items.len(), 2]),
        );
        let targets_data = TensorData::new(
            targets_iter.flatten().collect(),
            Shape::new([items.len(), 1]),
        );

        let inputs = Tensor::from_data(inputs_data, device);
        let targets = Tensor::from_data(targets_data, device);

        RegressionBatch { inputs, targets }
    }
}
