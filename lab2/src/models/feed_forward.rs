use burn::{nn::LinearConfig, prelude::*};

use crate::impl_burn_steps;

use super::Model;

#[derive(Debug, Module)]
pub struct FeedForward<B: Backend> {
    inp_layer: nn::Linear<B>,
    hidd_layer: nn::Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(device: &Device<B>, hidd_neuron_number: usize) -> Self {
        FeedForward {
            inp_layer: LinearConfig::new(2, hidd_neuron_number).init(device),
            hidd_layer: LinearConfig::new(hidd_neuron_number, 1).init(device),
        }
    }
}

impl<B: Backend> Model<B> for FeedForward<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let act1 = nn::activation::Tanh::new();
        let x = self.inp_layer.forward(input);
        let x = act1.forward(x);
        self.hidd_layer.forward(x)
    }
}

impl_burn_steps!(FeedForward);
