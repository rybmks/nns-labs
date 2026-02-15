use burn::{nn::LinearConfig, prelude::*};

use crate::{impl_burn_steps, models::Model};

#[derive(Debug, Module)]
pub struct CascadeForward<B: Backend> {
    hidd_layers: Vec<nn::Linear<B>>,
    skip_layer: nn::Linear<B>,
}

impl<B: Backend> CascadeForward<B> {
    pub fn try_new(device: &Device<B>, layout: impl Into<Vec<usize>>) -> Result<Self, &str> {
        let layout = layout.into();

        if layout.len() < 3 {
            return Err("Layout should contain at least 3 elements (at least one hidd layer)");
        }

        let hidd_layers: Vec<nn::Linear<B>> = layout
            .windows(2)
            .map(|dims| LinearConfig::new(dims[0], dims[1]).init(device))
            .collect();

        let skip_layer = LinearConfig::new(layout[0], layout[layout.len() - 1]).init(device);

        Ok(CascadeForward {
            hidd_layers,
            skip_layer,
        })
    }
}

impl<B: Backend> Model<B> for CascadeForward<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input.clone();

        for (i, layer) in self.hidd_layers.iter().enumerate() {
            x = layer.forward(x);

            if i < self.hidd_layers.len() - 1 {
                x = burn::tensor::activation::relu(x)
            }
        }

        let s = self.skip_layer.forward(input);

        x + s
    }
}

impl_burn_steps!(CascadeForward);
