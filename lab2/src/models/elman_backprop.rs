use burn::{nn::LinearConfig, prelude::*};

use crate::impl_burn_steps;

use super::Model;

#[derive(Debug, Module)]
struct ElmanLayout<B: Backend> {
    hidd: nn::Linear<B>,
    context: nn::Linear<B>,
    out_size: usize,
}

#[derive(Debug, Module)]
pub struct ElmanBackprop<B: Backend> {
    contexted_layers: Vec<ElmanLayout<B>>,
    output_layer: nn::Linear<B>,
}

impl<B: Backend> ElmanBackprop<B> {
    pub fn try_new(device: &Device<B>, layout: impl Into<Vec<usize>>) -> Result<Self, &str> {
        let layout = layout.into();
        let layout_len = layout.len();

        if layout.len() < 3 {
            return Err("Layout should contain at least 3 elements (at least one hidd layer)");
        }

        let mut contexted_layers = vec![];

        for dims in layout.windows(2).take(layout_len - 2) {
            let in_size = dims[0];
            let out_size = dims[1];

            contexted_layers.push(ElmanLayout {
                hidd: LinearConfig::new(in_size, out_size).init(device),
                context: LinearConfig::new(out_size, out_size).init(device),
                out_size,
            });
        }

        let output_layer =
            LinearConfig::new(layout[layout_len - 2], layout[layout_len - 1]).init(device);

        Ok(ElmanBackprop {
            contexted_layers,
            output_layer,
        })
    }
}

impl<B: Backend> Model<B> for ElmanBackprop<B> {
    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let [seq_len, features] = input.dims();
        let device = input.device();

        let mut states: Vec<Tensor<B, 2>> = self
            .contexted_layers
            .iter()
            .map(|layer| Tensor::zeros([1, layer.out_size], &device))
            .collect();

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let mut x = input.clone().slice([t..t + 1, 0..features]);

            for (i, layer) in self.contexted_layers.iter().enumerate() {
                let in_signal = layer.hidd.forward(x.clone());
                let ctx_signal = layer.context.forward(states[i].clone());

                x = burn::tensor::activation::tanh(in_signal + ctx_signal);

                states[i] = x.clone();
            }

            let out_point = self.output_layer.forward(x.clone());
            outputs.push(out_point);
        }

        Tensor::cat(outputs, 0)
    }
}

impl_burn_steps!(ElmanBackprop);
