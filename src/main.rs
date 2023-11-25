use bitvec::prelude::*;
use rand::Rng;

mod neural_net;
mod layer;
mod utility;
mod constants;
mod experience;

use crate::experience::Experience;
use crate::neural_net::NeuralNet;


fn generate_training_data(num_samples: usize, input_size: usize) -> Vec<Experience> {
    let mut rng = rand::thread_rng();
    let mut training_data = Vec::with_capacity(num_samples);

    for _ in 0..num_samples {
        let mut state = bitvec![0; input_size];
        let mut new_state = bitvec![0; input_size];
        for mut bit in state.iter_mut() {
            *bit = rng.gen(); // Randomly set each bit
        }

        for mut bit in new_state.iter_mut() {
            *bit = rng.gen(); // Randomly set each bit
        }

        // Assuming Experience has a new method that takes a state and other necessary fields
        let experience = Experience::new(state, new_state);
        training_data.push(experience);
    }

    training_data
}

fn main() {
    let num_samples = 100;
    let num_layers = 10;
    let input_size = 10;
    let output_size = 10;
    let layer_layout = vec![input_size, 8, 3, 11, 6, 12, 7, 9, 9, output_size];

    let training_data = generate_training_data(num_samples, input_size);

    let mut nn = NeuralNet::new(num_layers, layer_layout, 0.5);

    nn.train(training_data, 10, 10);
}