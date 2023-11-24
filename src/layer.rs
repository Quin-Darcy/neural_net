use rand::Rng;

use crate::utility::sigmoid_vec;

#[derive(Debug)]
pub struct Layer {
    weights: Vec<f32>,
    biases: Vec<f32>,
}

impl Layer {
    // Generate a new layer with random weights and biases
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Initialize the weights using Xavier initialization
        let weight_scale = (6.0 / (input_size + output_size) as f32).sqrt();
        let weights = (0..input_size * output_size)
            .map(|_| rng.gen_range(-weight_scale..weight_scale))
            .collect();
        
        // Initialize the biases to zero
        let biases = vec![0.0; output_size];

        Self { weights, biases }
    }

    pub fn feed_forward(&mut self, input: &Vec<f32>) -> Vec<f32> {
        let num_inputs = input.len();
        let num_neurons = self.biases.len();

        let mut output = Vec::with_capacity(num_neurons);

        for i in 0..num_neurons {
            let mut neuron_output = self.biases[i];
            let weight_base_index = i * num_inputs;

            for j in 0..num_inputs {
                neuron_output += input[j] * self.weights[weight_base_index + j];
            }

            output.push(neuron_output);
        }

        sigmoid_vec(&output)
    }
}