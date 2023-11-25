use rand::Rng;

use crate::utility::{sigmoid_vec, sub_vec, hadamard_prod_vec, sigmoid_derivative_vec, flat_matrix_vector_mult};
use crate::constants::NUM_LAYERS;

#[derive(Debug)]
pub struct Layer {
    num_neurons: usize,
    num_inputs: usize,
    layer_index: usize,

    weights: Vec<f32>,
    biases: Vec<f32>,
    weighted_sums: Vec<f32>,
    output: Vec<f32>,
    error_terms: Vec<f32>,
}

impl Layer {
    // Generate a new layer with random weights and biases
    pub fn new(input_size: usize, output_size: usize, layer_index: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Initialize layer information
        let num_neurons = output_size;
        let num_inputs = input_size;

        // Initialize the weights using Xavier initialization
        let weight_scale = (6.0 / (input_size + output_size) as f32).sqrt();
        let weights = (0..input_size * output_size)
            .map(|_| rng.gen_range(-weight_scale..weight_scale))
            .collect();
        
        // Initialize the biases to zero
        let biases = vec![0.0; num_neurons];

        // Initialize the weighted sums, output, and error terms to zero
        let weighted_sums = vec![0.0; num_neurons];
        let output = vec![0.0; num_neurons];
        let error_terms = vec![0.0; num_neurons];

        Self { num_neurons, num_inputs, layer_index, weights, biases, weighted_sums, output, error_terms }
    }

    // Feed the input of the previous layer into the current layer and return its output
    pub fn feed_forward(&mut self, input: &Vec<f32>) -> Vec<f32> {
        let mut pre_activations = Vec::with_capacity(self.num_neurons);

        // Calculate weighted sum neuron by neuron
        for i in 0..self.num_neurons {
            // The bias is added first
            let mut neuron_output = self.biases[i];

            // This signifies the index for the group of weights corresponding to the current neuron
            let weight_base_index = i * self.num_inputs;

            // Create the linear combination of weights and inputs
            for j in 0..self.num_inputs {
                neuron_output += input[j] * self.weights[weight_base_index + j];
            }
            pre_activations.push(neuron_output);
        }
        
        // Set the pre-activation values (weighted_sums) and store the output for later reference
        self.weighted_sums = pre_activations.clone();
        self.output = sigmoid_vec(&pre_activations);

        return self.output.clone();
    }

    pub fn calculate_error_terms(&mut self, target: &[f32], next_layer_weights: &[f32], next_layer_error_terms: &[f32]) -> Vec<f32> {
        let mut errors = Vec::with_capacity(self.num_neurons);

        // Error terms are different for the last layer
        errors = if self.layer_index == NUM_LAYERS - 1 {
            hadamard_prod_vec(
                &sub_vec(&self.output, &target), 
                &sigmoid_derivative_vec(&self.weighted_sums)
            )
        } else {
            hadamard_prod_vec(
                &flat_matrix_vector_mult(
                    &next_layer_weights, 
                    &next_layer_error_terms, 
                    self.num_neurons, 
                    next_layer_error_terms.len()
                ), 
                &sigmoid_derivative_vec(&self.weighted_sums)
            )
        };

        self.error_terms = errors.clone();
        errors
    }
}