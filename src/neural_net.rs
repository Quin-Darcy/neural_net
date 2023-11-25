use core::num;

use crate::layer::Layer;
use crate::experience::{Experience, self};
use crate::utility::get_batches;


struct NeuralNet {
    num_layers: usize,
    layer_layout: Vec<usize>,
    layers: Vec<Layer>,
    learning_rate: f32,
    average_cost: f32,
}

impl NeuralNet {
    pub fn new(num_layers: usize, layer_layout: Vec<usize>, learning_rate: f32) -> Self {
        // Initialize the vector of layers based on the number of layers and the layer layout
        let mut layers: Vec::<Layer> = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            // Layer is initialized with num_inputs (number of neurons in previous layer), 
            // num_outputs (number of neurons in current layer), and its layer index
            if i == 0 {
                // If it is the input layer, it has no inputs since there is no layer before it
                layers.push(Layer::new(0, layer_layout[i], i));
            } else {
                layers.push(Layer::new(layer_layout[i-1], layer_layout[i], i));
            }
        }

        Self { num_layers, layer_layout, layers, learning_rate, average_cost: 0.0 }
    }

    pub fn train(&mut self, training_data: Vec<Experience>, num_batches: usize, epochs: usize) {
        // Break the training data into batches
        if num_batches > training_data.len() {
            panic!("Number of batches cannot be greater than the number of training data");
        } 

        let mut batches: Vec<Vec<Experience>> = get_batches(&training_data, num_batches);

        // Train the neural net for the specified number of epochs
        let training_data_len = training_data.len();
        let mut cost_sum = 0.0;
        for _ in 0..epochs {
            for batch in batches.iter_mut() {
                cost_sum += self.train_on_batch(batch);
            }
            self.average_cost = cost_sum / training_data_len as f32;
        }
    }

    fn train_on_batch(&mut self, batch: &mut [Experience]) -> f32 {
    // Create vectors to hold the total weight and bias gradients for each layer
    let mut total_biases_gradients: Vec<Vec<f32>> = Vec::with_capacity(self.num_layers - 1);
    let mut total_weights_gradients: Vec<Vec<f32>> = Vec::with_capacity(self.num_layers - 1);

        // Initialize these to zero - Note that we are starting the loop at 1 since the input layer
        // does not have any weights or biases 
        for i in 1..self.num_layers {
            total_biases_gradients.push(vec![0.0; self.layer_layout[i]]);
            total_weights_gradients.push(vec![0.0; self.layer_layout[i-1] * self.layer_layout[i]]);
        }

        let num_experiences = batch.len();

        // Perform feed forward and back propagation and get update the bias and weight gradients
        let mut batch_loss = 0.0;
        let mut nn_output: Vec<f32> = Vec::with_capacity(self.layer_layout[self.num_layers - 1]);
        for experience in batch.iter_mut() {
            nn_output = self.feed_forward(experience);
            
            // Compute the cost of this experience and add it to batch loss
            batch_loss += self.cost(experience, &nn_output);

            self.backwards_propagate(experience, &mut total_biases_gradients, &mut total_weights_gradients);
        }

        // Average the gradients and update the weights and biases of each layer
        for i in 1..self.num_layers {
            // Average the gradients
            let average_biases_gradients = total_biases_gradients[i-1].iter().map(|sum| sum / num_experiences as f32).collect::<Vec<f32>>();
            let average_weights_gradients = total_weights_gradients[i-1].iter().map(|sum| sum / num_experiences as f32).collect::<Vec<f32>>();

            // Update the weights and biases of layer i using the averaged gradients of the ith layer's weights and biases
            self.layers[i].update(&average_weights_gradients, &average_biases_gradients);
        }

        batch_loss
    }

    fn feed_forward(&mut self, data: &mut Experience) -> Vec<f32> {
        // Convert the Experience's 'state' member into a Vec<f32> so that it can be consumed by the neural net
        let input = data.convert_to_f32_vec();
        let mut layer_output: Vec<f32> = Vec::new();
        let mut nn_output: Vec<f32> = Vec::with_capacity(self.layer_layout[self.num_layers-1]);

        for i in 0..self.num_layers {
            if i == 0 {
                layer_output = self.layers[i].feed_forward(&input);
            } else if i == self.num_layers - 1 {
                nn_output = self.layers[i].feed_forward(&layer_output);
            } else {
                layer_output = self.layers[i].feed_forward(&layer_output);
            }
        }

        // Return the output of the neural net
        nn_output
    }

    fn backwards_propagate(&mut self, target_experience: &mut Experience, total_bias_gradients: &mut Vec<Vec<f32>>, total_weight_gradients: &mut Vec<Vec<f32>>) {
        let mut layer_neuron_errors: Vec<f32> = Vec::new();
        let mut layer_bias_gradients: Vec<f32> = Vec::new();
        let mut layer_weight_gradients: Vec<f32> = Vec::new();

        let temp_weights: Vec<f32> = vec![0.0; 1];
        let temp_error_terms: Vec<f32> = vec![0.0; 1];

        let target = &target_experience.convert_to_f32_vec();

        for i in (1..self.num_layers).rev() {
            if i == self.num_layers-1 {
                layer_neuron_errors = self.layers[i].calculate_neuron_errors(target, &temp_weights, &temp_error_terms);
            } else {
                let next_layer_weights = self.layers[i+1].weights.clone();
                layer_neuron_errors = self.layers[i].calculate_neuron_errors(target, &next_layer_weights, &layer_neuron_errors);
            }

            let previous_layer_output = self.layers[i-1].output.clone();
            layer_bias_gradients = layer_neuron_errors.clone();
            layer_weight_gradients = self.layers[i].calculate_weight_gradients(&previous_layer_output);

            // Add each component of the layer_bias_gradients and layer_weights_gradients to the total_bias_gradients and total_weight_gradients
            for (a, &b) in total_bias_gradients[i].iter_mut().zip(layer_bias_gradients.iter()) {
                *a += b;
            }

            for (a, &b) in total_weight_gradients[i].iter_mut().zip(layer_weight_gradients.iter()) {
                *a += b;
            }
        }
    }

    fn cost(&self, target_experience: &mut Experience, prediction: &[f32]) -> f32 {
        let target = target_experience.convert_to_f32_vec();

        target.iter()
            .zip(prediction.iter())
            .map(|(&a, &p)| (a - p).powi(2))
            .sum::<f32>() / 2.0
    }
}