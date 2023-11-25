use crate::layer::Layer;
use crate::experience::Experience;


struct NeuralNet {
    num_layers: usize,
    layer_layout: Vec<usize>,
    layers: Vec<Layer>,
    learning_rate: f32,
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

        Self { num_layers, layer_layout, layers, learning_rate }
    }

    pub fn train(&self, training_data: Vec<Experience>, num_batches: usize, epochs: usize) {
        let batches: Vec<&[Experience]> = Vec::with_capacity(num_batches);
    }
}