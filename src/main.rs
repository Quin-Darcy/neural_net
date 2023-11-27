use bitvec::prelude::*;
use rand::Rng;

mod neural_net;
mod layer;
mod utility;
mod constants;
mod experience;
mod errors;

use crate::experience::Experience;
use crate::neural_net::NeuralNet;


fn create_xor_data() -> Vec<Experience> {
    vec![
        Experience::new(bitvec![0, 0], bitvec![0]),
        Experience::new(bitvec![0, 1], bitvec![1]),
        Experience::new(bitvec![1, 0], bitvec![1]),
        Experience::new(bitvec![1, 1], bitvec![0]),
    ]
}

fn main() {
    let training_data = create_xor_data();
    let num_layers = 3; // input, one hidden, output
    let layer_layout = vec![2, 4, 1]; // 2 inputs, 2 neurons in hidden layer, 1 output
    let learning_rate = 0.1;
    let epochs = 10000;

    let mut neural_net = NeuralNet::new(num_layers, layer_layout, learning_rate);

    // Test the untrained network
    println!("------------ UNTRAINED NETWORK --------------\n");
    println!("{:<15} {:<15} {:<15}", "Input", "Prediction", "Expected");
    for mut experience in create_xor_data() {
        let prediction = neural_net.feed_forward(&mut experience).unwrap();
        let input_str = format!("{:?}", experience.state.iter().map(|bit| if *bit { 1 } else { 0 }).collect::<Vec<i32>>());
        let expected_str = format!("{:?}", experience.new_state.iter().map(|bit| if *bit { 1 } else { 0 }).collect::<Vec<i32>>());
        let prediction_str = format!("{:?}", prediction);

        println!("{:<15} {:<15} {:<15}", input_str, prediction_str, expected_str);
    }
    println!("-------------------------------------------\n");

    match neural_net.train(training_data, 2, epochs) {
        Ok(_) => println!("Training successful"),
        Err(e) => println!("Training failed: {:?}", e),
    }

    // Test the trained network
    println!("\n-------------- TRAINED NETWORK ----------------");
    println!("{:<15} {:<15} {:<15}", "Input", "Prediction", "Expected");
    for mut experience in create_xor_data() {
        let prediction = neural_net.feed_forward(&mut experience).unwrap();
        let input_str = format!("{:?}", experience.state.iter().map(|bit| if *bit { 1 } else { 0 }).collect::<Vec<i32>>());
        let expected_str = format!("{:?}", experience.new_state.iter().map(|bit| if *bit { 1 } else { 0 }).collect::<Vec<i32>>());
        let prediction_str = format!("{:?}", prediction);

        println!("{:<15} {:<15} {:<15}", input_str, prediction_str, expected_str);
    }
    println!("-------------------------------------------\n");
}
