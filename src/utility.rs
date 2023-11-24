// Single f32 input sigmoid function
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Vectorized sigmoid function
pub fn sigmoid_vec(inputs: &[f32]) -> Vec<f32> {
    inputs.iter().map(|x| sigmoid(*x)).collect()
}

// Vectorized sigmoid derivative function
pub fn sigmoid_derivative_vec(inputs: &[f32]) -> Vec<f32> {
    inputs.iter().map(|x| {
        let sig = sigmoid(*x);
        sig * (1.0 - sig)
    }).collect()
}