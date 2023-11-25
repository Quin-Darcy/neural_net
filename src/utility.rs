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

pub fn sub_vec(vec1: &[f32], vec2: &[f32]) -> Vec<f32> {
    // Ensure vec1 and vec2 are of the same length to avoid panics
    assert_eq!(vec1.len(), vec2.len(), "Vectors must be of the same length");

    // Create a new vector with the same capacity as vec1
    let mut result = Vec::with_capacity(vec1.len());

    // Iterate over the elements of vec1 and vec2, subtract and push to result
    for (&a, &b) in vec1.iter().zip(vec2.iter()) {
        result.push(a - b);
    }

    result
}

pub fn hadamard_prod_vec(vec1: &[f32], vec2: &[f32]) -> Vec<f32> {
    // Ensure vec1 and vec2 are the same length
    assert_eq!(vec1.len(), vec2.len(), "Vectors must be of the same length");

    // Create new vector with capacity equal to size of first vector
    let mut result = Vec::with_capacity(vec1.len());

    // Iterate over the elements and perform element-wise multiplications
    for (&a, &b) in vec1.iter().zip(vec2.iter()) {
        result.push(a * b);
    }

    result
}

pub fn flat_matrix_vector_mult(flat_matrix: &[f32], v: &[f32], columns: usize, rows: usize) -> Vec<f32> {
    let mut result = Vec::with_capacity(rows);

    for i in 0..columns {
        let mut sum: f32 = 0.0;
        
        for j in 0..rows {
            sum += v[j] * flat_matrix[j * columns + i];
        }

        result.push(sum);
    }

    result
}