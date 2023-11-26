use crate::experience::Experience;
use crate::errors::NeuralNetError;


// Single f32 input sigmoid function
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Vectorized sigmoid function
pub fn sigmoid_vec(inputs: &[f32]) -> Result<Vec<f32>, NeuralNetError> {
    if inputs.is_empty() {
        return Err(NeuralNetError::EmptyVector{
            message: "sigmoid_vec received empty inputs vector".to_string(),
            line: line!(),
            file: file!().to_string(),
        });
    }

    Ok(inputs.iter().map(|x| sigmoid(*x)).collect())
}

// Vectorized sigmoid derivative function
pub fn sigmoid_derivative_vec(inputs: &[f32]) -> Result<Vec<f32>, NeuralNetError>  {
    if inputs.is_empty() {
        return Err(NeuralNetError::EmptyVector { 
            message: "sigmoid_derivative_vec received empty inputs vector".to_string(), 
            line: line!(), 
            file: file!().to_string(), 
        });
    }

    Ok(inputs.iter().map(|x| {
        let sig = sigmoid(*x);
        sig * (1.0 - sig)
    }).collect())
}

pub fn sub_vec(vec1: &[f32], vec2: &[f32]) -> Result<Vec<f32>, NeuralNetError> {
    // Check if either vector is empty
    if vec1.is_empty() || vec2.is_empty() {
        return Err(NeuralNetError::EmptyVector {
            message: "sub_vec received one or two empty vectors".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    // Ensure vec1 and vec2 are of the same length to avoid panics
    if vec1.len() != vec2.len() {
        return Err(NeuralNetError::InvalidDimensions {
            message: "sub_vec received vectors of different sizes".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    // Create a new vector with the same capacity as vec1
    let mut result = Vec::with_capacity(vec1.len());

    // Iterate over the elements of vec1 and vec2, subtract and push to result
    for (&a, &b) in vec1.iter().zip(vec2.iter()) {
        result.push(a - b);
    }

    Ok(result)
}

pub fn hadamard_prod_vec(vec1: &[f32], vec2: &[f32]) -> Result<Vec<f32>, NeuralNetError> {
    // Check if arguments are empty
    if vec1.is_empty() || vec2.is_empty() {
        return Err(NeuralNetError::EmptyVector {
            message: "hadamard_prod_vec received one or two empty vectors".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    // Ensure vec1 and vec2 are the same length
    if vec1.len() != vec2.len() {
        return Err(NeuralNetError::InvalidDimensions {
            message: "hadamard_prod_vec received vectors with non-matching lengths".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    // Create new vector with capacity equal to size of first vector
    let mut result = Vec::with_capacity(vec1.len());

    // Iterate over the elements and perform element-wise multiplications
    for (&a, &b) in vec1.iter().zip(vec2.iter()) {
        result.push(a * b);
    }

    Ok(result)
}

pub fn flat_matrix_vector_mult(flat_matrix: &[f32], v: &[f32], columns: usize, rows: usize) -> Result<Vec<f32>, NeuralNetError> {
    // Check if inputs are empty
    if flat_matrix.is_empty() || v.is_empty() {
        return Err(NeuralNetError::EmptyVector {
            message: "flat_matrix_vector_mult received one or two empty vectors".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    // Check bounds on matrix and vector
    if flat_matrix.len() != columns * rows {
        return Err(NeuralNetError::InvalidDimensions {
            message: "flat_matrix_vector_mult received invalid matrix or columns/rows values".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    if v.len() != rows {
        return Err(NeuralNetError::InvalidDimensions {
            message: "flat_matrix_vector_mult received vector with invalid length".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    let mut result = Vec::with_capacity(rows);
    let mut sum: f32 = 0.0;

    for i in 0..columns {
        for j in 0..rows {
            sum += v[j] * flat_matrix[j * columns + i];
        }
        result.push(sum);
        sum = 0.0;
    }

    Ok(result)
}

// This returns a flat matrix
pub fn outer_product(v1: &[f32], v2: &[f32]) -> Result<Vec<f32>, NeuralNetError> {
    // Check if either vector is empty
    if v1.is_empty() || v2.is_empty() {
        return Err(NeuralNetError::EmptyVector {
            message: "outer_product received one or two empty vectors".to_string(),
            line: line!(), 
            file: file!().to_string(),
        })
    }

    let mut result = Vec::with_capacity(v1.len() * v2.len());

    for &a in v2.iter() {
        for &b in v1.iter() {
            result.push(a * b);
        }
    }

    Ok(result)
}

pub fn get_batches(training_data: &[Experience], num_batches: usize) -> Result<Vec<Vec<Experience>>, NeuralNetError> {
    // Check if training data is empty
    if training_data.is_empty() {
        return Err(NeuralNetError::EmptyVector {
            message: "get_batches received empty set of Experiences".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    // Check that num_batches is not 0
    if num_batches == 0 {
        return Err(NeuralNetError::InvalidDimensions {
            message: "get_batches received request for 0 batches".to_string(), 
            line: line!(), 
            file: file!().to_string(),
        })
    }

    // Check that num_batches is not greater than the length of training data
    if num_batches > training_data.len() {
        return Err(NeuralNetError::InvalidDimensions {
            message: "get_batches received request for more batches than the size of training data".to_string(),
            line: line!(),
            file: file!().to_string(),
        })
    }

    let mut batches: Vec<Vec<Experience>> = Vec::with_capacity(num_batches);

    if training_data.len() % num_batches == 0 {
        // If the number of training data is divisible by the number of batches, 
        // then each batch will have the same number of training data
        let batch_size = training_data.len() / num_batches;
        for i in 0..num_batches {
            batches.push(training_data[i*batch_size..(i+1)*batch_size].to_vec());
        }
    } else {
        // If the number of training data is not divisible by the number of batches, 
        // then each batch will have the same number of training data except for the last batch
        let batch_size = training_data.len() / num_batches;
        for i in 0..num_batches-1 {
            batches.push(training_data[i*batch_size..(i+1)*batch_size].to_vec());
        }
        batches.push(training_data[(num_batches-1)*batch_size..].to_vec());
    }  

    Ok(batches)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        let input1: f32 = 0.5;
        let input2: f32 = 12.0;
        let input3: f32 = -43.2;

        let expected_output1: f32 = 1.0 / (1.0 + (-input1).exp());
        let expected_output2: f32 = 1.0 / (1.0 + (-input2).exp());
        let expected_output3: f32 = 1.0 / (1.0 + (-input3).exp()); 

        assert!((sigmoid(input1) - expected_output1).abs() < f32::EPSILON);
        assert!((sigmoid(input2) - expected_output2).abs() < f32::EPSILON);
        assert!((sigmoid(input3) - expected_output3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_sigmoid_vec() {
        let inputs1: Vec<f32> = vec![0.0f32, 2.0, -2.0];
        let inputs2: Vec<f32> = vec![0.01];
        let inputs3: Vec<f32> = Vec::new();

        let expected_outputs1: Vec<f32> = inputs1.iter().map(|&x| sigmoid(x)).collect::<Vec<f32>>();
        let expected_outputs2: Vec<f32> = inputs2.iter().map(|&x| sigmoid(x)).collect::<Vec<f32>>();

        // Test successful cases
        assert_eq!(sigmoid_vec(&inputs1).unwrap(), expected_outputs1);
        assert_eq!(sigmoid_vec(&inputs2).unwrap(), expected_outputs2);

        // Test error case
        assert!(matches!(sigmoid_vec(&inputs3), Err(NeuralNetError::EmptyVector)));
    }
}
