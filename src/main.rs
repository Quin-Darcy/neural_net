mod layer;
mod utility;
mod constants;

use layer::Layer;
use utility::flat_matrix_vector_mult;

fn main() {
    let flat_matrix = [1.0, 0.0, 2.0, 3.0, 1.0, 1.0, 2.0, 4.0, 0.0, 1.0, 3.0, 1.0];
    let v = [2.0, 1.0, 0.0, 3.0];

    let result = flat_matrix_vector_mult(&flat_matrix, &v, 3, 4);

    println!("{:?}", result);
}