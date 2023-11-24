use rand::Rng;

mod layer;
mod utility;

use layer::Layer;

fn main() {
    let layer = Layer::new(2, 3);
    println!("{:?}", layer);
}