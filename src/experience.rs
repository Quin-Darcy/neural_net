use bitvec::prelude::*;


pub struct Experience {
    state: BitVec,
    action: usize, // Represents index in state bitvec to toggle on/off
    reward: f32,
    new_state: BitVec,
}