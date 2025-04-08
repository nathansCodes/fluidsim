pub mod cpu;
pub mod gpu;

pub(super) use cpu::pos_to_cell_coord;

use bevy::prelude::*;

use crate::SimState;

pub const CELL_OFFSETS: [Vec2; 9] = [
    Vec2::new(-1.0, -1.0),
    Vec2::new(0.0, -1.0),
    Vec2::new(1.0, -1.0),
    Vec2::new(-1.0, 0.0),
    Vec2::new(0.0, 0.0),
    Vec2::new(1.0, 0.0),
    Vec2::new(-1.0, 1.0),
    Vec2::new(0.0, 1.0),
    Vec2::new(1.0, 1.0),
];

#[derive(Resource)]
pub struct Sim {
    pub positions: Vec<Vec2>,
    pub predicted_positions: Vec<Vec2>,
    pub velocities: Vec<Vec2>,
    pub densities: Vec<(f32, f32)>,
    pub spatial_lookup: Vec<(usize, usize)>,
    pub start_indices: Vec<usize>,
    pub gravity: Vec2,
    pub bounds_size: Vec2,
    pub particle_radius: f32,
    pub smoothing_radius: f32,
    pub target_density: f32,
    pub pressure_multiplier: f32,
    pub near_pressure_multiplier: f32,
    pub viscosity: f32,
    pub delta: f32,
}

impl Default for Sim {
    fn default() -> Self {
        Self {
            positions: default(),
            predicted_positions: default(),
            velocities: default(),
            densities: default(),
            spatial_lookup: default(),
            start_indices: default(),
            gravity: Vec2::new(0.0, -10.0),
            bounds_size: Vec2::new(16.0, 9.0) * 2.0,
            particle_radius: 0.1,
            smoothing_radius: 1.2,
            target_density: 10.0,
            pressure_multiplier: 1000.0,
            near_pressure_multiplier: 40.0,
            viscosity: 0.2,
            delta: 120.0,
        }
    }
}

#[derive(Debug, Default, SubStates, Clone, Copy, Hash, PartialEq, Eq)]
#[source(SimState = SimState::Running)]
pub enum Device {
    #[default]
    CPU,
    GPU,
}
