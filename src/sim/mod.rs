pub mod cpu;
pub mod gpu;

pub(super) use cpu::pos_to_cell_coord;

use bevy::prelude::*;

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
    pub particle_radius: f32,
    pub smoothing_radius: f32,
    pub gravity: Vec2,
    pub bounds_size: Vec2,
    pub positions: Vec<Vec2>,
    pub predicted_positions: Vec<Vec2>,
    pub velocities: Vec<Vec2>,
    pub densities: Vec<(f32, f32)>,
    pub spatial_lookup: Vec<(usize, usize)>,
    pub start_indices: Vec<usize>,
    pub target_density: f32,
    pub pressure_multiplier: f32,
    pub near_pressure_multiplier: f32,
    pub delta: f32,
    pub viscosity: f32,
}

impl Default for Sim {
    fn default() -> Self {
        Self {
            particle_radius: 0.1,
            smoothing_radius: 1.2,
            gravity: Vec2::new(0.0, -10.0),
            bounds_size: Vec2::new(16.0, 9.0) * 2.0,
            positions: default(),
            predicted_positions: default(),
            velocities: default(),
            densities: default(),
            spatial_lookup: default(),
            start_indices: default(),
            target_density: 10.0,
            pressure_multiplier: 1000.0,
            delta: 120.0,
            viscosity: 0.2,
            near_pressure_multiplier: 40.0,
        }
    }
}
