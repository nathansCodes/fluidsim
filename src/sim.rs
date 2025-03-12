use core::f32;
use std::{
    f32::consts::PI,
    num::Wrapping,
    sync::{Arc, Mutex},
};

use bevy::{color, prelude::*, window::PrimaryWindow};
use ops::FloatPow;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    controls::{InteractionMode, InteractionSettings, SimCamera},
    debug::{DebugData, LogLevel},
};

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

impl Sim {
    pub fn density_at_point(&self, point: Vec2) -> (f32, f32) {
        if self.positions.is_empty() {
            return (0.0, 0.0);
        }

        let mut density = 0.0;
        let mut near_density = 0.0;

        let center = pos_to_cell_coord(point, self.smoothing_radius);
        let sqr_radius = self.smoothing_radius.squared();

        for offset in CELL_OFFSETS {
            let hash = hash_cell_coord(center + offset);
            let key = self.get_key_from_hash(hash);
            let start_index = self.start_indices[key];

            if start_index == usize::MAX {
                continue;
            }

            for (particle_index, particle_cell_key) in &self.spatial_lookup[start_index..] {
                if *particle_cell_key != key {
                    break;
                }

                let sqr_dst = (self.predicted_positions[*particle_index] - point).length_squared();

                if sqr_dst > sqr_radius {
                    continue;
                }

                density += smoothing_kernel(sqr_dst.sqrt(), self.smoothing_radius);
                near_density += spiky_kernel(sqr_dst.sqrt(), self.smoothing_radius);
            }
        }

        (density, near_density)
    }

    pub fn calculate_pressure_force(&self, particle: usize) -> Vec2 {
        let mut pressure_force = Vec2::ZERO;

        let point = self.predicted_positions[particle];
        let (density, near_density) = self.densities[particle];
        let pressure = self.density_to_pressure(density);
        let near_pressure = self.near_density_to_pressure(near_density);

        let center = pos_to_cell_coord(point, self.smoothing_radius);
        let sqr_radius = self.smoothing_radius.squared();

        for offset in CELL_OFFSETS {
            let hash = hash_cell_coord(center + offset);
            let key = self.get_key_from_hash(hash);
            let start_index = self.start_indices[key];

            if start_index == usize::MAX {
                continue;
            }

            for (particle_index, particle_cell_key) in &self.spatial_lookup[start_index..] {
                if *particle_cell_key != key {
                    break;
                }

                let pos = self.positions[*particle_index];

                let sqr_dst = (pos - point).length_squared();

                if sqr_dst > sqr_radius || *particle_index == particle {
                    continue;
                }

                let distance = sqr_dst.sqrt();

                if distance < 0.0001 {
                    let x = rand::random_range(-0.3..0.3);
                    let y = rand::random_range(-0.3..0.3);
                    pressure_force += Vec2::new(x, y);
                    continue;
                }

                let direction = (pos - point) / distance;

                let (other_density, other_near_density) = self.densities[*particle_index];

                let shared_pressure = (pressure + self.density_to_pressure(other_density)) / 2.0;
                let shared_near_pressure =
                    (near_pressure + self.near_density_to_pressure(other_near_density)) / 2.0;

                let force = smoothing_kernel_derivative(distance, self.smoothing_radius);
                let near_force = spiky_kernel_derivative(distance, self.smoothing_radius);

                pressure_force += shared_pressure * direction * force / other_density;
                pressure_force +=
                    shared_near_pressure * direction * near_force / other_near_density;
            }
        }

        pressure_force
    }

    pub fn calculate_viscosity_force(&self, particle: usize) -> Vec2 {
        let mut viscosity_force = Vec2::ZERO;

        let pos = self.predicted_positions[particle];
        let vel = self.velocities[particle];

        let center = pos_to_cell_coord(pos, self.smoothing_radius);
        let sqr_radius = self.smoothing_radius.squared();

        for offset in CELL_OFFSETS {
            let hash = hash_cell_coord(center + offset);
            let key = self.get_key_from_hash(hash);
            let start_index = self.start_indices[key];

            if start_index == usize::MAX {
                continue;
            }

            for (particle_index, particle_cell_key) in &self.spatial_lookup[start_index..] {
                if *particle_cell_key != key {
                    break;
                }

                let other_pos = self.positions[*particle_index];

                let sqr_dst = (other_pos - pos).length_squared();

                if sqr_dst > sqr_radius || *particle_index == particle {
                    continue;
                }

                let distance = sqr_dst.sqrt();

                let slope = smoothing_kernel_derivative(distance, self.smoothing_radius);

                let other_vel = self.velocities[*particle_index];

                viscosity_force += (other_vel - vel) * slope;
            }
        }

        viscosity_force * self.viscosity
    }

    pub fn spatial_query(&self, pos: Vec2) -> Vec<(usize, usize)> {
        if self.positions.is_empty() {
            return vec![];
        }

        let center = pos_to_cell_coord(pos, self.smoothing_radius);
        let sqr_radius = self.smoothing_radius.squared();

        let mut particles = Vec::new();

        for offset in CELL_OFFSETS {
            let hash = hash_cell_coord(center + offset);
            let key = self.get_key_from_hash(hash);
            let start_index = self.start_indices[key];

            if start_index == usize::MAX {
                continue;
            }

            let mut indices = self.spatial_lookup[start_index..]
                .iter()
                .take_while(|(_, particle_cell_key)| *particle_cell_key == key)
                .filter(|(particle_index, _)| {
                    let other_pos = self.positions[*particle_index];

                    let sqr_dst = (other_pos - pos).length_squared();

                    sqr_dst < sqr_radius
                })
                .cloned()
                .collect::<Vec<_>>();

            particles.append(&mut indices);
        }

        particles
    }

    pub fn spatial_query_details(&self, pos: Vec2) -> (usize, Vec<(usize, usize)>) {
        if self.positions.is_empty() {
            return (0, vec![]);
        }

        let center = pos_to_cell_coord(pos, self.smoothing_radius);
        let sqr_radius = self.smoothing_radius.squared();

        let mut particles = Vec::new();

        let mut discarded_particles = 0;

        for offset in CELL_OFFSETS {
            let hash = hash_cell_coord(center + offset);
            let key = self.get_key_from_hash(hash);
            let start_index = self.start_indices[key];

            if start_index == usize::MAX {
                continue;
            }

            let mut indices = self.spatial_lookup[start_index..]
                .iter()
                .take_while(|(_, particle_cell_key)| *particle_cell_key == key)
                .filter(|(particle_index, _)| {
                    let other_pos = self.positions[*particle_index];

                    let sqr_dst = (other_pos - pos).length_squared();

                    if sqr_dst < sqr_radius {
                        true
                    } else {
                        discarded_particles += 1;
                        false
                    }
                })
                .cloned()
                .collect::<Vec<_>>();

            particles.append(&mut indices);
        }

        (discarded_particles, particles)
    }

    fn density_to_pressure(&self, density: f32) -> f32 {
        (self.target_density - density) * self.pressure_multiplier
    }

    fn near_density_to_pressure(&self, near_density: f32) -> f32 {
        near_density * self.near_pressure_multiplier
    }

    fn get_key_from_hash(&self, hash: usize) -> usize {
        hash % self.positions.len()
    }
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

#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn simulate(
    sim: ResMut<Sim>,
    time: Res<Time>,
    mut gizmos: Gizmos,
    q_camera: Query<(&Camera, &GlobalTransform), (With<Camera2d>, With<SimCamera>)>,
    q_window: Query<&Window, With<PrimaryWindow>>,
    interaction: Res<State<InteractionMode>>,
    mouse_settings: Res<InteractionSettings>,
    mut debug: ResMut<DebugData>,
    mut last_time: Local<u128>,
) {
    let Ok((cam, global_transform)) = q_camera.get_single() else {
        return;
    };
    let Ok(window) = q_window.get_single() else {
        return;
    };

    let delta = 1.0 / sim.delta;

    let gravity = sim.gravity;
    let smoothing_radius = sim.smoothing_radius;
    let num_particles = sim.positions.len();

    let sim_shared = Arc::new(Mutex::new(sim.into_inner()));

    // update cell keys and reset start indices
    (0..num_particles).into_par_iter().for_each(|i| {
        let mut sim = sim_shared.lock().unwrap();
        let cell_coord = pos_to_cell_coord(sim.positions[i], smoothing_radius);

        let cell_key = sim.get_key_from_hash(hash_cell_coord(cell_coord));

        sim.spatial_lookup[i] = (i, cell_key);
        sim.start_indices[i] = usize::MAX;
    });

    // sort
    sim_shared
        .lock()
        .unwrap()
        .spatial_lookup
        .sort_by(|(_, k1), (_, k2)| k1.cmp(k2));

    // set start indices
    for i in 0..num_particles {
        let mut sim = sim_shared.lock().unwrap();

        let curr_key = sim.spatial_lookup[i].1;
        let prev_key = if i == 0 {
            usize::MAX
        } else {
            sim.spatial_lookup[i - 1].1
        };

        if curr_key != prev_key {
            sim.start_indices[curr_key] = i;
        }
    }

    // gravity and predicted positions
    (0..num_particles).into_par_iter().for_each(|i| {
        let mut sim = sim_shared.lock().unwrap();

        sim.velocities[i] += gravity * delta;
        sim.predicted_positions[i] = sim.positions[i] + sim.velocities[i] * delta;
    });

    // density calculations
    (0..num_particles).into_par_iter().for_each(|i| {
        let mut sim = sim_shared.lock().unwrap();

        sim.densities[i] = sim.density_at_point(sim.predicted_positions[i]);
    });

    let mouse_pos_maybe = window
        .cursor_position()
        .map(|p| cam.viewport_to_world_2d(global_transform, p).unwrap());

    if *interaction != InteractionMode::None {
        if let Some(mouse_pos) = mouse_pos_maybe {
            gizmos.circle_2d(mouse_pos, mouse_settings.radius, color::LinearRgba::GREEN);
        }
    }

    // viscosity
    (0..num_particles).into_par_iter().for_each(|i| {
        let mut sim = sim_shared.lock().unwrap();

        let viscosity_force = sim.calculate_viscosity_force(i);

        sim.velocities[i] -= viscosity_force * delta;
    });

    // the actual sim
    (0..num_particles).into_par_iter().for_each(|i| {
        let mut sim = sim_shared.lock().unwrap();

        let (density, near_density) = sim.densities[i];

        let pressure_force = sim.calculate_pressure_force(i);

        let pressure_acceleration = pressure_force / density;

        sim.velocities[i] -= pressure_acceleration * delta;

        if debug.log_level == LogLevel::Always {
            info!(
                "pressure_acceleration for particle {i} is {pressure_acceleration}; pressure_force = {pressure_force}; density = {density}; near_density = {near_density}; velocity = {}",
                sim.velocities[i],
            );
        } else if debug.log_level == LogLevel::IllegalValues
            && (
            !pressure_force.is_finite()
            || !pressure_acceleration.is_finite()
            || !density.is_finite()
            || !near_density.is_finite()
        ) {
            warn!(
                "pressure_acceleration for particle {i} is {pressure_acceleration}; pressure_force = {pressure_force}; density = {density}; near_density = {near_density}; velocity = {}",
                sim.velocities[i],
            );
        }

        if let Some(mouse_pos) = mouse_pos_maybe {
            let distance = (sim.positions[i] - mouse_pos).length();
            let direction = (sim.positions[i] - mouse_pos) / distance;
            let slope = smoothing_kernel_derivative(distance, mouse_settings.radius);

            let strength = mouse_settings.force * mouse_settings.radius;

            let vel = -direction * slope * strength;

            sim.velocities[i] += match interaction.get() {
                InteractionMode::Attract => vel,
                InteractionMode::Repel => -vel,
                InteractionMode::None => Vec2::ZERO,
            };
        };
    });

    // apply velocities and collide with boundaries
    (0..num_particles).into_par_iter().for_each(|i| {
        let mut sim = sim_shared.lock().unwrap();

        let vel = sim.velocities[i];
        sim.positions[i] += vel * delta;

        let half_bounds_width = sim.bounds_size.x / 2.0 - sim.particle_radius;
        let half_bounds_height = sim.bounds_size.y / 2.0 - sim.particle_radius;

        if sim.positions[i].y < -half_bounds_height {
            sim.positions[i].y = -half_bounds_height;
            sim.velocities[i].y *= -0.8;
        }

        if sim.positions[i].y > half_bounds_height {
            sim.positions[i].y = half_bounds_height;
            sim.velocities[i].y *= -0.8;
        }

        if sim.positions[i].x < -half_bounds_width {
            sim.positions[i].x = -half_bounds_width;
            sim.velocities[i].x *= -0.8;
        }

        if sim.positions[i].x > half_bounds_width {
            sim.positions[i].x = half_bounds_width;
            sim.velocities[i].x *= -0.8;
        }
    });

    let current_time = time.elapsed().as_millis();
    debug.step_execution_time = current_time - *last_time;
    *last_time = current_time;
}

fn smoothing_kernel(distance: f32, radius: f32) -> f32 {
    if distance >= radius {
        return 0.0;
    }

    let volume = 6.0 / (PI * radius.powi(4));
    (radius - distance).squared() * volume
}

fn smoothing_kernel_derivative(distance: f32, radius: f32) -> f32 {
    if distance > radius {
        return 0.0;
    }

    let scale = 12.0 / (PI * radius.powi(4));

    scale * -(radius - distance)
}

fn spiky_kernel(distance: f32, radius: f32) -> f32 {
    if distance >= radius {
        return 0.0;
    }

    let volume = 10.0 / (PI * radius.powi(5));
    (radius - distance).powi(3) * volume
}

fn spiky_kernel_derivative(distance: f32, radius: f32) -> f32 {
    if distance >= radius {
        return 0.0;
    }

    let volume = 30.0 / (PI * radius.powi(5));
    (radius - distance).squared() * volume
}

pub fn pos_to_cell_coord(pos: Vec2, cell_size: f32) -> Vec2 {
    (pos / cell_size).floor()
}

pub fn hash_cell_coord(cell_coord: Vec2) -> usize {
    let x = Wrapping((cell_coord.x.floor() as i32) as usize);
    let y = Wrapping((cell_coord.y.floor() as i32) as usize);

    (x * Wrapping(101) + y * Wrapping(307)).0
}
