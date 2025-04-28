pub mod cpu;
pub mod gpu;

pub(super) use cpu::pos_to_cell_coord;

use bevy::{
    color::{self, palettes::basic::*},
    diagnostic::FrameTimeDiagnosticsPlugin,
    math::FloatPow,
    prelude::*,
    render::extract_resource::ExtractResource,
};
use gpu::GpuSim;

use crate::{
    controls::ControlsPlugin,
    debug::{debug_overlay, DebugData, DebugOverlay, ParticleColoring},
    ui,
};

#[derive(Component)]
#[require(Mesh2d, Transform)]
struct Particle;

#[derive(States, Clone, PartialEq, Eq, Hash, Debug, Default)]
pub enum SimState {
    #[default]
    Prepare,
    Running,
    Paused,
    Step,
}

#[derive(Debug, Clone, Copy)]
pub struct SpawnInfo {
    pub num_particles: usize,
    pub center: Vec2,
    pub spacing: f32,
}

impl Default for SpawnInfo {
    fn default() -> Self {
        Self {
            num_particles: 1024,
            center: Vec2::ZERO,
            spacing: 0.1,
        }
    }
}

#[derive(Event)]
pub enum SimEvents {
    StartSimEvent(SpawnInfo, bool),
    ResetSim,
}

fn update_particles(
    mut cmds: Commands,
    sim: Res<Sim>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut q_particles: Query<
        (Entity, &mut Transform, &MeshMaterial2d<ColorMaterial>),
        With<Particle>,
    >,
    debug_data: Res<DebugData>,
    mut gizmos: Gizmos,
) {
    let mut particles = q_particles.iter_mut().collect::<Vec<_>>();

    for (i, (pos, vel)) in sim.positions.iter().zip(&sim.velocities).enumerate() {
        let color: Color = match debug_data.particle_colors {
            ParticleColoring::Density => {
                let density = sim.density_at_point(*pos).x;
                color::Srgba::BLUE
                    .mix(&color::Srgba::WHITE, density / sim.target_density)
                    .mix(
                        &color::Srgba::RED,
                        (density / sim.target_density - 1.0).clamp(0.0, 1.0),
                    )
                    .into()
            }
            ParticleColoring::Velocity => {
                let factor = (vel.length_squared() / 8.0_f32.squared()).clamp(0.0, 10.0);
                Oklaba::from(LinearRgba::from(BLUE))
                    .mix(&LinearRgba::from(TEAL).into(), (factor / 2.0).min(1.0))
                    .mix(
                        &LinearRgba::from(YELLOW).into(),
                        ((factor - 2.0) / 4.0).clamp(0.0, 1.0),
                    )
                    .mix(
                        &LinearRgba::from(RED).into(),
                        ((factor - 6.0) / 4.0).clamp(0.0, 1.0),
                    )
                    .into()
            }
        };

        // let color = Color::linear_rgba(sim.densities[i].x, 0.0, 0.0, 1.0);

        if let Some((_, transform, material_handle)) = particles.get_mut(i) {
            transform.translation = pos.extend(0.0);
            transform.scale = Vec3::new(sim.particle_radius, sim.particle_radius, 1.0);

            materials.get_mut(*material_handle).unwrap().color = color;
            if debug_data
                .debug_overlay
                .intersects(DebugOverlay::VelocityArrows)
            {
                gizmos.arrow_2d(*pos, pos + vel / sim.delta * 10.0, color);
            }
        } else {
            cmds.spawn((
                Particle,
                Mesh2d(meshes.add(Circle::default())),
                MeshMaterial2d(materials.add(Color::LinearRgba(color::LinearRgba::gray(0.7)))),
                Transform::from_xyz(pos.x, pos.y, 0.0).with_scale(Vec3::new(
                    sim.particle_radius,
                    sim.particle_radius,
                    1.0,
                )),
            ));
        };
    }

    if particles.len() > sim.positions.len() {
        particles[sim.positions.len()..]
            .iter()
            .for_each(|(e, ..)| cmds.entity(*e).despawn());
    }
}

fn recieve_sim_events(
    mut evr: EventReader<SimEvents>,
    mut sim: ResMut<Sim>,
    mut next_state: ResMut<NextState<SimState>>,
    particles: Query<Entity, With<Particle>>,
    mut cmds: Commands,
) {
    for ev in evr.read() {
        match ev {
            SimEvents::ResetSim => {
                for e in particles.iter() {
                    cmds.entity(e).despawn();
                }
                next_state.set(SimState::Prepare);
                sim.positions.clear();
                sim.predicted_positions.clear();
                sim.velocities.clear();
                sim.densities.clear();
                sim.spatial_lookup.clear();
                sim.start_indices.clear();
            }
            SimEvents::StartSimEvent(info, start_paused) => {
                sim.velocities.resize(info.num_particles, Vec2::ZERO);
                sim.positions.resize(info.num_particles, Vec2::ZERO);
                sim.predicted_positions
                    .resize(info.num_particles, Vec2::ZERO);
                sim.densities.resize(info.num_particles, Vec2::ZERO);
                sim.spatial_lookup.resize(info.num_particles, (0, 0));
                sim.start_indices.resize(info.num_particles, usize::MAX);

                let particles_per_row = info.num_particles.isqrt() as f32;
                let particles_per_col = (info.num_particles as f32 - 1.0) / particles_per_row + 1.0;

                let spacing = sim.particle_radius * 2.0 + info.spacing;

                for i in 0..info.num_particles {
                    let x =
                        (i as f32 % particles_per_row - particles_per_row / 2.0 + 0.5) * spacing;
                    let y =
                        (i as f32 / particles_per_col - particles_per_col / 2.0 + 0.5) * spacing;

                    sim.positions[i] = Vec2::new(x, y) + info.center;
                }

                if *start_paused {
                    next_state.set(SimState::Step);
                } else {
                    next_state.set(SimState::Running);
                }
            }
        }
    }
}

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

#[derive(Resource, ExtractResource, Clone)]
pub struct Sim {
    pub positions: Vec<Vec2>,
    pub predicted_positions: Vec<Vec2>,
    pub velocities: Vec<Vec2>,
    pub densities: Vec<Vec2>,
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

pub struct SimPlugin;

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        app.insert_state(SimState::default())
            .insert_state(Device::CPU)
            .init_resource::<Sim>()
            .init_resource::<DebugData>()
            .add_systems(
                // FixedUpdate,
                Update,
                (
                    recieve_sim_events,
                    // simulate
                    (
                        (
                            cpu::simulate.run_if(in_state(Device::CPU)),
                            update_particles,
                        )
                            .chain()
                            .run_if(in_state(SimState::Running).or(in_state(SimState::Step))),
                        debug_overlay.pipe(ui::debug_info),
                    )
                        .chain(),
                    (|mut next: ResMut<NextState<SimState>>| next.set(SimState::Paused))
                        .run_if(in_state(SimState::Step))
                        .after(update_particles),
                ),
            )
            // .add_systems(Update, update_particles)
            .add_plugins((gpu::SimComputePlugin, FrameTimeDiagnosticsPlugin));
    }
}
