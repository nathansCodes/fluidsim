mod controls;
mod debug;
mod sim;
mod ui;

use bevy::color::palettes::basic::*;
use bevy::math::FloatPow;
use bevy::render::gpu_readback::GpuReadbackPlugin;
use bevy::{app::App, prelude::Component, DefaultPlugins};
use bevy::{color, prelude::*};

use bevy_simple_compute::prelude::{AppComputePlugin, AppComputeWorker, AppComputeWorkerPlugin};
use controls::ControlsPlugin;
use debug::{debug_overlay, DebugData, DebugOverlay, ParticleColoring};
use sim::gpu::SimComputeWorker;
use sim::{cpu, gpu, Device, Sim};
use ui::UiPlugin;

fn main() {
    App::new().add_plugins((DefaultPlugins, SimPlugin)).run();
}

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
struct SpawnInfo {
    num_particles: usize,
    center: Vec2,
    spacing: f32,
}

impl Default for SpawnInfo {
    fn default() -> Self {
        Self {
            num_particles: 1500,
            center: Vec2::ZERO,
            spacing: 0.15,
        }
    }
}

#[derive(Event)]
enum SimEvents {
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
                let density = sim.density_at_point(*pos).0;
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
                sim.velocities.clear();
                sim.positions.clear();
                sim.predicted_positions.clear();
                sim.densities.clear();
                sim.spatial_lookup.clear();
                sim.start_indices.clear();
            }
            SimEvents::StartSimEvent(info, start_paused) => {
                sim.velocities.resize(info.num_particles, Vec2::ZERO);
                sim.positions.resize(info.num_particles, Vec2::ZERO);
                sim.predicted_positions
                    .resize(info.num_particles, Vec2::ZERO);
                sim.densities.resize(info.num_particles, (0.0, 0.0));
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

struct SimPlugin;

impl Plugin for SimPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(AppComputePlugin)
            .add_plugins(AppComputeWorkerPlugin::<SimComputeWorker>::default())
            .insert_state(SimState::default())
            .insert_state(sim::Device::CPU)
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
                            cpu::simulate.run_if(in_state(sim::Device::CPU)),
                            gpu::simulate.run_if(in_state(sim::Device::GPU)),
                            gpu::read_data.run_if(in_state(sim::Device::GPU)),
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
            .add_systems(
                OnExit(SimState::Prepare),
                gpu::start.run_if(in_state(Device::GPU)),
            )
            .add_plugins((ControlsPlugin, UiPlugin));
    }
}
