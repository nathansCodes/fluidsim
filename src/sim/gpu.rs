use std::time::Duration;

use bevy::prelude::*;
use bevy_simple_compute::prelude::*;

use super::Sim;

mod pass {
    use bevy::reflect::TypePath;
    use bevy_simple_compute::prelude::{ComputeShader, ShaderRef};

    #[derive(TypePath)]
    pub(super) struct ComputeSpatialLookup;

    impl ComputeShader for ComputeSpatialLookup {
        fn shader() -> ShaderRef {
            "shaders/compute_spatial_lookup.wgsl".into()
        }

        fn entry_point<'a>() -> &'a str {
            "compute_spatial_lookup"
        }
    }

    #[derive(TypePath)]
    pub(super) struct SortSpatialLookup;

    impl ComputeShader for SortSpatialLookup {
        fn shader() -> ShaderRef {
            "shaders/sort_spatial_lookup.wgsl".into()
        }

        fn entry_point<'a>() -> &'a str {
            "sort_spatial_lookup"
        }
    }

    #[derive(TypePath)]
    pub(super) struct SetStartIndices;

    impl ComputeShader for SetStartIndices {
        fn shader() -> ShaderRef {
            "shaders/set_start_indices.wgsl".into()
        }

        fn entry_point<'a>() -> &'a str {
            "set_start_indices"
        }
    }

    #[derive(TypePath)]
    pub(super) struct PredictPositions;

    impl ComputeShader for PredictPositions {
        fn shader() -> ShaderRef {
            "shaders/predict_positions.wgsl".into()
        }

        fn entry_point<'a>() -> &'a str {
            "predict_positions"
        }
    }

    #[derive(TypePath)]
    pub(super) struct PrecalculateDensities;

    impl ComputeShader for PrecalculateDensities {
        fn shader() -> ShaderRef {
            "shaders/precalculate_densities.wgsl".into()
        }

        fn entry_point<'a>() -> &'a str {
            "precalculate_densities"
        }
    }

    #[derive(TypePath)]
    pub(super) struct ApplyViscosity;

    impl ComputeShader for ApplyViscosity {
        fn shader() -> ShaderRef {
            "shaders/apply_viscosity.wgsl".into()
        }

        fn entry_point<'a>() -> &'a str {
            "apply_viscosity"
        }
    }

    #[derive(TypePath)]
    pub(super) struct ApplyPressure;

    impl ComputeShader for ApplyPressure {
        fn shader() -> ShaderRef {
            "shaders/apply_pressure.wgsl".into()
        }

        fn entry_point<'a>() -> &'a str {
            "apply_pressure"
        }
    }

    #[derive(TypePath)]
    pub(super) struct ApplyVelocitiesAndCollide;

    impl ComputeShader for ApplyVelocitiesAndCollide {
        fn shader() -> ShaderRef {
            "shaders/apply_velocity_and_collide.wgsl".into()
        }

        fn entry_point<'a>() -> &'a str {
            "apply_velocity_and_collide"
        }
    }
}

pub struct SimComputeWorker;

impl ComputeWorker for SimComputeWorker {
    fn build(world: &mut World) -> AppComputeWorker<Self> {
        world.resource_scope(|world, sim: Mut<Sim>| {
            let buffer_size_1x32bit = (sim.positions.len() * size_of::<u32>()) as u64;
            let buffer_size_2x32bit = buffer_size_1x32bit * 2;
            let buffer_size_3x32bit = buffer_size_1x32bit * 3;

            let mut worker_builder = AppComputeWorkerBuilder::new(world);

            let num_stages = sim.positions.len().next_power_of_two().ilog2();

            worker_builder
                .add_staging("positions", &sim.positions)
                .add_empty_staging("predicted_positions", buffer_size_2x32bit)
                .add_staging("velocities", &sim.velocities)
                .add_empty_staging("densities", buffer_size_2x32bit)
                .add_empty_staging("spatial_lookup", buffer_size_3x32bit)
                .add_empty_staging("start_indices", buffer_size_1x32bit)
                .add_uniform("gravity", &sim.gravity)
                .add_uniform("bounds_size", &sim.bounds_size)
                .add_uniform("particle_radius", &sim.particle_radius)
                .add_uniform("smoothing_radius", &sim.smoothing_radius)
                .add_uniform("target_density", &sim.target_density)
                .add_uniform("pressure_multiplier", &sim.pressure_multiplier)
                .add_uniform("near_pressure_multiplier", &sim.near_pressure_multiplier)
                .add_uniform("viscosity", &sim.viscosity)
                .add_uniform("delta", &sim.delta)
                .add_uniform("num_particles", &(sim.positions.len() as u32))
                .add_staging("positions", &sim.positions)
                .add_uniform("num_stages", &num_stages)
                .add_rw_storage("stage_index", &0u32)
                .add_rw_storage("step_index", &0u32);

            worker_builder.add_pass::<pass::ComputeSpatialLookup>(
                [sim.positions.len() as u32, 1, 1],
                &[
                    "positions",
                    "spatial_lookup",
                    "start_indices",
                    "smoothing_radius",
                    "num_particles",
                ],
            );

            for stage_index in 0..num_stages {
                for _ in 0..stage_index + 1 {
                    worker_builder.add_pass::<pass::SortSpatialLookup>(
                        [sim.positions.len().next_power_of_two() as u32 / 2, 1, 1],
                        &[
                            "spatial_lookup",
                            "num_particles",
                            "num_stages",
                            "stage_index",
                            "step_index",
                        ],
                    );
                }
            }

            worker_builder
                .add_pass::<pass::SetStartIndices>(
                    [sim.positions.len() as u32, 1, 1],
                    &["spatial_lookup", "start_indices", "num_particles"],
                )
                .add_pass::<pass::PredictPositions>(
                    [sim.positions.len() as u32, 1, 1],
                    &[
                        "predicted_positions",
                        "positions",
                        "velocities",
                        "gravity",
                        "delta",
                    ],
                )
                .add_pass::<pass::PrecalculateDensities>(
                    [sim.positions.len() as u32, 1, 1],
                    &[
                        "densities",
                        "predicted_positions",
                        "spatial_lookup",
                        "start_indices",
                        "smoothing_radius",
                        "num_particles",
                    ],
                )
                .add_pass::<pass::ApplyViscosity>(
                    [sim.positions.len() as u32, 1, 1],
                    &[
                        "velocities",
                        "predicted_positions",
                        "spatial_lookup",
                        "start_indices",
                        "smoothing_radius",
                        "viscosity",
                        "num_particles",
                        "delta",
                    ],
                )
                .add_pass::<pass::ApplyPressure>(
                    [sim.positions.len() as u32, 1, 1],
                    &[
                        "predicted_positions",
                        "velocities",
                        "densities",
                        "spatial_lookup",
                        "start_indices",
                        "smoothing_radius",
                        "target_density",
                        "pressure_multiplier",
                        "near_pressure_multiplier",
                        "delta",
                        "num_particles",
                    ],
                )
                .add_pass::<pass::ApplyVelocitiesAndCollide>(
                    [sim.positions.len() as u32, 1, 1],
                    &[
                        "positions",
                        "velocities",
                        "bounds_size",
                        "particle_radius",
                        "delta",
                    ],
                )
                .one_shot()
                .build()
        })
    }
}

pub fn start(world: &mut World) {
    world.remove_resource::<AppComputeWorker<SimComputeWorker>>();
    let worker = SimComputeWorker::build(world);
    world.insert_resource(worker);
}

pub fn simulate(mut compute_worker: ResMut<AppComputeWorker<SimComputeWorker>>, sim: Res<Sim>) {
    compute_worker.write_slice("positions", &sim.positions);
    compute_worker.write_slice("velocities", &sim.velocities);
    compute_worker.write("gravity", &sim.gravity);
    compute_worker.write("bounds_size", &sim.bounds_size);
    compute_worker.write("particle_radius", &sim.particle_radius);
    compute_worker.write("smoothing_radius", &sim.smoothing_radius);
    compute_worker.write("bounds_size", &sim.bounds_size);
    compute_worker.write("target_density", &sim.target_density);
    compute_worker.write("pressure_multiplier", &sim.pressure_multiplier);
    compute_worker.write("near_pressure_multiplier", &sim.near_pressure_multiplier);
    compute_worker.write("viscosity", &sim.viscosity);
    compute_worker.write("delta", &sim.delta);

    compute_worker.execute();
}

pub fn read_data(compute_worker: Res<AppComputeWorker<SimComputeWorker>>, mut sim: ResMut<Sim>) {
    if !compute_worker.ready() {
        return;
    };

    let result: Vec<Vec2> = compute_worker.read_vec("positions");

    if sim.positions.len() != result.len() {
        return;
    }

    sim.positions.copy_from_slice(result.as_slice());
}
