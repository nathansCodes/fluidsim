use std::sync::{Arc, Mutex};

use bevy::{
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        gpu_readback::{Readback, ReadbackComplete},
        render_asset::RenderAssets,
        render_graph::{self, RenderGraph, RenderLabel},
        render_resource::{
            binding_types::*, BindGroup, BindGroupEntries, BindGroupLayout, BindGroupLayoutEntries,
            BufferUsages, CachedComputePipelineId, ComputePassDescriptor,
            ComputePipelineDescriptor, IntoBinding, PipelineCache, PushConstantRange, ShaderStages,
            UniformBuffer,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        storage::{GpuShaderStorageBuffer, ShaderStorageBuffer},
        Render, RenderApp, RenderSet,
    },
};

use super::SimState;

fn sim_just_stopped(gpu_sim: Option<Res<GpuSim>>) -> bool {
    gpu_sim.is_some_and(|gpu_sim| {
        gpu_sim.state == SimState::Prepare && gpu_sim.old_state != SimState::Prepare
    })
}

fn sim_running_or_paused(gpu_sim: Option<Res<GpuSim>>) -> bool {
    gpu_sim.is_some_and(|gpu_sim| gpu_sim.state != SimState::Prepare)
}

// We need a plugin to organize all the systems and render node required for this example
pub struct SimComputePlugin;
impl Plugin for SimComputePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(ExtractResourcePlugin::<GpuSim>::default())
            .insert_resource(ClearColor(Color::BLACK))
            .add_systems(
                OnExit(SimState::Prepare),
                setup.run_if(in_state(super::Device::GPU)),
            )
            .add_systems(
                OnTransition {
                    exited: SimState::Running,
                    entered: SimState::Prepare,
                },
                cleanup.run_if(in_state(super::Device::GPU)),
            )
            .add_systems(
                PostUpdate,
                update.run_if(in_state(super::Device::GPU).and(resource_exists::<GpuSim>)),
            );
    }

    fn finish(&self, app: &mut App) {
        let render_app = app.sub_app_mut(RenderApp);

        render_app.add_systems(
            Render,
            (
                (|mut cmds: Commands| {
                    cmds.remove_resource::<GpuBufferBindGroups>();
                    cmds.remove_resource::<SimComputePipeline>();
                })
                .run_if(sim_just_stopped),
                (
                    (|mut cmds: Commands| {
                        cmds.init_resource::<SimComputePipeline>();
                    })
                    .run_if(not(resource_exists::<SimComputePipeline>)),
                    prepare_bind_groups,
                )
                    .in_set(RenderSet::PrepareBindGroups)
                    .run_if(not(resource_exists::<GpuBufferBindGroups>).and(sim_running_or_paused))
                    .chain(),
            )
                .in_set(RenderSet::PrepareBindGroups)
                .chain(),
        );

        // Add the compute node as a top level node to the render graph
        // This means it will only execute once per frame
        render_app
            .world_mut()
            .resource_mut::<RenderGraph>()
            .add_node(SimNodeLabel, SimNode);
    }
}

fn cleanup(mut commands: Commands, readback: Single<Entity, With<Readback>>) {
    commands.entity(readback.into_inner()).despawn();
}

#[derive(Resource, ExtractResource, Clone, Default)]
pub(super) struct GpuSim {
    positions: Handle<ShaderStorageBuffer>,
    predicted_positions: Handle<ShaderStorageBuffer>,
    velocities: Handle<ShaderStorageBuffer>,
    densities: Handle<ShaderStorageBuffer>,
    spatial_lookup: Handle<ShaderStorageBuffer>,
    start_indices: Handle<ShaderStorageBuffer>,
    gravity: Arc<Mutex<UniformBuffer<Vec2>>>,
    bounds_size: Arc<Mutex<UniformBuffer<Vec2>>>,
    particle_radius: Arc<Mutex<UniformBuffer<f32>>>,
    smoothing_radius: Arc<Mutex<UniformBuffer<f32>>>,
    target_density: Arc<Mutex<UniformBuffer<f32>>>,
    pressure_multiplier: Arc<Mutex<UniformBuffer<f32>>>,
    near_pressure_multiplier: Arc<Mutex<UniformBuffer<f32>>>,
    viscosity: Arc<Mutex<UniformBuffer<f32>>>,
    delta: Arc<Mutex<UniformBuffer<f32>>>,
    state: super::SimState,
    old_state: super::SimState,
    num_particles: u32,
}

fn setup(
    mut commands: Commands,
    mut buffers: ResMut<Assets<ShaderStorageBuffer>>,
    sim: Res<super::Sim>,
) {
    let empty_data = vec![0u32; sim.positions.len() * 2];

    let mut positions = ShaderStorageBuffer::from(sim.positions.clone());
    positions.buffer_description.usage |= BufferUsages::COPY_SRC;
    let positions = buffers.add(positions);

    let mut predicted_positions = ShaderStorageBuffer::from(empty_data.clone());
    predicted_positions.buffer_description.usage |= BufferUsages::COPY_SRC;
    let predicted_positions = buffers.add(predicted_positions);

    let mut velocities = ShaderStorageBuffer::from(empty_data.clone());
    velocities.buffer_description.usage |= BufferUsages::COPY_SRC;
    let velocities = buffers.add(velocities);

    let mut densities = ShaderStorageBuffer::from(empty_data);
    densities.buffer_description.usage |= BufferUsages::COPY_SRC;
    let densities = buffers.add(densities);

    let mut spatial_lookup = ShaderStorageBuffer::from(vec![UVec3::ZERO; sim.positions.len()]);
    spatial_lookup.buffer_description.usage |= BufferUsages::COPY_SRC;
    let spatial_lookup = buffers.add(spatial_lookup);

    let mut start_indices = ShaderStorageBuffer::from(vec![0u32; sim.positions.len()]);
    start_indices.buffer_description.usage |= BufferUsages::COPY_SRC;
    let start_indices = buffers.add(start_indices);

    commands
        .spawn(Readback::buffer(positions.clone()))
        .observe(on_readback_positions);

    commands.insert_resource(GpuSim {
        positions,
        predicted_positions,
        velocities,
        densities,
        spatial_lookup,
        start_indices,
        gravity: Arc::new(Mutex::new(UniformBuffer::from(sim.gravity))),
        bounds_size: Arc::new(Mutex::new(UniformBuffer::from(sim.bounds_size))),
        particle_radius: Arc::new(Mutex::new(UniformBuffer::from(sim.particle_radius))),
        smoothing_radius: Arc::new(Mutex::new(UniformBuffer::from(sim.smoothing_radius))),
        target_density: Arc::new(Mutex::new(UniformBuffer::from(sim.target_density))),
        pressure_multiplier: Arc::new(Mutex::new(UniformBuffer::from(sim.pressure_multiplier))),
        near_pressure_multiplier: Arc::new(Mutex::new(UniformBuffer::from(
            sim.near_pressure_multiplier,
        ))),
        viscosity: Arc::new(Mutex::new(UniformBuffer::from(sim.viscosity))),
        delta: Arc::new(Mutex::new(UniformBuffer::from(1.0 / sim.delta))),
        state: super::SimState::Running,
        old_state: super::SimState::Prepare,
        num_particles: sim.positions.len() as u32,
    });
}

fn on_readback_positions(
    trigger: Trigger<ReadbackComplete>,
    mut sim: ResMut<super::Sim>,
    state: Res<State<super::SimState>>,
) {
    if *state.get() == super::SimState::Paused {
        return;
    }

    let data: Vec<Vec2> = trigger.event().to_shader_type();

    sim.positions = data;
}

#[derive(Resource)]
struct GpuBufferBindGroups {
    compute_spatial_lookup: BindGroup,
    sort_spatial_lookup: BindGroup,
    set_start_indices: BindGroup,
    predict_positions: BindGroup,
    precalculate_densities: BindGroup,
    apply_viscosity: BindGroup,
    apply_pressure: BindGroup,
    apply_velocity_and_collide: BindGroup,
}

fn update(
    mut gpu_sim: ResMut<GpuSim>,
    state: Res<State<super::SimState>>,
    sim: Res<super::Sim>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    gpu_sim.old_state = gpu_sim.state.clone();
    gpu_sim.state = state.clone();

    let mut gravity = gpu_sim.gravity.lock().unwrap();
    let mut bounds_size = gpu_sim.bounds_size.lock().unwrap();
    let mut particle_radius = gpu_sim.particle_radius.lock().unwrap();
    let mut smoothing_radius = gpu_sim.smoothing_radius.lock().unwrap();
    let mut target_density = gpu_sim.target_density.lock().unwrap();
    let mut pressure_multiplier = gpu_sim.pressure_multiplier.lock().unwrap();
    let mut near_pressure_multiplier = gpu_sim.near_pressure_multiplier.lock().unwrap();
    let mut viscosity = gpu_sim.viscosity.lock().unwrap();
    let mut delta = gpu_sim.delta.lock().unwrap();

    gravity.set(sim.gravity);
    bounds_size.set(sim.bounds_size);
    particle_radius.set(sim.particle_radius);
    smoothing_radius.set(sim.smoothing_radius);
    target_density.set(sim.target_density);
    pressure_multiplier.set(sim.pressure_multiplier);
    near_pressure_multiplier.set(sim.near_pressure_multiplier);
    viscosity.set(sim.viscosity);
    delta.set(1.0 / sim.delta);

    gravity.write_buffer(&render_device, &render_queue);
    bounds_size.write_buffer(&render_device, &render_queue);
    particle_radius.write_buffer(&render_device, &render_queue);
    smoothing_radius.write_buffer(&render_device, &render_queue);
    target_density.write_buffer(&render_device, &render_queue);
    pressure_multiplier.write_buffer(&render_device, &render_queue);
    near_pressure_multiplier.write_buffer(&render_device, &render_queue);
    viscosity.write_buffer(&render_device, &render_queue);
    delta.write_buffer(&render_device, &render_queue);
}

fn prepare_bind_groups(
    mut commands: Commands,
    pipeline: Res<SimComputePipeline>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    gpu_sim: Option<Res<GpuSim>>,
    gpu_buffers: Res<RenderAssets<GpuShaderStorageBuffer>>,
) {
    let Some(gpu_sim) = gpu_sim else {
        return;
    };

    let positions = gpu_buffers.get(&gpu_sim.positions).unwrap();
    let predicted_positions = gpu_buffers.get(&gpu_sim.predicted_positions).unwrap();
    let velocities = gpu_buffers.get(&gpu_sim.velocities).unwrap();
    let densities = gpu_buffers.get(&gpu_sim.densities).unwrap();
    let spatial_lookup = gpu_buffers.get(&gpu_sim.spatial_lookup).unwrap();
    let start_indices = gpu_buffers.get(&gpu_sim.start_indices).unwrap();

    let mut num_particles = UniformBuffer::from(gpu_sim.num_particles);
    let mut gravity = gpu_sim.gravity.lock().unwrap();
    let mut bounds_size = gpu_sim.bounds_size.lock().unwrap();
    let mut particle_radius = gpu_sim.particle_radius.lock().unwrap();
    let mut smoothing_radius = gpu_sim.smoothing_radius.lock().unwrap();
    let mut target_density = gpu_sim.target_density.lock().unwrap();
    let mut pressure_multiplier = gpu_sim.pressure_multiplier.lock().unwrap();
    let mut near_pressure_multiplier = gpu_sim.near_pressure_multiplier.lock().unwrap();
    let mut viscosity = gpu_sim.viscosity.lock().unwrap();
    let mut delta = gpu_sim.delta.lock().unwrap();

    num_particles.write_buffer(&render_device, &render_queue);
    gravity.write_buffer(&render_device, &render_queue);
    bounds_size.write_buffer(&render_device, &render_queue);
    particle_radius.write_buffer(&render_device, &render_queue);
    smoothing_radius.write_buffer(&render_device, &render_queue);
    target_density.write_buffer(&render_device, &render_queue);
    pressure_multiplier.write_buffer(&render_device, &render_queue);
    near_pressure_multiplier.write_buffer(&render_device, &render_queue);
    viscosity.write_buffer(&render_device, &render_queue);
    delta.write_buffer(&render_device, &render_queue);

    commands.insert_resource(GpuBufferBindGroups {
        compute_spatial_lookup: render_device.create_bind_group(
            "compute_spatial_lookup",
            &pipeline.compute_spatial_lookup.layout,
            &BindGroupEntries::sequential((
                predicted_positions.buffer.as_entire_buffer_binding(),
                spatial_lookup.buffer.as_entire_buffer_binding(),
                start_indices.buffer.as_entire_buffer_binding(),
                smoothing_radius.into_binding(),
                num_particles.into_binding(),
            )),
        ),
        sort_spatial_lookup: render_device.create_bind_group(
            "sort_spatial_lookup",
            &pipeline.sort_spatial_lookup.layout,
            &BindGroupEntries::sequential((
                spatial_lookup.buffer.as_entire_buffer_binding(),
                num_particles.into_binding(),
            )),
        ),
        set_start_indices: render_device.create_bind_group(
            "set_start_indices",
            &pipeline.set_start_indices.layout,
            &BindGroupEntries::sequential((
                spatial_lookup.buffer.as_entire_buffer_binding(),
                start_indices.buffer.as_entire_buffer_binding(),
                num_particles.into_binding(),
            )),
        ),
        predict_positions: render_device.create_bind_group(
            "predict_positions",
            &pipeline.predict_positions.layout,
            &BindGroupEntries::sequential((
                positions.buffer.as_entire_buffer_binding(),
                predicted_positions.buffer.as_entire_buffer_binding(),
                velocities.buffer.as_entire_buffer_binding(),
                gravity.into_binding(),
                delta.into_binding(),
            )),
        ),
        precalculate_densities: render_device.create_bind_group(
            "precalculate_densities",
            &pipeline.precalculate_densities.layout,
            &BindGroupEntries::sequential((
                densities.buffer.as_entire_buffer_binding(),
                predicted_positions.buffer.as_entire_buffer_binding(),
                spatial_lookup.buffer.as_entire_buffer_binding(),
                start_indices.buffer.as_entire_buffer_binding(),
                smoothing_radius.into_binding(),
                num_particles.into_binding(),
            )),
        ),
        apply_viscosity: render_device.create_bind_group(
            "viscosity",
            &pipeline.apply_viscosity.layout,
            &BindGroupEntries::sequential((
                velocities.buffer.as_entire_buffer_binding(),
                predicted_positions.buffer.as_entire_buffer_binding(),
                spatial_lookup.buffer.as_entire_buffer_binding(),
                start_indices.buffer.as_entire_buffer_binding(),
                num_particles.into_binding(),
                smoothing_radius.into_binding(),
                viscosity.into_binding(),
                delta.into_binding(),
            )),
        ),
        apply_pressure: render_device.create_bind_group(
            "apply_pressure",
            &pipeline.apply_pressure.layout,
            &BindGroupEntries::sequential((
                predicted_positions.buffer.as_entire_buffer_binding(),
                velocities.buffer.as_entire_buffer_binding(),
                densities.buffer.as_entire_buffer_binding(),
                spatial_lookup.buffer.as_entire_buffer_binding(),
                start_indices.buffer.as_entire_buffer_binding(),
                num_particles.into_binding(),
                smoothing_radius.into_binding(),
                target_density.into_binding(),
                pressure_multiplier.into_binding(),
                near_pressure_multiplier.into_binding(),
                delta.into_binding(),
            )),
        ),
        apply_velocity_and_collide: render_device.create_bind_group(
            "apply_velocity_and_collide",
            &pipeline.apply_velocity_and_collide.layout,
            &BindGroupEntries::sequential((
                gpu_buffers
                    .get(&gpu_sim.positions)
                    .unwrap()
                    .buffer
                    .as_entire_buffer_binding(),
                gpu_buffers
                    .get(&gpu_sim.velocities)
                    .unwrap()
                    .buffer
                    .as_entire_buffer_binding(),
                bounds_size.into_binding(),
                particle_radius.into_binding(),
                delta.into_binding(),
            )),
        ),
    });
    commands.init_resource::<SimComputePipeline>();
}

// aka a compute pass
struct SimStep {
    pipeline: CachedComputePipelineId,
    layout: BindGroupLayout,
}

#[derive(Resource)]
struct SimComputePipeline {
    compute_spatial_lookup: SimStep,
    sort_spatial_lookup: SimStep,
    set_start_indices: SimStep,
    predict_positions: SimStep,
    precalculate_densities: SimStep,
    apply_viscosity: SimStep,
    apply_pressure: SimStep,
    apply_velocity_and_collide: SimStep,
}

impl FromWorld for SimComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let pipeline_cache = world.resource::<PipelineCache>();

        let compute_spatial_lookup_layout = render_device.create_bind_group_layout(
            Some("compute_spatial_lookup"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer::<Vec<Vec2>>(false),
                    storage_buffer::<Vec<UVec3>>(false),
                    storage_buffer::<Vec<u32>>(false),
                    uniform_buffer::<f32>(false),
                    uniform_buffer::<u32>(false),
                ),
            ),
        );
        let sort_spatial_lookup_layout = render_device.create_bind_group_layout(
            Some("sort_spatial_lookup"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer::<Vec<UVec3>>(false),
                    uniform_buffer::<u32>(false),
                ),
            ),
        );
        let set_start_indices_layout = render_device.create_bind_group_layout(
            Some("set_start_indices"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer::<Vec<UVec3>>(false),
                    storage_buffer::<Vec<u32>>(false),
                    uniform_buffer::<u32>(false),
                ),
            ),
        );
        let predict_positions_layout = render_device.create_bind_group_layout(
            Some("predict_positions"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer::<Vec<Vec2>>(false),
                    storage_buffer::<Vec<Vec2>>(false),
                    storage_buffer::<Vec<Vec2>>(false),
                    uniform_buffer::<Vec2>(false),
                    uniform_buffer::<f32>(false),
                ),
            ),
        );
        let precalculate_densities_layout = render_device.create_bind_group_layout(
            Some("precalculate_densities"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer::<Vec<Vec2>>(false),
                    storage_buffer::<Vec<Vec2>>(false),
                    storage_buffer::<Vec<UVec3>>(false),
                    storage_buffer::<Vec<u32>>(false),
                    uniform_buffer::<f32>(false),
                    uniform_buffer::<u32>(false),
                ),
            ),
        );
        let apply_viscosity_layout = render_device.create_bind_group_layout(
            Some("apply_viscosity"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer::<Vec<Vec2>>(false),
                    storage_buffer::<Vec<Vec2>>(false),
                    storage_buffer::<Vec<UVec3>>(false),
                    storage_buffer::<Vec<u32>>(false),
                    uniform_buffer::<u32>(false),
                    uniform_buffer::<f32>(false),
                    uniform_buffer::<f32>(false),
                    uniform_buffer::<f32>(false),
                ),
            ),
        );
        let apply_pressure_layout = render_device.create_bind_group_layout(
            Some("apply_pressure"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer::<Vec<Vec2>>(false),
                    storage_buffer::<Vec<Vec2>>(false),
                    storage_buffer::<Vec<Vec2>>(false),
                    storage_buffer::<Vec<UVec3>>(false),
                    storage_buffer::<Vec<u32>>(false),
                    uniform_buffer::<u32>(false),
                    uniform_buffer::<f32>(false),
                    uniform_buffer::<f32>(false),
                    uniform_buffer::<f32>(false),
                    uniform_buffer::<f32>(false),
                    uniform_buffer::<f32>(false),
                ),
            ),
        );
        let apply_velocity_and_collide_layout = render_device.create_bind_group_layout(
            Some("apply_velocity_and_collide"),
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer::<Vec<Vec2>>(false),
                    storage_buffer::<Vec<Vec2>>(false),
                    uniform_buffer::<Vec2>(false),
                    uniform_buffer::<f32>(false),
                    uniform_buffer::<f32>(false),
                ),
            ),
        );

        SimComputePipeline {
            compute_spatial_lookup: SimStep {
                pipeline: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("compute_spatial_lookup".into()),
                    layout: vec![compute_spatial_lookup_layout.clone()],
                    push_constant_ranges: Vec::new(),
                    shader: world.load_asset("shaders/compute_spatial_lookup.wgsl"),
                    shader_defs: Vec::new(),
                    entry_point: "compute_spatial_lookup".into(),
                    zero_initialize_workgroup_memory: false,
                }),
                layout: compute_spatial_lookup_layout,
            },
            sort_spatial_lookup: SimStep {
                pipeline: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("sort_spatial_lookup".into()),
                    layout: vec![sort_spatial_lookup_layout.clone()],
                    push_constant_ranges: vec![PushConstantRange {
                        stages: ShaderStages::COMPUTE,
                        range: (0..12),
                    }],
                    shader: world.load_asset("shaders/sort_spatial_lookup.wgsl"),
                    shader_defs: Vec::new(),
                    entry_point: "sort_spatial_lookup".into(),
                    zero_initialize_workgroup_memory: false,
                }),
                layout: sort_spatial_lookup_layout,
            },
            set_start_indices: SimStep {
                pipeline: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("set_start_indices".into()),
                    layout: vec![set_start_indices_layout.clone()],
                    push_constant_ranges: Vec::new(),
                    shader: world.load_asset("shaders/set_start_indices.wgsl"),
                    shader_defs: Vec::new(),
                    entry_point: "set_start_indices".into(),
                    zero_initialize_workgroup_memory: false,
                }),
                layout: set_start_indices_layout,
            },
            predict_positions: SimStep {
                pipeline: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("predict_positions".into()),
                    layout: vec![predict_positions_layout.clone()],
                    push_constant_ranges: Vec::new(),
                    shader: world.load_asset("shaders/predict_positions.wgsl"),
                    shader_defs: Vec::new(),
                    entry_point: "predict_positions".into(),
                    zero_initialize_workgroup_memory: false,
                }),
                layout: predict_positions_layout,
            },
            precalculate_densities: SimStep {
                pipeline: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("precalculate_densities".into()),
                    layout: vec![precalculate_densities_layout.clone()],
                    push_constant_ranges: Vec::new(),
                    shader: world.load_asset("shaders/precalculate_densities.wgsl"),
                    shader_defs: Vec::new(),
                    entry_point: "precalculate_densities".into(),
                    zero_initialize_workgroup_memory: false,
                }),
                layout: precalculate_densities_layout,
            },
            apply_viscosity: SimStep {
                pipeline: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("apply_viscosity".into()),
                    layout: vec![apply_viscosity_layout.clone()],
                    push_constant_ranges: Vec::new(),
                    shader: world.load_asset("shaders/apply_viscosity.wgsl"),
                    shader_defs: Vec::new(),
                    entry_point: "apply_viscosity".into(),
                    zero_initialize_workgroup_memory: false,
                }),
                layout: apply_viscosity_layout,
            },
            apply_pressure: SimStep {
                pipeline: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("apply_pressure".into()),
                    layout: vec![apply_pressure_layout.clone()],
                    push_constant_ranges: Vec::new(),
                    shader: world.load_asset("shaders/apply_pressure.wgsl"),
                    shader_defs: Vec::new(),
                    entry_point: "apply_pressure".into(),
                    zero_initialize_workgroup_memory: false,
                }),
                layout: apply_pressure_layout,
            },
            apply_velocity_and_collide: SimStep {
                pipeline: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("apply_velocity_and_collide".into()),
                    layout: vec![apply_velocity_and_collide_layout.clone()],
                    push_constant_ranges: Vec::new(),
                    shader: world.load_asset("shaders/apply_velocity_and_collide.wgsl"),
                    shader_defs: Vec::new(),
                    entry_point: "apply_velocity_and_collide".into(),
                    zero_initialize_workgroup_memory: false,
                }),
                layout: apply_velocity_and_collide_layout,
            },
        }
    }
}

/// Label to identify the node in the render graph
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct SimNodeLabel;

/// The node that will execute the compute shader
#[derive(Default)]
struct SimNode;

impl render_graph::Node for SimNode {
    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline_maybe = world.get_resource::<SimComputePipeline>();
        let bind_groups_maybe = world.get_resource::<GpuBufferBindGroups>();
        let gpu_sim_maybe = world.get_resource::<GpuSim>();

        let Some(pipeline) = pipeline_maybe else {
            return Ok(());
        };

        let Some(gpu_sim) = gpu_sim_maybe else {
            return Ok(());
        };

        if gpu_sim.state != super::SimState::Running && gpu_sim.state != super::SimState::Step {
            return Ok(());
        }

        let Some(bind_groups) = bind_groups_maybe else {
            return Ok(());
        };

        if let Some(pipeline) =
            pipeline_cache.get_compute_pipeline(pipeline.predict_positions.pipeline)
        {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: None,
                        ..default()
                    });

            pass.set_bind_group(0, &bind_groups.predict_positions, &[]);
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups((gpu_sim.num_particles as f32 / 64.0).ceil() as u32, 1, 1);
        }
        if let Some(pipeline) =
            pipeline_cache.get_compute_pipeline(pipeline.compute_spatial_lookup.pipeline)
        {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: None,
                        ..default()
                    });

            pass.set_bind_group(0, &bind_groups.compute_spatial_lookup, &[]);
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups((gpu_sim.num_particles as f32 / 64.0).ceil() as u32, 1, 1);
        }
        if let Some(pipeline) =
            pipeline_cache.get_compute_pipeline(pipeline.sort_spatial_lookup.pipeline)
        {
            let num_stages = gpu_sim.num_particles.next_power_of_two().ilog2();

            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: None,
                        ..default()
                    });

            pass.set_bind_group(0, &bind_groups.sort_spatial_lookup, &[]);
            pass.set_pipeline(pipeline);

            for stage_index in 0..num_stages {
                for step_index in 0..stage_index + 1 {
                    let group_width: u32 = 1 << (stage_index - step_index);
                    let group_height: u32 = 2 * group_width - 1;

                    pass.set_push_constants(0, &group_width.to_ne_bytes());
                    pass.set_push_constants(4, &group_height.to_ne_bytes());
                    pass.set_push_constants(8, &step_index.to_ne_bytes());

                    pass.dispatch_workgroups(
                        (gpu_sim.num_particles.next_power_of_two() as f32 / 256.0).ceil() as u32,
                        1,
                        1,
                    );
                }
            }
        }
        if let Some(pipeline) =
            pipeline_cache.get_compute_pipeline(pipeline.set_start_indices.pipeline)
        {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: None,
                        ..default()
                    });

            pass.set_bind_group(0, &bind_groups.set_start_indices, &[]);
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups((gpu_sim.num_particles as f32 / 64.0).ceil() as u32, 1, 1);
        }
        if let Some(pipeline) =
            pipeline_cache.get_compute_pipeline(pipeline.precalculate_densities.pipeline)
        {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: None,
                        ..default()
                    });

            pass.set_bind_group(0, &bind_groups.precalculate_densities, &[]);
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups((gpu_sim.num_particles as f32 / 64.0).ceil() as u32, 1, 1);
        }
        if let Some(pipeline) =
            pipeline_cache.get_compute_pipeline(pipeline.apply_viscosity.pipeline)
        {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: None,
                        ..default()
                    });

            pass.set_bind_group(0, &bind_groups.apply_viscosity, &[]);
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups((gpu_sim.num_particles as f32 / 64.0).ceil() as u32, 1, 1);
        }
        if let Some(pipeline) =
            pipeline_cache.get_compute_pipeline(pipeline.apply_pressure.pipeline)
        {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: None,
                        ..default()
                    });

            pass.set_bind_group(0, &bind_groups.apply_pressure, &[]);
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups((gpu_sim.num_particles as f32 / 64.0).ceil() as u32, 1, 1);
        }
        if let Some(pipeline) =
            pipeline_cache.get_compute_pipeline(pipeline.apply_velocity_and_collide.pipeline)
        {
            let mut pass =
                render_context
                    .command_encoder()
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: None,
                        ..default()
                    });

            pass.set_bind_group(0, &bind_groups.apply_velocity_and_collide, &[]);
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups((gpu_sim.num_particles as f32 / 64.0).ceil() as u32, 1, 1);
        }
        Ok(())
    }
}
