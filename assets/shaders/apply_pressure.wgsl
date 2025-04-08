@group(0) @binding(1) var<storage, read_write> predicted_positions: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> densities: array<vec2f>;
@group(0) @binding(4) var<storage, read_write> spatial_lookup: array<vec3u>;
@group(0) @binding(5) var<storage, read_write> start_indices: array<u32>;
@group(0) @binding(6) var<uniform> smoothing_radius: f32;
@group(0) @binding(7) var<uniform> target_density: f32;
@group(0) @binding(8) var<uniform> pressure_multiplier: f32;
@group(0) @binding(9) var<uniform> near_pressure_multiplier: f32;
@group(0) @binding(10) var<uniform> delta: f32;

@compute @workgroup_size(1)
fn apply_pressure(@builtin(global_invocation_id) global_id: vec3<u32>) {
}

