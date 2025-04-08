@group(0) @binding(1) var<storage, read_write> positions: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec2f>;
@group(0) @binding(3) var<uniform> bounds_size: vec2f;
@group(0) @binding(4) var<uniform> particle_radius: f32;
@group(0) @binding(5) var<uniform> delta: f32;

@compute @workgroup_size(1)
fn apply_velocity_and_collide(@builtin(global_invocation_id) global_id: vec3<u32>) {
}

