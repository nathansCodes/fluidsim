@group(0) @binding(0) var<storage, read_write> positions: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> predicted_positions: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec2f>;
@group(0) @binding(3) var<uniform> gravity: vec2f;
@group(0) @binding(4) var<uniform> delta: f32;

@compute @workgroup_size(64)
fn external_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    velocities[global_id.x] += gravity * delta;
    predicted_positions[global_id.x] = positions[global_id.x] + velocities[global_id.x] * delta;
}

