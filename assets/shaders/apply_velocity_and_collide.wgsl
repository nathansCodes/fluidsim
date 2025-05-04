@group(0) @binding(0) var<storage, read_write> positions: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec2f>;
@group(0) @binding(2) var<uniform> bounds_size: vec2f;
@group(0) @binding(3) var<uniform> particle_diameter: f32;
@group(0) @binding(4) var<uniform> delta: f32;

@compute @workgroup_size(64)
fn apply_velocity_and_collide(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vel = velocities[global_id.x];
    positions[global_id.x] += vel * delta;

    let half_bounds_width = bounds_size.x / 2.0 - particle_diameter / 2.0;
    let half_bounds_height = bounds_size.y / 2.0 - particle_diameter / 2.0;

    if positions[global_id.x].y < -half_bounds_height {
        positions[global_id.x].y = -half_bounds_height;
        velocities[global_id.x].y *= -0.8;
    }

    if positions[global_id.x].y > half_bounds_height {
        positions[global_id.x].y = half_bounds_height;
        velocities[global_id.x].y *= -0.8;
    }

    if positions[global_id.x].x < -half_bounds_width {
        positions[global_id.x].x = -half_bounds_width;
        velocities[global_id.x].x *= -0.8;
    }

    if positions[global_id.x].x > half_bounds_width {
        positions[global_id.x].x = half_bounds_width;
        velocities[global_id.x].x *= -0.8;
    }
}

