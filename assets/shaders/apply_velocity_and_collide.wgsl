@group(0) @binding(0) var<storage, read_write> positions: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec2f>;
@group(0) @binding(2) var<uniform> bounds_size: vec2f;
@group(0) @binding(3) var<uniform> particle_diameter: f32;
@group(0) @binding(4) var<uniform> delta: f32;
@group(0) @binding(5) var<uniform> planetary_gravity: u32;
@group(0) @binding(6) var<uniform> planet_position: vec2f;
@group(0) @binding(7) var<uniform> planet_radius: f32;

@compute @workgroup_size(64)
fn apply_velocity_and_collide(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let vel = velocities[global_id.x];
    positions[global_id.x] += vel * delta;

    let half_bounds_width = bounds_size.x / 2.0 - particle_diameter / 2.0;
    let half_bounds_height = bounds_size.y / 2.0 - particle_diameter / 2.0;

    if planetary_gravity != 0 {
        let relative_position = positions[global_id.x] - planet_position;
        let sqr_dst = pow(relative_position.x, 2.0) + pow(relative_position.y, 2.0);

        if sqr_dst < planet_radius * planet_radius {
            let direction = relative_position / sqrt(sqr_dst);
            let vel = velocities[global_id.x];
            velocities[global_id.x] = (vel - 2 * dot(vel, direction) * direction) * 0.8;
            positions[global_id.x] = planet_position + direction * planet_radius;
        }
    }

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

