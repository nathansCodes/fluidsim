@group(0) @binding(0) var<storage, read_write> predicted_positions: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> densities: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> spatial_lookup: array<Entry>;
@group(0) @binding(4) var<storage, read_write> start_indices: array<u32>;
@group(0) @binding(5) var<uniform> num_particles: u32;
@group(0) @binding(6) var<uniform> smoothing_radius: f32;
@group(0) @binding(7) var<uniform> target_density: f32;
@group(0) @binding(8) var<uniform> pressure_multiplier: f32;
@group(0) @binding(9) var<uniform> near_pressure_multiplier: f32;
@group(0) @binding(10) var<uniform> delta: f32;

#import "shaders/utils.wgsl" as utils

struct Entry {
    original_index: u32,
	hash: u32,
	key: u32,
}

var<private> CELL_OFFSETS = array(
    vec2i(-1, -1),
    vec2i(0, -1),
    vec2i(1, -1),
    vec2i(-1, 0),
    vec2i(0, 0),
    vec2i(1, 0),
    vec2i(-1, 1),
    vec2i(0, 1),
    vec2i(1, 1)
);

@compute @workgroup_size(64)
fn apply_pressure(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var pressure_force = vec2f(0, 0);

    let particle = global_id.x;

    let density = densities[particle].x;
    let near_density = densities[particle].y;

    let point = predicted_positions[particle];
    let pressure = density_to_pressure(density);
    let near_pressure = near_density_to_pressure(near_density);

    let center = utils::pos_to_cell_coord(point, smoothing_radius);
    let sqr_radius = smoothing_radius * smoothing_radius;

    for (var i = 0u; i < 9u; i++) {
        let offset = CELL_OFFSETS[i];
        let hash = utils::hash_cell_coord(center + offset);
        let key = utils::key_from_hash(hash, num_particles);
        let start_index = start_indices[key];

        if start_index == num_particles {
            continue;
        }

        for (var j = start_index; j < num_particles; j++) {
            let other_index = spatial_lookup[j].original_index;
            let other_cell_key = spatial_lookup[j].key;

            if other_cell_key != key {
                break;
            }
            if spatial_lookup[j].hash != hash {
                continue;
            }

            let other_point = predicted_positions[other_index];

            let offset = predicted_positions[other_index] - point;
            let sqr_dst = dot(offset, offset);

            if sqr_dst > sqr_radius || other_index == particle {
                continue;
            }

            let dst = sqrt(sqr_dst);

            let direction = offset / dst;

            let other_density = densities[other_index].x;
            let other_near_density = densities[other_index].y;

            let shared_pressure = (pressure + density_to_pressure(other_density)) / 2.0;
            let shared_near_pressure =
                (near_pressure + near_density_to_pressure(other_near_density)) / 2.0;

            let force = utils::smoothing_kernel_derivative(dst, smoothing_radius);
            let near_force = utils::spiky_kernel_derivative(dst, smoothing_radius);

            pressure_force += shared_pressure * direction * force / other_density;
            pressure_force +=
                shared_near_pressure * direction * near_force / other_near_density;
        }
    }

    let pressure_acceleration = pressure_force / density;

    velocities[particle] -= pressure_acceleration * delta;
}

fn density_to_pressure(density: f32) -> f32 {
    return (target_density - density) * pressure_multiplier;
}

fn near_density_to_pressure(near_density: f32) -> f32 {
    return near_density * near_pressure_multiplier;
}

