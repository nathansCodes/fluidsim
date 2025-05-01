@group(0) @binding(0) var<storage, read_write> velocities: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> predicted_positions: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> spatial_lookup: array<Entry>;
@group(0) @binding(3) var<storage, read_write> start_indices: array<u32>;
@group(0) @binding(4) var<uniform> num_particles: u32;
@group(0) @binding(5) var<uniform> smoothing_radius: f32;
@group(0) @binding(6) var<uniform> viscosity: f32;
@group(0) @binding(7) var<uniform> delta: f32;

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
fn apply_viscosity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var viscosity_force = vec2f(0, 0);

    let particle = global_id.x;

    let point = predicted_positions[particle];
    let vel = velocities[particle];

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

            let offset = predicted_positions[other_index] - point;
            let sqr_dst = dot(offset, offset);

            if sqr_dst > sqr_radius || other_index == particle {
                continue;
            }

            let dst = sqrt(sqr_dst);

            let slope = utils::smoothing_kernel_derivative(dst, smoothing_radius);

            let other_vel = velocities[other_index];

            viscosity_force += (other_vel - vel) * slope;
        }
    }

    velocities[particle] -= viscosity_force * viscosity * delta;
}

