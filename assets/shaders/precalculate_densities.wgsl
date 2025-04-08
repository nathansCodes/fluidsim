@group(0) @binding(0) var<storage, read_write> densities: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> predicted_positions: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> spatial_lookup: array<vec3u>;
@group(0) @binding(3) var<storage, read_write> start_indices: array<u32>;
@group(0) @binding(4) var<uniform> smoothing_radius: f32;
@group(0) @binding(5) var<uniform> num_particles: u32;

#import utils

@compute @workgroup_size(1)
fn precalculate_densities(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var density = 0.0;
    var near_density = 0.0;

    let point = predicted_positions[global_id.x];

    let center = utils::pos_to_cell_coord(point, smoothing_radius);
    let sqr_radius = smoothing_radius * smoothing_radius;

    for (var i = 0u; i < arrayLength(CELL_OFFSETS); i++) {
        let offset = utils::CELL_OFFSETS[i];
        let hash = utils::hash_cell_coord(center + offset);
        let key = utils::key_from_hash(hash, num_particles);
        let start_index = start_indices[key];

        if start_index == num_particles {
            continue;
        }

        for (var j = start_index; j < num_particles; j++) {
            let particle_index = spatial_lookup[j].z;
            let particle_cell_key = spatial_lookup[j].x;

            if particle_cell_key != key {
                break;
            }

            let dst = length(predicted_positions[particle_index] - point);

            if dst * dst > sqr_radius {
                continue;
            }

            density += utils::smoothing_kernel(dst, smoothing_radius);
            near_density += utils::spiky_kernel(dst, smoothing_radius);
        }
    }

    densities[global_id.x] = vec2f(density, near_density);
}

