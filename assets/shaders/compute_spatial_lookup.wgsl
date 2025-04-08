@group(0) @binding(1) var<storage, read_write> positions: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> spatial_lookup: array<vec3u>;
@group(0) @binding(3) var<storage, read_write> start_indices: array<u32>;
@group(0) @binding(4) var<uniform> smoothing_radius: f32;
@group(0) @binding(5) var<uniform> num_particles: u32;

#import utils

@compute @workgroup_size(1)
fn compute_spatial_lookup(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let cell_coord = utils::pos_to_cell_coord(positions[global_id.x], smoothing_radius);

    let hash = utils::hash_cell_coord(cell_coord);

    let cell_key = utils::key_from_hash(hash, num_particles);

    spatial_lookup[global_id.x] = vec3u(global_id.x, hash, cell_key);
    start_indices[global_id.x] = num_particles;
}

