@group(0) @binding(0) var<storage, read_write> predicted_positions: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> spatial_lookup: array<Entry>;
@group(0) @binding(2) var<storage, read_write> start_indices: array<u32>;
@group(0) @binding(3) var<uniform> smoothing_radius: f32;
@group(0) @binding(4) var<uniform> num_particles: u32;

struct Entry {
    original_index: u32,
	hash: u32,
	key: u32,
}

#import "shaders/utils.wgsl" as utils

@compute @workgroup_size(64)
fn compute_spatial_lookup(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= num_particles {
        return;
    }

    start_indices[global_id.x] = num_particles;
    let cell_coord = utils::pos_to_cell_coord(predicted_positions[global_id.x], smoothing_radius);

    let hash = utils::hash_cell_coord(cell_coord);

    let cell_key = hash % num_particles;

    spatial_lookup[global_id.x] = Entry(global_id.x, hash, cell_key);
}

