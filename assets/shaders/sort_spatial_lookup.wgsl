@group(0) @binding(0) var<storage, read_write> spatial_lookup: array<Entry>;
@group(0) @binding(1) var<uniform> num_particles: u32;
@group(0) @binding(2) var<storage, read_write> step_index: u32;
@group(0) @binding(3) var<storage, read_write> stage_index: u32;

struct Entry {
    original_index: u32,
	hash: u32,
	key: u32,
}

// Sort the given entries by their keys (smallest to largest)
// This is done using bitonic merge sort, and takes multiple iterations
@compute @workgroup_size(1)
fn sort_spatial_lookup(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    let group_width = 1 << (stage_index - step_index);
    let group_height = 2 * group_width - 1;

    let h_index = i & (group_width - 1);
    let index_left = h_index + (group_height + 1) * (i / group_width);
    var right_step_size = (group_height + 1) / 2;
    if step_index == 0 {
        right_step_size = group_height - 2 * h_index;
    }
    let index_right: u32 = index_left + right_step_size;

	// Exit if out of bounds (for non-power of 2 input sizes)
    if index_right >= num_particles {
        return;
    }

    let value_left = spatial_lookup[index_left].key;
    let value_right = spatial_lookup[index_right].key;

	// Swap entries if value is descending
    if value_left > value_right {
        let temp = spatial_lookup[index_left];
        spatial_lookup[index_left] = spatial_lookup[index_right];
        spatial_lookup[index_right] = temp;
    }

    if global_id.x == 0 {
        step_index += 1;
        if step_index > stage_index + 1 {
            step_index = 0;
            stage_index += 1;
        }
    }
}

