@group(0) @binding(0) var<storage, read_write> spatial_lookup: array<Entry>;
@group(0) @binding(1) var<storage, read_write> start_indices: array<u32>;
@group(0) @binding(2) var<uniform> num_particles: u32;

struct Entry {
    original_index: u32,
	hash: u32,
	key: u32,
}

// NOTE: TOTALLY not copied from Sebastian Lague...
//
// Calculate offsets into the sorted Entries buffer (used for spatial hashing).
// For example, given an Entries buffer sorted by key like so: {2, 2, 2, 3, 6, 6, 9, 9, 9, 9}
// The resulting Offsets calculated here should be:            {-, -, 0, 3, -, -, 4, -, -, 6}
// (where '-' represents elements that won't be read/written)
//
// Usage example:
// Say we have a particular particle P, and we want to know which particles are in the same grid cell as it.
// First we would calculate the Key of P based on its position. Let's say in this example that Key = 9.
// Next we can look up Offsets[Key] to get: Offsets[9] = 6
// This tells us that SortedEntries[6] is the first particle that's in the same cell as P.
// We can then loop until we reach a particle with a different cell key in order to iterate over all the particles in the cell.
//
// NOTE: offsets buffer must filled with values equal to (or greater than) its length to ensure that this works correctly
@compute @workgroup_size(128)
fn set_start_indices(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;

    if i >= num_particles { return; }

    let key: u32 = spatial_lookup[i].key;
    var key_prev: u32 = 0;

    if i == 0 {
        key_prev = num_particles;
    } else {
        key_prev = spatial_lookup[i - 1].key;
    }

    if key != key_prev {
        start_indices[key] = i;
    }
}
