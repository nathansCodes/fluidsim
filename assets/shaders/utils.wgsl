#define_import_path utils

var PI = radians(180.0);

var CELL_OFFSETS = array<vec2f>(
    vec2f(-1.0, -1.0),
    vec2f(0.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(-1.0, 0.0),
    vec2f(0.0, 0.0),
    vec2f(1.0, 0.0),
    vec2f(-1.0, 1.0),
    vec2f(0.0, 1.0),
    vec2f(1.0, 1.0)
);

fn pos_to_cell_coord(pos: vec2f, cell_size: f32) -> vec2f {
    return floor(pos / cell_size);
}

fn hash_cell_coord(coord: vec2f) -> u32 {
    return u32(coord.x * 617 + coord.y * 307);
}

fn key_from_hash(hash: u32, num_particles: u32) -> u32 {
    return hash % num_particles;
}

fn smoothing_kernel(distance: f32, radius: f32) -> f32 {
    if distance >= radius {
        return 0.0;
    }

    let volume = 6.0 / (PI * pow(radius, 4));
    return pow(radius - distance, 2) * volume;
}

fn smoothing_kernel_derivative(distance: f32, radius: f32) -> f32 {
    if distance > radius {
        return 0.0;
    }

    let scale = 12.0 / (PI * pow(radius, 4));

    return scale * -(radius - distance);
}

fn spiky_kernel(distance: f32, radius: f32) -> f32 {
    if distance >= radius {
        return 0.0;
    }

    let volume = 10.0 / (PI * pow(radius, 5));
    return pow(radius - distance, 3) * volume;
}

fn spiky_kernel_derivative(distance: f32, radius: f32) -> f32 {
    if distance >= radius {
        return 0.0;
    }

    let volume = 30.0 / (PI * pow(radius, 5));
    return pow(radius - distance, 2) * volume;
}
