var<private> PI = radians(180.0);

fn pos_to_cell_coord(pos: vec2f, cell_size: f32) -> vec2i {
    return vec2i(floor(pos / cell_size));
}

fn hash_cell_coord(coord: vec2i) -> u32 {
    let u_coord = vec2u(coord);
    return u32(u_coord.x * 617 + u_coord.y * 307);
}

fn key_from_hash(hash: u32, num_particles: u32) -> u32 {
    return hash % num_particles;
}

fn smoothing_kernel(distance: f32, radius: f32) -> f32 {
    if distance >= radius {
        return 0.0;
    }

    let volume = 6.0 / (radians(180.0) * pow(radius, 4.0));
    return pow(radius - distance, 2.0) * volume;
}

fn smoothing_kernel_derivative(distance: f32, radius: f32) -> f32 {
    if distance > radius {
        return 0.0;
    }

    let scale = 12.0 / (PI * pow(radius, 4.0));

    return scale * -(radius - distance);
}

fn spiky_kernel(distance: f32, radius: f32) -> f32 {
    if distance >= radius {
        return 0.0;
    }

    let volume = 10.0 / (radians(180.0) * pow(radius, 5.0));
    return pow(radius - distance, 3.0) * volume;
}

fn spiky_kernel_derivative(distance: f32, radius: f32) -> f32 {
    if distance >= radius {
        return 0.0;
    }

    let volume = 30.0 / (PI * pow(radius, 5.0));
    return pow(radius - distance, 2.0) * volume;
}
