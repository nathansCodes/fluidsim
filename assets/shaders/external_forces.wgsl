@group(0) @binding(0) var<storage, read_write> positions: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> predicted_positions: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec2f>;
@group(0) @binding(3) var<uniform> gravity: vec2f;
@group(0) @binding(4) var<uniform> planetary_gravity: u32;
@group(0) @binding(5) var<uniform> planet_position: vec2f;
@group(0) @binding(6) var<uniform> delta: f32;

fn rotate_around_origin(v: vec2f, theta: f32) -> vec2f {
    let cs = cos(theta);
    let sn = sin(theta);

    return vec2f(v.x * cs - v.y * sn, v.x * sn + v.y * cs);
}

@compute @workgroup_size(64)
fn external_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var force = gravity;

    if planetary_gravity != 0 {
        let relative_position = positions[global_id.x] - planet_position;
        let sqr_dst = pow(relative_position.x, 2.0) + pow(relative_position.y, 2.0);
        // direction from origin to position
        let direction = relative_position / sqrt(sqr_dst);

        //let angle = atan(direction.y / direction.x);

        //let rotated_gravity = rotate_around_origin(gravity, angle);
        force = direction * (gravity.y / sqr_dst);
    }

    velocities[global_id.x] += force * delta;
    predicted_positions[global_id.x] = positions[global_id.x] + velocities[global_id.x] * delta;
}

