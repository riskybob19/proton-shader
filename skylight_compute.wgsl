const SKYLIGHT_WORKGROUP_SIZE: u32 = 64u;
const SKYLIGHT_SAMPLE_COUNT: u32 = 2u * SKYLIGHT_WORKGROUP_SIZE;
const SKYLIGHT_COEFFICIENTS: u32 = 9u;

var<workgroup> coefficients_by_lane: array<array<vec3<f32>, SKYLIGHT_COEFFICIENTS>, SKYLIGHT_WORKGROUP_SIZE>;

// Photon projects hemispherical radiance into order-two SH. This orthonormal
// basis uses Horizon's Y-up world coordinates.
fn skylight_basis(direction: vec3<f32>) -> array<f32, SKYLIGHT_COEFFICIENTS> {
    let x = direction.x;
    let y = direction.y;
    let z = direction.z;

    return array(
        0.282094792,
        0.488602512 * x,
        0.488602512 * y,
        0.488602512 * z,
        1.092548431 * x * y,
        1.092548431 * y * z,
        0.315391565 * (3.0 * y * y - 1.0),
        1.092548431 * x * z,
        0.546274215 * (x * x - z * z),
    );
}

fn skylight_sample(index: u32) -> vec3<f32> {
    let height = (f32(index) + 0.5) / f32(SKYLIGHT_SAMPLE_COUNT);
    let radius = sqrt(1.0 - height * height);
    let azimuth = TAU * fract(f32(index) * 0.618033989);
    return vec3(cos(azimuth) * radius, height, sin(azimuth) * radius);
}

fn irradiance(normal: vec3<f32>) -> vec3<f32> {
    let basis = skylight_basis(normal);
    var diffuse = vec3(0.0);
    for (var band = 0u; band < SKYLIGHT_COEFFICIENTS; band += 1u) {
        // Cosine convolution divided by pi converts radiance to diffuse light.
        let kernel = select(0.25, select(2.0 / 3.0, 1.0, band == 0u), band < 4u);
        diffuse += coefficients_by_lane[0][band] * (basis[band] * kernel);
    }
    return max(diffuse, vec3(0.0));
}

fn skylight_map_normal(uv: vec2<f32>) -> vec3<f32> {
    let oct = uv * 2.0 - 1.0;
    let height = 1.0 - abs(oct.x) - abs(oct.y);
    let plane = select(oct, octahedral_fold(oct), height < 0.0);
    return normalize(vec3(plane.x, height, plane.y));
}

fn compute(input: ComputeInput) {
    if !directional_skylight || frame.dimension != 0 {
        return;
    }

    // The manifest dispatches exactly one workgroup, with two directions per lane.
    let lane = input.local;
    let a = skylight_sample(lane);
    let b = skylight_sample(lane + SKYLIGHT_WORKGROUP_SIZE);
    let a_basis = skylight_basis(a);
    let b_basis = skylight_basis(b);
    let a_radiance = environment_radiance(a) * (TAU / f32(SKYLIGHT_SAMPLE_COUNT));
    let b_radiance = environment_radiance(b) * (TAU / f32(SKYLIGHT_SAMPLE_COUNT));
    for (var band = 0u; band < SKYLIGHT_COEFFICIENTS; band += 1u) {
        coefficients_by_lane[lane][band] = a_radiance * a_basis[band] + b_radiance * b_basis[band];
    }
    workgroupBarrier();

    for (var stride = SKYLIGHT_WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        if lane < stride {
            for (var band = 0u; band < SKYLIGHT_COEFFICIENTS; band += 1u) {
                coefficients_by_lane[lane][band] += coefficients_by_lane[lane + stride][band];
            }
        }
        workgroupBarrier();
    }

    // Match the previous upward-facing brightness and keep a little bounced
    // light in the lower hemisphere. Bake this map for one texture read per face.
    let ambient = ambient_atmosphere_light();
    let upward = irradiance(vec3(0.0, 1.0, 0.0));
    let gain = clamp(dot(ambient, frame.luminance) / max(dot(upward, frame.luminance), 1e-5), 0.4, 4.0);
    let calibrated_gain = mix(1.05, gain, 0.85);
    let size = vec2<u32>(u32(SKYLIGHT_ATLAS_SIZE), u32(SKYLIGHT_ATLAS_SIZE));
    let offset = vec2<i32>(SKYLIGHT_ATLAS_OFFSET);
    for (var index = lane; index < size.x * size.y; index += SKYLIGHT_WORKGROUP_SIZE) {
        let pixel = vec2(index % size.x, index / size.x);
        let uv = vec2<f32>(pixel) / vec2<f32>(size - 1u);
        let normal = skylight_map_normal(uv);
        let light = mix(irradiance(normal) * calibrated_gain, ambient, 0.12);
        textureStore(cloud_aux, offset + vec2<i32>(pixel), vec4(light, 1.0));
    }
}
