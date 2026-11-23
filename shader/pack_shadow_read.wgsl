fn pack_shadow_position(position: vec3<f32>) -> vec3<f32> {
    if frame.has_shadow != 0u {
        return shadow_position(position);
    }
    let phase = textureLoad(cloud_aux, ray_shadow_phase_pixel(), 0).xyz;
    let light = transpose(ray_shadow_basis()) * position;
    let texel = light.xy / ray_shadow_grid().xy + phase.xy;
    return vec3(0.5 + texel / RAY_SHADOW_SIZE, 0.5 - (light.z + phase.z) / (2.0 * RAY_SHADOW_DEPTH));
}

fn pack_shadow_size() -> vec2<i32> {
    if frame.has_shadow != 0u {
        return vec2<i32>(textureDimensions(shadow_map));
    }
    return vec2<i32>(i32(RAY_SHADOW_SIZE));
}

fn ray_shadow_sample(pixel: vec2<i32>) -> vec4<f32> {
    let size = vec2<i32>(i32(RAY_SHADOW_SIZE));
    return textureLoad(cloud_aux, ray_shadow_atlas_pixel(clamp(pixel, vec2(0), size - 1)), 0);
}

fn pack_shadow_depth(pixel: vec2<i32>, translucent: bool) -> f32 {
    if frame.has_shadow != 0u {
        if translucent {
            return textureLoad(shadow_map_translucent, pixel, 0);
        }
        return textureLoad(shadow_map, pixel, 0);
    }
    let sample = ray_shadow_sample(pixel);
    return 0.5 + select(sample.x, sample.y, translucent) * (0.5 / RAY_SHADOW_DEPTH);
}

fn pack_shadow_color(pixel: vec2<i32>) -> vec4<f32> {
    if frame.has_shadow != 0u {
        return textureLoad(shadow_color, pixel, 0);
    }
    return ray_shadow_color(ray_shadow_sample(pixel).zw);
}

fn ray_shadow_color(encoded: vec2<f32>) -> vec4<f32> {
    let packed = vec2<u32>(encoded);
    let rgb = vec3(packed.x & 31u, packed.x >> 5u, packed.y & 31u);
    return vec4(vec3<f32>(rgb), f32(packed.y >> 5u)) / 31.0;
}

struct RayShadowReceiver {
    position: vec2<f32>,
    depth: f32,
    slope: vec2<f32>,
    grazing_fade: f32,
}

fn ray_shadow_surface(projected: vec3<f32>, normal: vec3<f32>) -> RayShadowReceiver {
    let plane = transpose(ray_shadow_basis()) * normal;
    // Project each tap onto the actual receiving plane. Fade the shadow
    // smoothly where the plane is too close to parallel to the rays.
    let denominator = select(-1.0, 1.0, plane.z >= 0.0) * max(abs(plane.z), 0.008);
    let slope = plane.xy * ray_shadow_grid().xy / denominator;
    let fade = 1.0 - smoothstep(0.008, 0.025, abs(plane.z));
    return RayShadowReceiver(projected.xy * RAY_SHADOW_SIZE, (projected.z - 0.5) * (2.0 * RAY_SHADOW_DEPTH), slope, fade);
}

fn ray_shadow_volume(uv: vec2<f32>, projected_depth: f32) -> RayShadowReceiver {
    return RayShadowReceiver(uv * RAY_SHADOW_SIZE, (projected_depth - 0.5) * (2.0 * RAY_SHADOW_DEPTH), vec2(0.0), 0.0);
}

fn ray_shadow_receiver_at(receiver: RayShadowReceiver, pixel: vec2<i32>, surface: bool) -> f32 {
    let tap = vec2<f32>(clamp(pixel, vec2(0), vec2(i32(RAY_SHADOW_SIZE) - 1))) + 0.5;
    let depth = receiver.depth + dot(tap - receiver.position, receiver.slope);
    // Leave room for half-float depth quantization and ray intersection error.
    return depth - select(0.0, 0.015625 + abs(depth) * 0.00055, surface);
}

fn ray_shadow_tap(pixel: vec2<i32>, receiver: RayShadowReceiver, surface: bool) -> vec3<f32> {
    let sample = ray_shadow_sample(pixel);
    let depth = ray_shadow_receiver_at(receiver, pixel, surface);
    if sample.x < depth && sample.x < RAY_SHADOW_DEPTH {
        return vec3(0.0);
    }
    if sample.y >= depth || sample.w < 32.0 {
        return vec3(1.0);
    }
    let caster = ray_shadow_color(sample.zw);
    return mix(vec3(1.0), caster.rgb, caster.a);
}

struct ShadowBilinearTap {
    pixel: vec2<i32>,
    weight: f32,
}

// texel is measured from pixel centers. Keep the weights at the original
// position even when a reader clamps a tap to the edge of the map.
fn shadow_bilinear_tap(texel: vec2<f32>, index: u32) -> ShadowBilinearTap {
    let offset = vec2<i32>(i32(index & 1u), i32(index >> 1u));
    let blend = fract(texel);
    let weights = select(1.0 - blend, blend, offset == vec2(1));
    return ShadowBilinearTap(vec2<i32>(floor(texel)) + offset, weights.x * weights.y);
}

fn ray_shadow_filtered(uv: vec2<f32>, receiver: RayShadowReceiver, surface: bool) -> vec3<f32> {
    let texel = uv * RAY_SHADOW_SIZE - 0.5;
    var visible = vec3(0.0);
    for (var tap = 0u; tap < 4u; tap += 1u) {
        let filter_tap = shadow_bilinear_tap(texel, tap);
        visible += ray_shadow_tap(filter_tap.pixel, receiver, surface) * filter_tap.weight;
    }
    return visible;
}

fn ray_shadow_box_weights(blend: vec2<f32>) -> array<vec2<f32>, 4> {
    // Integrate a sliding three-texel box. A ray contributes at most 1/9.
    return array<vec2<f32>, 4>((1.0 - blend) / 3.0, vec2(1.0 / 3.0), vec2(1.0 / 3.0), blend / 3.0);
}

fn ray_shadow_antialiased(uv: vec2<f32>, receiver: RayShadowReceiver) -> vec3<f32> {
    let texel = uv * RAY_SHADOW_SIZE - 0.5;
    let corner = vec2<i32>(floor(texel));
    let weights = ray_shadow_box_weights(fract(texel));
    var visible = vec3(0.0);
    for (var y = 0; y < 4; y += 1) {
        for (var x = 0; x < 4; x += 1) {
            let pixel = corner + vec2(x - 1, y - 1);
            visible += ray_shadow_tap(pixel, receiver, true) * (weights[x].x * weights[y].y);
        }
    }
    return visible;
}

// Filter water coverage and thickness together so a partially covered texel
// cannot paint a hard strip across dry terrain at the shoreline.
fn ray_shadow_water(projected: vec3<f32>) -> vec2<f32> {
    let texel = projected.xy * RAY_SHADOW_SIZE - 0.5;
    let depth = (projected.z - 0.5) * (2.0 * RAY_SHADOW_DEPTH);
    var result = vec2(0.0);
    for (var tap = 0u; tap < 4u; tap += 1u) {
        let filter_tap = shadow_bilinear_tap(texel, tap);
        let sample = ray_shadow_sample(filter_tap.pixel);
        let thickness = clamp(depth - sample.y, 0.0, 20.0);
        let water = sample.w >= f32(RAY_SHADOW_WATER_FLAG << 5u);
        let coverage = filter_tap.weight * f32(water) * smoothstep(0.045, 0.22, thickness);
        result += coverage * vec2(thickness, 1.0);
    }
    return vec2(result.x / max(result.y, 1e-5), result.y);
}

fn pack_shadow_compare(uv: vec2<f32>, depth: f32, translucent: bool) -> f32 {
    if frame.has_shadow != 0u {
        if translucent {
            return textureSampleCompareLevel(shadow_map_translucent, shadow_sampler, uv, depth);
        }
        return textureSampleCompareLevel(shadow_map, shadow_sampler, uv, depth);
    }
    return ray_shadow_compare(uv, ray_shadow_volume(uv, depth), translucent, false);
}

fn ray_shadow_compare(uv: vec2<f32>, receiver: RayShadowReceiver, translucent: bool, surface: bool) -> f32 {
    let texel = uv * RAY_SHADOW_SIZE - 0.5;
    var visible = 0.0;
    for (var tap = 0u; tap < 4u; tap += 1u) {
        let filter_tap = shadow_bilinear_tap(texel, tap);
        let sample = ray_shadow_sample(filter_tap.pixel);
        let depth = select(sample.x, sample.y, translucent);
        let receiver_depth = ray_shadow_receiver_at(receiver, filter_tap.pixel, surface);
        visible += filter_tap.weight * f32(depth >= receiver_depth || depth >= RAY_SHADOW_DEPTH);
    }
    return visible;
}
