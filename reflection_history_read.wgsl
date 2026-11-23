// Reconstruct a distant hit at its fractional pixel position. Include only
// neighbors on the same depth layer so sky cannot bleed across a treeline.
fn distant_reflection_color(colors: texture_2d<f32>, depths: texture_depth_2d,
    uv: vec2<f32>, depth: f32, path_length: f32, original: vec3<f32>) -> vec3<f32> {
    let strength = smoothstep(20.0, 72.0, path_length);
    if strength <= 0.0 || depth <= 0.0 {
        return original;
    }
    let size = vec2<i32>(textureDimensions(colors));
    let texel = uv * vec2<f32>(size) - 0.5;
    let center = clamp(vec2<i32>(uv * vec2<f32>(size)), vec2(0), size - 1);
    let noise = cloud_hash(center);
    let angle = TAU * noise;
    let axis = vec2(cos(angle), sin(angle));
    let radius = mix(1.0, 2.75, strength);
    var sum = textureLoad(colors, center, 0).rgb;
    var total = 1.0;
    // Rotate a symmetric tap pair with a stable noise seed; TAA blends across frames.
    for (var i = 0u; i < 2u; i += 1u) {
        let direction = axis * select(-1.0, 1.0, i == 1u);
        let pixel = clamp(vec2<i32>(floor(texel + 0.5 + radius * direction)), vec2(0), size - 1);
        let neighbor = textureLoad(depths, pixel, 0);
        let relative_gap = abs(neighbor - depth) / max(depth, 1e-8);
        let weight = 0.65 * (1.0 - smoothstep(0.007, 0.035, relative_gap));
        sum += textureLoad(colors, pixel, 0).rgb * weight;
        total += weight;
    }
    let filtered = sum / total;
    let grain = (noise - 0.5) * 0.012;
    return mix(original, filtered * (1.0 + grain), strength);
}

fn refl_reinhard(rgb: vec3<f32>) -> vec3<f32> {
    let c = max(rgb, vec3(0.0));
    return c / (c + 1.0);
}

fn refl_reinhard_inv(rgb: vec3<f32>) -> vec3<f32> {
    let c = clamp(rgb, vec3(0.0), vec3(0.9999));
    return c / (1.0 - c);
}

fn reflected_history(position: vec3<f32>, current: vec3<f32>, roughness: f32) -> vec3<f32> {
    if !temporal_reflections || frame.delta_time <= 0.0 || frame.delta_time > 0.16 {
        return current;
    }
    let previous_position = position + frame.camera_motion;
    let clip = frame.previous_view_projection * vec4(previous_position, 1.0);
    if clip.w <= 0.01 || any(abs(clip.xy) >= vec2(clip.w)) {
        return current;
    }
    let uv = 0.5 + clip.xy / clip.w * vec2(0.5, -0.5);
    let size = vec2<i32>(textureDimensions(taa_history_previous));
    let center = clamp(vec2<i32>(uv * vec2<f32>(size)), vec2(0), size - 1);
    let expected = length(previous_position);
    let tolerance = max(0.28, 0.045 * expected);
    let history = textureLoad(taa_history_previous, center, 0);
    if history.a <= 0.0 || abs(history.a - expected) >= tolerance {
        return current;
    }

    var color_tm = refl_reinhard(history.rgb);
    var weight = 1.0;
    if roughness > 0.10 {
        let radius = 1 + i32(roughness * 4.0);
        for (var tap = 0u; tap < 4u; tap += 1u) {
            let quad = vec2<i32>(i32(tap & 1u), i32(tap >> 1u)) * 2 - 1;
            let offset = clamp(center + quad * radius, vec2(0), size - 1);
            let sample_value = textureLoad(taa_history_previous, offset, 0);
            let blend = 1.0 - smoothstep(0.0, tolerance, abs(sample_value.a - expected));
            if sample_value.a > 0.0 {
                color_tm += blend * refl_reinhard(sample_value.rgb);
                weight += blend;
            }
        }
    }
    let current_tm = refl_reinhard(current);
    let hist_tm = color_tm / weight;
    let range = vec3(0.14 + 0.18 * roughness);
    let clipped_tm = clamp(hist_tm, current_tm - range, current_tm + range);
    return refl_reinhard_inv(mix(current_tm, clipped_tm, 0.62));
}

fn reflected_aerial_color(color: vec3<f32>, ray: vec3<f32>, distance: f32, sky_access: f32) -> vec3<f32> {
    let density = mix(0.0012, 0.010, frame.rain) + 0.0015 * weather_humidity();
    let haze = 1.0 - exp(-density * distance * sky_access);
    return mix(color, atmosphere(ray), haze);
}
