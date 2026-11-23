override spatial_aa: bool = true;

fn aa_sample(uv: vec2<f32>, size: vec2<i32>) -> vec3<f32> {
    let pos = uv * vec2<f32>(size) - 0.5;
    let base = vec2<i32>(floor(pos));
    let f = fract(pos);
    let c00 = textureLoad(taa_current, clamp(base, vec2(0), size - 1), 0).rgb;
    let c10 = textureLoad(taa_current, clamp(base + vec2(1, 0), vec2(0), size - 1), 0).rgb;
    let c01 = textureLoad(taa_current, clamp(base + vec2(0, 1), vec2(0), size - 1), 0).rgb;
    let c11 = textureLoad(taa_current, clamp(base + vec2(1, 1), vec2(0), size - 1), 0).rgb;
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

fn aa_luma_pixel(pixel: vec2<i32>, size: vec2<i32>) -> f32 {
    let color = textureLoad(taa_current, clamp(pixel, vec2(0), size - 1), 0).rgb;
    let luma = max(0.0, dot(color, frame.luminance));
    return sqrt(luma / (1.0 + luma));
}

fn aa_luma_uv(uv: vec2<f32>, size: vec2<i32>) -> f32 {
    let color = aa_sample(uv, size);
    let luma = max(0.0, dot(color, frame.luminance));
    return sqrt(luma / (1.0 + luma));
}

fn spatial_aa_resolve(pixel: vec2<i32>, size: vec2<i32>, uv: vec2<f32>, current: vec3<f32>) -> vec3<f32> {
    if !spatial_aa {
        return current;
    }

    let texel = 1.0 / vec2<f32>(size);
    let x = vec2(texel.x, 0.0);
    let y = vec2(0.0, texel.y);
    let center = sqrt(max(0.0, dot(current, frame.luminance)) / (1.0 + max(0.0, dot(current, frame.luminance))));
    let left = aa_luma_pixel(pixel - vec2(1, 0), size);
    let right = aa_luma_pixel(pixel + vec2(1, 0), size);
    let up = aa_luma_pixel(pixel - vec2(0, 1), size);
    let down = aa_luma_pixel(pixel + vec2(0, 1), size);
    let low = min(center, min(min(left, right), min(up, down)));
    let high = max(center, max(max(left, right), max(up, down)));
    let contrast = high - low;
    // Conservative CMAA2-style pixel-art preservation: use a sensitive edge threshold
    // on genuine geometric depth silhouettes, and a strict threshold on coplanar block
    // interiors so 16x16 Minecraft textures stay razor-sharp instead of softening.
    let z_c = textureLoad(scene_depth, pixel, 0);
    let z_l = textureLoad(scene_depth, clamp(pixel - vec2(1, 0), vec2(0), size - 1), 0);
    let z_r = textureLoad(scene_depth, clamp(pixel + vec2(1, 0), vec2(0), size - 1), 0);
    let z_u = textureLoad(scene_depth, clamp(pixel - vec2(0, 1), vec2(0), size - 1), 0);
    let z_d = textureLoad(scene_depth, clamp(pixel + vec2(0, 1), vec2(0), size - 1), 0);
    let z_span = max(max(abs(z_c - z_l), abs(z_c - z_r)), max(abs(z_c - z_u), abs(z_c - z_d)));
    let is_silhouette = z_span > max(z_c * 0.035, 1e-5);
    let thresh = select(max(0.085, 0.24 * high), max(0.032, 0.11 * high), is_silhouette);
    if contrast < thresh {
        return current;
    }

    let ul = aa_luma_pixel(pixel + vec2(-1, -1), size);
    let ur = aa_luma_pixel(pixel + vec2(1, -1), size);
    let dl = aa_luma_pixel(pixel + vec2(-1, 1), size);
    let dr = aa_luma_pixel(pixel + vec2(1, 1), size);
    let horizontal = abs(ul + dl - 2.0 * left) + 2.0 * abs(up + down - 2.0 * center)
        + abs(ur + dr - 2.0 * right);
    let vertical = abs(ul + ur - 2.0 * up) + 2.0 * abs(left + right - 2.0 * center)
        + abs(dl + dr - 2.0 * down);
    let along = select(y, x, horizontal >= vertical);
    let across = select(x, y, horizontal >= vertical);
    let first = select(left, up, horizontal >= vertical);
    let second = select(right, down, horizontal >= vertical);
    let use_first = abs(first - center) >= abs(second - center);
    let neighbor = select(second, first, use_first);
    let normal = select(across, -across, use_first);
    let edge_luma = 0.5 * (center + neighbor);
    let threshold = 0.25 * abs(neighbor - center);
    let edge = uv + 0.5 * normal;

    var distances = vec2(1.0);
    var end_delta = vec2(0.0);
    for (var side = 0u; side < 2u; side += 1u) {
        let direction = select(-along, along, side == 1u);
        for (var step = 0u; step < 4u; step += 1u) {
            let distance = 1.0 + f32(step * (step + 1u)) * 0.5;
            end_delta[side] = aa_luma_uv(edge + direction * distance, size) - edge_luma;
            distances[side] = distance;
            if abs(end_delta[side]) >= threshold {
                break;
            }
        }
    }

    let near_side = select(1u, 0u, distances.x <= distances.y);
    let same_side = (end_delta[near_side] < 0.0) == (center < edge_luma);
    let edge_offset = select(0.5 - distances[near_side] / (distances.x + distances.y), 0.0, same_side);
    let mean = (2.0 * (left + right + up + down) + ul + ur + dl + dr) / 12.0;
    let subpixel = smoothstep(0.0, 1.0, abs(mean - center) / contrast);
    let offset = max(edge_offset, 0.5 * subpixel * subpixel);
    return aa_sample(uv + offset * normal, size);
}
