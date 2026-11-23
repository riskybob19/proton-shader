// Shared Minecraft Voxel-Normal-Strict 5-Tap Rotated A-Trous Wavelet GI Denoiser:
// Pass 1 filters cardinal (+-step, 0) / (0, +-step); Pass 2 filters diagonal (+-step, +-step)
// so both passes together cover the full 2D footprint at half the texture/ALU cost of 9-tap.
fn gi_atrous_filter(gi_tex: texture_2d<f32>, input: PostInput, step_px: i32, diagonal: bool) -> vec4<f32> {
    if (!ssilvb_gi && !gi_rsm_bounce) || frame.dimension != 0 || input.depth <= 0.0 {
        return vec4(0.0);
    }
    let gi_size = vec2<i32>(textureDimensions(gi_tex));
    let center_cell = clamp(vec2<i32>(input.uv * vec2<f32>(gi_size)), vec2(0), gi_size - 1);
    let center_sample = textureLoad(gi_tex, center_cell, 0);
    let center_dist = center_sample.a;
    if center_dist <= 0.0 {
        return vec4(0.0);
    }

    let full_size_f = vec2<f32>(textureDimensions(scene_depth));
    let full_size_i = vec2<i32>(full_size_f);
    let inv_full = 1.0 / full_size_f;
    let inv_gi = 1.0 / vec2<f32>(gi_size);

    let full_center = clamp(vec2<i32>((vec2<f32>(center_cell) + 0.5) * inv_gi * full_size_f), vec2(0), full_size_i - 1);
    let surf_c = surface_at(full_center);
    let depth_c = textureLoad(scene_depth, full_center, 0);
    if !surf_c.present || depth_c <= 0.0 {
        return center_sample;
    }
    let pos_c = view_position((vec2<f32>(full_center) + 0.5) * inv_full, depth_c);
    let inv_z_tol = 1.0 / max(0.38, 0.065 * center_dist);

    var offsets = array<vec2<i32>, 4>(vec2(-1, 0), vec2(1, 0), vec2(0, -1), vec2(0, 1));
    if diagonal {
        offsets = array<vec2<i32>, 4>(vec2(-1, -1), vec2(1, -1), vec2(-1, 1), vec2(1, 1));
    }

    var accum = center_sample.rgb * 0.36;
    var w_sum = 0.36;

    for (var i = 0u; i < 4u; i += 1u) {
        let tap_cell = clamp(center_cell + offsets[i] * step_px, vec2(0), gi_size - 1);
        let tap = textureLoad(gi_tex, tap_cell, 0);
        if tap.a <= 0.0 {
            continue;
        }
        let dz = abs(tap.a - center_dist);
        let w_z = exp(-dz * inv_z_tol);
        if w_z <= 0.02 {
            continue;
        }
        let tap_full = clamp(vec2<i32>((vec2<f32>(tap_cell) + 0.5) * inv_gi * full_size_f), vec2(0), full_size_i - 1);
        let surf_t = surface_at(tap_full);
        if !surf_t.present {
            continue;
        }
        // Strict voxel normal stop (pow 16 kills any leak across 90-degree block corners).
        let ndot = max(dot(surf_c.normal, surf_t.normal), 0.0);
        if ndot < 0.75 {
            continue;
        }
        let n2 = ndot * ndot;
        let n4 = n2 * n2;
        let w_n = n4 * n4 * n4 * n4;

        // Coplanar plane-distance stop (kills light leaks across 1-block steps/thin walls).
        // Skip inverse-projection view_position on clearly flat coplanar voxel neighbors.
        var w_p = 1.0;
        if dz >= 0.12 || ndot < 0.96 {
            let tap_depth = textureLoad(scene_depth, tap_full, 0);
            if tap_depth <= 0.0 {
                continue;
            }
            let pos_t = view_position((vec2<f32>(tap_full) + 0.5) * inv_full, tap_depth);
            let plane_dist = abs(dot(surf_c.normal, pos_t - pos_c));
            w_p = exp(-plane_dist * 6.5);
        }

        let w = 0.16 * w_z * w_n * w_p;
        accum += tap.rgb * w;
        w_sum += w;
    }
    return vec4(accum / max(w_sum, 1e-4), center_dist);
}
