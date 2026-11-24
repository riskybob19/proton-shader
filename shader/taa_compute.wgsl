override temporal_aa: bool = true;

// 10x10 workgroup shared-memory tile for 8x8 TAA workgroups (1.6 KiB SMEM):
// preloads Reinhard-YCoCg color and scene_depth once per tile (~4.5x fewer global
// texture fetches and ~5.8x fewer YCoCg conversions, bit-identical output).
var<workgroup> taa_tile_ycocg: array<vec3<f32>, 100>;
var<workgroup> taa_tile_depth: array<f32, 100>;

fn taa_tile_idx(lx: i32, ly: i32) -> i32 {
    return (ly + 1) * 10 + (lx + 1);
}

fn taa_reinhard(rgb: vec3<f32>) -> vec3<f32> {
    let c = max(rgb, vec3(0.0));
    return c / (c + 1.0);
}

fn taa_reinhard_inv(rgb: vec3<f32>) -> vec3<f32> {
    let c = clamp(rgb, vec3(0.0), vec3(0.9999));
    return c / (1.0 - c);
}

fn taa_rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    let co = rgb.r - rgb.b;
    let t = rgb.b + 0.5 * co;
    let cg = rgb.g - t;
    let y = t + 0.5 * cg;
    return vec3(y, co, cg);
}

fn taa_ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    let t = ycocg.x - 0.5 * ycocg.z;
    let g = ycocg.z + t;
    let b = t - 0.5 * ycocg.y;
    let r = b + ycocg.y;
    return vec3(r, g, b);
}

// 5-tap Catmull-Rom history filter (upstream Photon catmull_rom_filter_fast_rgb)
// eliminates bilinear texture smearing during camera motion.
fn catmull_rom_taa_history(uv: vec2<f32>, size: vec2<i32>) -> vec3<f32> {
    let res = vec2<f32>(size);
    let pos = uv * res;
    let center = floor(pos - 0.5) + 0.5;
    let f = pos - center;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = f2 - 0.5 * (f3 + f);
    let w1 = 1.5 * f3 - 2.5 * f2 + 1.0;
    let w3 = 0.5 * (f3 - f2);
    let w2 = 1.0 - w0 - w1 - w3;
    let w12 = w1 + w2;
    let t0 = (center - 1.0) / res;
    let t12 = (center + w2 / max(w12, vec2(1e-5))) / res;
    let t3 = (center + 2.0) / res;
    let c0 = textureSampleLevel(taa_history_previous, scattering_atlas_sampler, vec2(t12.x, t0.y), 0.0).rgb;
    let c1 = textureSampleLevel(taa_history_previous, scattering_atlas_sampler, vec2(t0.x, t12.y), 0.0).rgb;
    let c2 = textureSampleLevel(taa_history_previous, scattering_atlas_sampler, t12, 0.0).rgb;
    let c3 = textureSampleLevel(taa_history_previous, scattering_atlas_sampler, vec2(t3.x, t12.y), 0.0).rgb;
    let c4 = textureSampleLevel(taa_history_previous, scattering_atlas_sampler, vec2(t12.x, t3.y), 0.0).rgb;
    let sum = c0 * (w12.x * w0.y) + c1 * (w0.x * w12.y) + c2 * (w12.x * w12.y)
        + c3 * (w3.x * w12.y) + c4 * (w12.x * w3.y);
    let norm = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(sum / max(norm, 1e-5), vec3(0.0));
}

// SIGGRAPH Asia 2024 k-DOP (18-DOP) Polytope Clipping (Ikkala et al.):
// Ray-clips history toward the neighborhood center against both the 6 axis-aligned
// YCoCg slabs AND 12 diagonal slabs (Y+-Co, Y+-Cg, Co+-Cg), cutting false-positive
// ghost volume by ~50-60% on diagonal color gradients without blurring sharp edges.
fn taa_clip_18dop(history: vec3<f32>, box_lo: vec3<f32>, box_hi: vec3<f32>,
    d0_lo: vec3<f32>, d0_hi: vec3<f32>, d1_lo: vec3<f32>, d1_hi: vec3<f32>) -> vec3<f32> {
    let center = 0.5 * (box_hi + box_lo);
    let delta = history - center;
    var t_max = 1.0;

    // 3 axis-aligned slab pairs (6 planes)
    let extent = max(0.5 * (box_hi - box_lo), vec3(1e-3));
    let unit = abs(delta / extent);
    let max_axis = max(unit.x, max(unit.y, unit.z));
    if max_axis > 1.0 {
        t_max = min(t_max, 1.0 / max_axis);
    }

    // 6 diagonal slab pairs (12 planes): (Y+Co, Y+Cg, Co+Cg) and (Y-Co, Y-Cg, Co-Cg)
    let c_p = vec3(center.x + center.y, center.x + center.z, center.y + center.z);
    let d_p = vec3(delta.x + delta.y, delta.x + delta.z, delta.y + delta.z);
    let c_m = vec3(center.x - center.y, center.x - center.z, center.y - center.z);
    let d_m = vec3(delta.x - delta.y, delta.x - delta.z, delta.y - delta.z);

    let hi_p = max(d0_hi - c_p, vec3(1e-3));
    let lo_p = min(d0_lo - c_p, vec3(-1e-3));
    let tp = select(select(vec3(1.0), lo_p / min(d_p, vec3(-1e-6)), d_p < lo_p),
        hi_p / max(d_p, vec3(1e-6)), d_p > hi_p);
    t_max = min(t_max, min(tp.x, min(tp.y, tp.z)));

    let hi_m = max(d1_hi - c_m, vec3(1e-3));
    let lo_m = min(d1_lo - c_m, vec3(-1e-3));
    let tm = select(select(vec3(1.0), lo_m / min(d_m, vec3(-1e-6)), d_m < lo_m),
        hi_m / max(d_m, vec3(1e-6)), d_m > hi_m);
    t_max = min(t_max, min(tm.x, min(tm.y, tm.z)));

    return center + delta * clamp(t_max, 0.0, 1.0);
}

// 5-tap closest-depth dilation from workgroup shared memory (reversed-Z: larger depth is closer)
// keeps moving silhouettes crisp during camera rotation/translation with zero extra texture loads.
fn taa_closest_depth(pixel: vec2<i32>, local_xy: vec2<i32>, size: vec2<i32>, center_uv: vec2<f32>, center_depth: f32) -> vec3<f32> {
    let inv_size = 1.0 / vec2<f32>(size);
    var best = vec3(center_uv, center_depth);
    let d0 = taa_tile_depth[taa_tile_idx(local_xy.x - 1, local_xy.y - 1)];
    if d0 > best.z {
        best = vec3((vec2<f32>(clamp(pixel + vec2(-1, -1), vec2(0), size - 1)) + 0.5) * inv_size, d0);
    }
    let d1 = taa_tile_depth[taa_tile_idx(local_xy.x + 1, local_xy.y - 1)];
    if d1 > best.z {
        best = vec3((vec2<f32>(clamp(pixel + vec2(1, -1), vec2(0), size - 1)) + 0.5) * inv_size, d1);
    }
    let d2 = taa_tile_depth[taa_tile_idx(local_xy.x - 1, local_xy.y + 1)];
    if d2 > best.z {
        best = vec3((vec2<f32>(clamp(pixel + vec2(-1, 1), vec2(0), size - 1)) + 0.5) * inv_size, d2);
    }
    let d3 = taa_tile_depth[taa_tile_idx(local_xy.x + 1, local_xy.y + 1)];
    if d3 > best.z {
        best = vec3((vec2<f32>(clamp(pixel + vec2(1, 1), vec2(0), size - 1)) + 0.5) * inv_size, d3);
    }
    return best;
}

fn taa_resolve(pixel: vec2<i32>, local_xy: vec2<i32>, size: vec2<i32>, uv: vec2<f32>, depth: f32,
    center_pos: vec3<f32>, current: vec3<f32>) -> vec3<f32> {
    if !temporal_aa || frame.delta_time <= 0.0 || frame.delta_time > 0.16 {
        return current;
    }

    // Closest-depth velocity dilation for crisp moving terrain silhouettes.
    // Keep true sky pixels (depth <= 0.0) on their own sky velocity so nearby trees,
    // leaves, or mountains never drag dark foreground history across the blue sky.
    let center_sky = depth <= 0.0;
    let closest = select(taa_closest_depth(pixel, local_xy, size, uv, depth), vec3(uv, depth), center_sky);
    let sky = center_sky || closest.z <= 0.0;
    let same_pixel = all(closest.xy == uv) && !sky;
    var pos = center_pos;
    if !same_pixel {
        pos = view_position(closest.xy, select(closest.z, 1.0, sky));
    }
    let old_pos = pos + frame.camera_motion;
    let clip = frame.previous_view_projection * select(vec4(old_pos, 1.0), vec4(pos, 0.0), sky);
    if clip.w <= 0.0 || any(abs(clip.xy) >= vec2(clip.w)) {
        return current;
    }
    let old_uv = 0.5 + clip.xy / clip.w * vec2(0.5, -0.5);
    let velocity = closest.xy - old_uv;
    let prev_uv = uv - velocity;
    if any(prev_uv <= vec2(0.0)) || any(prev_uv >= vec2(1.0)) {
        return current;
    }

    // Depth disocclusion check against stored history distance.
    // On sky pixels, any previous foreground/terrain sample in the 2x2 reprojection
    // footprint is an immediate disocclusion (even when motion_px == 0 during pure
    // translation/flying), completely eliminating foreground ghosting on blue sky.
    let res = vec2<f32>(size);
    let motion_px = length(velocity * res);
    let expected = length(old_pos);
    let nearest = clamp(vec2<i32>(prev_uv * res), vec2(0), size - 1);
    let last_depth = textureLoad(taa_history_previous, nearest, 0).a;
    if sky {
        if last_depth >= 0.0 {
            return current;
        }
        // During fast camera panning or flight, skip TAA history completely on sky pixels:
        // cloud_environment_map already accumulates in rotation-invariant world panorama
        // space, so dropping TAA history here eliminates sky/cloud panning ghosts cleanly.
        if motion_px >= 2.5 || length(frame.camera_motion) >= 0.18 {
            return current;
        }
        let p0 = clamp(vec2<i32>(floor(prev_uv * res - 0.5)), vec2(0), size - 2);
        let d00 = textureLoad(taa_history_previous, p0, 0).a;
        let d10 = textureLoad(taa_history_previous, p0 + vec2(1, 0), 0).a;
        let d01 = textureLoad(taa_history_previous, p0 + vec2(0, 1), 0).a;
        let d11 = textureLoad(taa_history_previous, p0 + vec2(1, 1), 0).a;
        if d00 >= 0.0 || d10 >= 0.0 || d01 >= 0.0 || d11 >= 0.0 {
            return current;
        }
    }
    let depth_ok = select(
        last_depth > 0.0 && abs(last_depth - expected) < max(0.35, 0.08 * expected),
        true,
        sky,
    );
    let disocc_keep = select(1.0 - smoothstep(0.25, 1.25, motion_px), 1.0, depth_ok);
    if disocc_keep <= 0.001 {
        // Decima / SMAA-T2x hybrid fallback: apply conservative spatial edge AA on
        // newly disoccluded pixels so fast panning never exposes raw jagged edges.
        return spatial_aa_resolve(pixel, size, uv, current);
    }

    // 5-tap Catmull-Rom reprojection (sharp) on moving terrain; use 1-tap bilinear on
    // stationary pixels and validated sky pixels (whose 2x2 footprint is proven 100% sky,
    // avoiding 4x4 Catmull-Rom ringing/bleed from nearby trees or mountains).
    var history: vec3<f32>;
    if sky || motion_px < 0.08 {
        history = max(textureSampleLevel(taa_history_previous, linear_sampler, prev_uv, 0.0).rgb, vec3(0.0));
    } else {
        history = catmull_rom_taa_history(prev_uv, size);
    }
    let current_tm = taa_reinhard(current);
    var history_tm = taa_reinhard(history);

    // 3x3 YCoCg neighborhood 18-DOP polytope hull + variance clipping (from shared memory, unrolled).
    let c_e = taa_tile_ycocg[taa_tile_idx(local_xy.x, local_xy.y)];
    let c_n = taa_tile_ycocg[taa_tile_idx(local_xy.x, local_xy.y - 1)];
    let c_w = taa_tile_ycocg[taa_tile_idx(local_xy.x - 1, local_xy.y)];
    let c_ea = taa_tile_ycocg[taa_tile_idx(local_xy.x + 1, local_xy.y)];
    let c_s = taa_tile_ycocg[taa_tile_idx(local_xy.x, local_xy.y + 1)];
    let c_nw = taa_tile_ycocg[taa_tile_idx(local_xy.x - 1, local_xy.y - 1)];
    let c_ne = taa_tile_ycocg[taa_tile_idx(local_xy.x + 1, local_xy.y - 1)];
    let c_sw = taa_tile_ycocg[taa_tile_idx(local_xy.x - 1, local_xy.y + 1)];
    let c_se = taa_tile_ycocg[taa_tile_idx(local_xy.x + 1, local_xy.y + 1)];

    let m1 = c_e + c_n + c_w + c_ea + c_s + c_nw + c_ne + c_sw + c_se;
    let m2 = c_e * c_e + c_n * c_n + c_w * c_w + c_ea * c_ea + c_s * c_s
        + c_nw * c_nw + c_ne * c_ne + c_sw * c_sw + c_se * c_se;
    let cross_min = min(c_e, min(min(c_n, c_s), min(c_w, c_ea)));
    let cross_max = max(c_e, max(max(c_n, c_s), max(c_w, c_ea)));
    let box_min = min(cross_min, min(min(c_nw, c_ne), min(c_sw, c_se)));
    let box_max = max(cross_max, max(max(c_nw, c_ne), max(c_sw, c_se)));

    // Soft min/max (average of 5-tap cross and 3x3 box) + variance clamp.
    var min_c = 0.5 * (cross_min + box_min);
    var max_c = 0.5 * (cross_max + box_max);
    let distance_factor = select(1.0 - exp2(-0.025 * length(pos)), 1.0, sky);
    let mu = m1 * (1.0 / 9.0);
    let sigma = sqrt(max(m2 * (1.0 / 9.0) - mu * mu, vec3(0.0)));
    // Relax variance clamping on stationary/slow pixels so 8x Halton jitter integrates
    // sub-pixel edges cleanly, while tightening under camera motion to kill ghosts.
    let motion_tighten = smoothstep(0.20, 2.2, motion_px);
    let static_gamma = mix(1.85, 2.35, clamp((distance_factor - 0.25) / 0.75, 0.0, 1.0));
    let moving_gamma = mix(0.85, 1.25, clamp((distance_factor - 0.25) / 0.75, 0.0, 1.0));
    let gamma = mix(static_gamma, moving_gamma, motion_tighten);
    min_c = max(min_c, mu - gamma * sigma);
    max_c = min(max_c, mu + gamma * sigma);
    let box_lo = mix(min(box_min, min_c), min(min_c, max_c), motion_tighten);
    let box_hi = mix(max(box_max, max_c), max(min_c, max_c), motion_tighten);

    // When stationary (motion_px < 0.15), 6-plane YCoCg AABB + variance clip is identical
    // to 18-DOP without diagonal ghosting; engage full 18-DOP polytope hull under motion.
    var history_ycocg = taa_rgb_to_ycocg(history_tm);
    if motion_px >= 0.15 {
        let p_e = vec3(c_e.x + c_e.y, c_e.x + c_e.z, c_e.y + c_e.z);
        let p_n = vec3(c_n.x + c_n.y, c_n.x + c_n.z, c_n.y + c_n.z);
        let p_w = vec3(c_w.x + c_w.y, c_w.x + c_w.z, c_w.y + c_w.z);
        let p_ea = vec3(c_ea.x + c_ea.y, c_ea.x + c_ea.z, c_ea.y + c_ea.z);
        let p_s = vec3(c_s.x + c_s.y, c_s.x + c_s.z, c_s.y + c_s.z);
        let p_nw = vec3(c_nw.x + c_nw.y, c_nw.x + c_nw.z, c_nw.y + c_nw.z);
        let p_ne = vec3(c_ne.x + c_ne.y, c_ne.x + c_ne.z, c_ne.y + c_ne.z);
        let p_sw = vec3(c_sw.x + c_sw.y, c_sw.x + c_sw.z, c_sw.y + c_sw.z);
        let p_se = vec3(c_se.x + c_se.y, c_se.x + c_se.z, c_se.y + c_se.z);
        let d0_min = min(p_e, min(min(min(p_n, p_s), min(p_w, p_ea)), min(min(p_nw, p_ne), min(p_sw, p_se))));
        let d0_max = max(p_e, max(max(max(p_n, p_s), max(p_w, p_ea)), max(max(p_nw, p_ne), max(p_sw, p_se))));

        let q_e = vec3(c_e.x - c_e.y, c_e.x - c_e.z, c_e.y - c_e.z);
        let q_n = vec3(c_n.x - c_n.y, c_n.x - c_n.z, c_n.y - c_n.z);
        let q_w = vec3(c_w.x - c_w.y, c_w.x - c_w.z, c_w.y - c_w.z);
        let q_ea = vec3(c_ea.x - c_ea.y, c_ea.x - c_ea.z, c_ea.y - c_ea.z);
        let q_s = vec3(c_s.x - c_s.y, c_s.x - c_s.z, c_s.y - c_s.z);
        let q_nw = vec3(c_nw.x - c_nw.y, c_nw.x - c_nw.z, c_nw.y - c_nw.z);
        let q_ne = vec3(c_ne.x - c_ne.y, c_ne.x - c_ne.z, c_ne.y - c_ne.z);
        let q_sw = vec3(c_sw.x - c_sw.y, c_sw.x - c_sw.z, c_sw.y - c_sw.z);
        let q_se = vec3(c_se.x - c_se.y, c_se.x - c_se.z, c_se.y - c_se.z);
        let d1_min = min(q_e, min(min(min(q_n, q_s), min(q_w, q_ea)), min(min(q_nw, q_ne), min(q_sw, q_se))));
        let d1_max = max(q_e, max(max(max(q_n, q_s), max(q_w, q_ea)), max(max(q_nw, q_ne), max(q_sw, q_se))));

        let dop_pad = vec3(mix(0.06, 0.005, motion_tighten));
        history_ycocg = taa_clip_18dop(history_ycocg, box_lo, box_hi,
            d0_min - dop_pad, d0_max + dop_pad, d1_min - dop_pad, d1_max + dop_pad);
    } else {
        history_ycocg = clamp(history_ycocg, box_lo, box_hi);
    }
    history_tm = taa_ycocg_to_rgb(history_ycocg);

    // Decima-style crisp motion clarity: reject fractional sub-pixel resampling blur
    // more decisively during panning so block textures stay sharp in motion, and blend
    // conservative spatial edge AA on partial-disocclusion/high-motion silhouettes.
    let pixel_offset = 1.0 - abs(2.0 * fract(res * prev_uv) - 1.0);
    var subpix_clarity = 1.0;
    if motion_px >= 0.08 {
        subpix_clarity = pow(max(pixel_offset.x * pixel_offset.y, 0.0), 0.675);
    }
    let offcenter = mix(subpix_clarity * 0.25 + 0.75, subpix_clarity * 0.42 + 0.58, motion_tighten);
    let base_alpha = mix(0.14, 0.08, distance_factor);
    let motion_alpha = mix(base_alpha, 0.52, smoothstep(0.8, 10.0, motion_px));
    var keep = (1.0 - motion_alpha) * offcenter * disocc_keep;
    if sky {
        let sky_move = clamp(smoothstep(0.015, 0.18, length(frame.camera_motion))
            + smoothstep(0.35, 2.5, motion_px), 0.0, 1.0);
        keep = min(keep, mix(0.62, 0.0, sky_move));
    }

    var cur_for_blend = current_tm;
    if spatial_aa && (disocc_keep < 0.70 || motion_px > 2.5) {
        let sa_rgb = taa_reinhard(spatial_aa_resolve(pixel, size, uv, current));
        let sa_w = clamp(max(1.0 - disocc_keep, smoothstep(2.5, 9.0, motion_px) * 0.65), 0.0, 1.0);
        cur_for_blend = mix(current_tm, sa_rgb, sa_w);
    }
    let blended_tm = mix(cur_for_blend, history_tm, keep);
    return taa_reinhard_inv(blended_tm);
}

fn compute(input: ComputeInput) {
    let size = vec2<i32>(textureDimensions(taa_history));
    let pixel = vec2<i32>(input.cell.xy);
    let local_xy = vec2<i32>(i32(input.local & 7u), i32(input.local >> 3u));
    let tile_origin = pixel - local_xy - vec2(1, 1);

    // Cooperatively preload the 10x10 shared tile in uniform control flow (100 loads / 64 threads).
    for (var idx = input.local; idx < 100u; idx += 64u) {
        let tx = i32(idx % 10u);
        let ty = i32(idx / 10u);
        let coord = clamp(tile_origin + vec2(tx, ty), vec2(0), size - 1);
        let rgb = textureLoad(taa_current, coord, 0).rgb;
        taa_tile_ycocg[idx] = taa_rgb_to_ycocg(taa_reinhard(rgb));
        taa_tile_depth[idx] = textureLoad(scene_depth, coord, 0);
    }
    workgroupBarrier();

    if !input.in_bounds || any(pixel >= size) { return; }
    let uv = (vec2<f32>(pixel) + 0.5) / vec2<f32>(size);
    let color = textureLoad(taa_current, pixel, 0).rgb;
    if !temporal_aa {
        let resolved = spatial_aa_resolve(pixel, size, uv, color);
        textureStore(taa_history, pixel, vec4(resolved, -1.0));
        return;
    }
    let depth = taa_tile_depth[taa_tile_idx(local_xy.x, local_xy.y)];
    let sky = depth <= 0.0;
    var position = vec3(0.0);
    var stored_dist = -1.0;
    if !sky {
        position = view_position(uv, depth);
        stored_dist = min(length(position), 65504.0);
    }
    let resolved = taa_resolve(pixel, local_xy, size, uv, depth, position, color);
    textureStore(taa_history, pixel, vec4(resolved, stored_dist));
}
