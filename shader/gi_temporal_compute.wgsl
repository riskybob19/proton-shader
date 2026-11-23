// Dedicated half-resolution SVGF-inspired Temporal GI + Variance-Guided Shadow Denoiser:
//   .rgb = Motion-reprojected, 5-tap variance-clamped 3-layer Hybrid Voxel GI (indirect_gi)
//   .a   = Motion-reprojected, moment/variance-clamped + depth-aware denoised distant shadow
//          (preserves razor-sharp <24m contact shadows while eliminating distant shadow dither).
fn compute(input: ComputeInput) {
    let size = vec2<i32>(textureDimensions(gi_history));
    if !input.in_bounds || any(vec2<i32>(input.cell.xy) >= size) {
        return;
    }
    let cell = vec2<i32>(input.cell.xy);
    if frame.dimension != 0 {
        textureStore(gi_history, cell, vec4(0.0));
        return;
    }

    let cur_gi = textureLoad(indirect_gi, cell, 0);
    let cur_ao = textureLoad(contact_visibility, cell, 0);
    let cur_shadow = cur_ao.b;

    let inv_size = 1.0 / vec2<f32>(size);
    let uv = (vec2<f32>(cell) + 0.5) * inv_size;
    let depth = depth_at(uv);
    if depth <= 0.0 {
        textureStore(gi_history, cell, vec4(0.0));
        return;
    }

    let pos = view_position(uv, depth);
    let view_dist = length(pos);

    // 5-tap cross neighborhood statistics for GI variance clamp + SVGF shadow moment estimation.
    let c0 = max(cur_gi.rgb, vec3(0.0));
    let cell_l = clamp(cell + vec2(-1, 0), vec2(0), size - 1);
    let cell_r = clamp(cell + vec2(1, 0), vec2(0), size - 1);
    let cell_d = clamp(cell + vec2(0, -1), vec2(0), size - 1);
    let cell_u = clamp(cell + vec2(0, 1), vec2(0), size - 1);

    let cl = max(textureLoad(indirect_gi, cell_l, 0).rgb, vec3(0.0));
    let cr = max(textureLoad(indirect_gi, cell_r, 0).rgb, vec3(0.0));
    let cd = max(textureLoad(indirect_gi, cell_d, 0).rgb, vec3(0.0));
    let cu = max(textureLoad(indirect_gi, cell_u, 0).rgb, vec3(0.0));
    let mu = (c0 + cl + cr + cd + cu) * 0.2;
    let m2 = (c0 * c0 + cl * cl + cr * cr + cd * cd + cu * cu) * 0.2;
    let sigma = sqrt(max(m2 - mu * mu, vec3(0.0)));
    let gi_min = max(mu - sigma * 1.35, vec3(0.0));
    let gi_max = mu + sigma * 1.35 + vec3(0.015);

    let sl = textureLoad(contact_visibility, cell_l, 0).b;
    let sr = textureLoad(contact_visibility, cell_r, 0).b;
    let sd = textureLoad(contact_visibility, cell_d, 0).b;
    let su = textureLoad(contact_visibility, cell_u, 0).b;
    let sh_mu = (cur_shadow + sl + sr + sd + su) * 0.2;
    let sh_m2 = (cur_shadow * cur_shadow + sl * sl + sr * sr + sd * sd + su * su) * 0.2;
    let sh_sigma = sqrt(max(sh_m2 - sh_mu * sh_mu, 0.0));

    // AMD FidelityFX-style variance-guided spatial pre-filter for jittered distant shadows:
    // skips uniform lit/shadowed tiles (sh_sigma <= 0.05), preserves sharp terrain edges via depth weights,
    // and down-weights 0.0 vertical terrace risers when cur_shadow > 0.25 so terraced hillsides never stripe.
    var filtered_sh = cur_shadow;
    if view_dist >= 88.0 && sh_sigma > 0.05 {
        let dl = depth_at((vec2<f32>(cell_l) + 0.5) * inv_size);
        let dr = depth_at((vec2<f32>(cell_r) + 0.5) * inv_size);
        let dd = depth_at((vec2<f32>(cell_d) + 0.5) * inv_size);
        let du = depth_at((vec2<f32>(cell_u) + 0.5) * inv_size);
        let tol = max(depth * 0.08, 1e-6);
        let riser_guard = step(0.25, cur_shadow);
        let wl = (1.0 - smoothstep(0.0, tol, abs(dl - depth))) * mix(1.0, step(0.05, sl), riser_guard * 0.85);
        let wr = (1.0 - smoothstep(0.0, tol, abs(dr - depth))) * mix(1.0, step(0.05, sr), riser_guard * 0.85);
        let wd = (1.0 - smoothstep(0.0, tol, abs(dd - depth))) * mix(1.0, step(0.05, sd), riser_guard * 0.85);
        let wu = (1.0 - smoothstep(0.0, tol, abs(du - depth))) * mix(1.0, step(0.05, su), riser_guard * 0.85);
        let w_tot = 1.2 + wl + wr + wd + wu;
        let spatial_avg = (cur_shadow * 1.2 + sl * wl + sr * wr + sd * wd + su * wu) / w_tot;
        let dark_preserve = max(spatial_avg, cur_shadow * 0.90);
        filtered_sh = mix(cur_shadow, dark_preserve, smoothstep(0.05, 0.24, sh_sigma) * 0.70);
    }

    let prev_clip = frame.previous_view_projection * vec4(pos + frame.camera_motion, 1.0);
    let valid_proj = prev_clip.w > 0.05 && frame.delta_time > 0.0 && frame.delta_time < 0.20;
    let prev_ndc = prev_clip.xyz / max(prev_clip.w, 1e-5);
    let prev_uv = 0.5 + prev_ndc.xy * vec2(0.5, -0.5);
    let in_bounds = valid_proj && all(prev_uv > vec2(0.002)) && all(prev_uv < vec2(0.998));

    if !in_bounds {
        textureStore(gi_history, cell, vec4(c0, filtered_sh));
        return;
    }

    let hist = textureSampleLevel(gi_history_previous, linear_sampler, prev_uv, 0.0);
    let clamped_gi = clamp(max(hist.rgb, vec3(0.0)), gi_min, gi_max);
    // SVGF moment-guided shadow clamp: keep upper clamp generous when cur_shadow is active
    // so alternating 0.0 vertical terrace risers never drag temporal shadow history down.
    let sh_min = max(max(sh_mu - sh_sigma * 1.35 - 0.02, cur_shadow * 0.78), 0.0);
    let sh_max = min(max(sh_mu + sh_sigma * 1.35 + 0.02, cur_shadow), 1.0);
    let clamped_sh = clamp(hist.a, sh_min, sh_max);

    let motion_px = length((uv - prev_uv) * vec2<f32>(size));
    let gi_alpha = mix(0.14, 0.42, clamp(motion_px * 0.18, 0.0, 1.0));
    // Keep near-field (<24m) contact shadows instant/razor-sharp; accumulate distant shadows smoothly.
    let sh_alpha = select(1.0, mix(0.16, 0.45, clamp(motion_px * 0.16, 0.0, 1.0)), view_dist >= 24.0);

    let out_gi = mix(clamped_gi, c0, gi_alpha);
    let out_sh = mix(clamped_sh, filtered_sh, sh_alpha);
    textureStore(gi_history, cell, vec4(out_gi, out_sh));
}
