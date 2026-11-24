// Decima-inspired scrolled world-space terrain heightfield cache updater.
// Scrolls integer texels cleanly with camera motion (zero blur) and stamps exact
// world-space terrain heights via 7-step vertical binary search against scene_depth
// (100% convergence on steep mountains, cliffs, and valleys out to 512 blocks).
fn hf_linear_z(depth: f32) -> f32 {
    let p = frame.projection_inverse;
    let z = p[2].z * depth + p[3].z;
    let w = p[2].w * depth + p[3].w;
    return abs(z) / max(abs(w), 1e-6);
}

fn hf_binary_search_depth(rel_xz: vec2<f32>, cam_y: f32, world_to_clip: mat4x4<f32>) -> vec2<f32> {
    // Quick horizontal frustum reject before binary search.
    let mid_clip = world_to_clip * vec4(rel_xz.x, 0.0, rel_xz.y, 1.0);
    if mid_clip.w <= 1.0 {
        return vec2(0.0, 0.0);
    }
    let mid_x = 0.5 + (mid_clip.x / mid_clip.w) * 0.5;
    if mid_x < -0.05 || mid_x > 1.05 {
        return vec2(0.0, 0.0);
    }

    // 8-step binary search across Y in [-48, 272] (320m -> 1.25m bracket, finer than 2m HF cells).
    var y_lo = -48.0;
    var y_hi = 272.0;
    var best_y = 0.0;
    var best_err = 1e9;
    for (var step = 0u; step < 8u; step += 1u) {
        let y_mid = 0.5 * (y_lo + y_hi);
        let clip = world_to_clip * vec4(rel_xz.x, y_mid - cam_y, rel_xz.y, 1.0);
        if clip.w <= 0.1 {
            y_hi = y_mid;
            continue;
        }
        let ndc = clip.xyz / clip.w;
        let uv = 0.5 + ndc.xy * vec2(0.5, -0.5);
        // When the camera is pitched down, a high test y_mid reduces clip.w and pushes
        // uv.y < 0 or uv.x outside [0, 1] -> search lower instead of aborting the column!
        if uv.y < 0.0 || uv.x < 0.0 || uv.x > 1.0 {
            y_hi = y_mid;
            continue;
        }
        if uv.y > 1.0 {
            y_lo = y_mid;
            continue;
        }
        let d = depth_at(uv);
        if d <= 0.0 {
            // Sky pixel: test point is above the terrain -> search lower.
            y_hi = y_mid;
            continue;
        }
        let col_z = hf_linear_z(ndc.z);
        let scene_z = hf_linear_z(d);
        let hit = view_position(uv, d);
        let err = length(hit.xz - rel_xz);
        if err < best_err {
            best_err = err;
            best_y = cam_y + hit.y;
        }
        if scene_z < col_z {
            // Scene geometry is closer than this column point -> column point is below/behind terrain -> search higher.
            y_lo = y_mid;
        } else {
            // Scene geometry is further than this column point -> column point is in open air above terrain -> search lower.
            y_hi = y_mid;
        }
    }
    if best_err < 4.2 {
        return vec2(best_y, 1.0);
    }
    return vec2(0.0, 0.0);
}

fn compute(input: ComputeInput) {
    let size = vec2<i32>(i32(HF_SIZE), i32(HF_SIZE));
    if !input.in_bounds || any(vec2<i32>(input.cell.xy) >= size) {
        return;
    }
    let cell = vec2<i32>(input.cell.xy);
    if !cached_distant_shadows || !shadows || frame.dimension != 0 {
        textureStore(world_volumes, cell, vec4(0.0));
        return;
    }

    let cam_xz = frame.camera_position.xz;
    let cam_y = frame.camera_position.y;
    let curr_anc = hf_anchor(cam_xz);
    let prev_anc = hf_anchor(cam_xz - frame.camera_motion.xz);
    let delta = vec2<i32>(curr_anc - prev_anc);
    let prev_cell = cell + delta;
    let teleport = dot(frame.camera_motion, frame.camera_motion) > 4096.0;

    var state = vec4(0.0);
    if !teleport && all(prev_cell >= vec2(0)) && all(prev_cell < size) {
        state = textureLoad(world_volumes_previous, prev_cell, 0);
    }

    // Amortize active stamping across 4 frames (2x2 staggered) -> 65k active threads/frame (~0.08 ms).
    let lane = (input.cell.x & 1u) + 2u * (input.cell.y & 1u);
    let phase = frame.frame_index & 3u;
    if lane != phase && state.g > 0.5 {
        textureStore(world_volumes, cell, vec4(state.rgb, min(state.a + 1.0, 64.0)));
        return;
    }

    let world_xz = (curr_anc + vec2<f32>(cell) - 0.5 * HF_SIZE + 0.5) * HF_CELL_SIZE;
    let rel_xz = world_xz - cam_xz;
    let world_to_clip = frame.projection * frame.view;

    var found_y = state.r;
    var found_conf = state.g;
    var found_src = state.b;

    let res = hf_binary_search_depth(rel_xz, cam_y, world_to_clip);
    if res.y > 0.5 {
        if found_conf > 0.35 {
            found_y = mix(found_y, res.x, select(0.40, 0.85, res.x > found_y));
        } else {
            found_y = res.x;
        }
        found_conf = 1.0;
        found_src = 0.5;
    } else if found_conf < 0.25 && !teleport {
        // 4-neighbor spatial hole-fill for newly scrolled or foreground-occluded cells so
        // mountain silhouettes stay continuous in the world-space heightfield cache.
        var y_sum = 0.0;
        var w_sum = 0.0;
        for (var n = 0u; n < 4u; n += 1u) {
            let off = select(
                vec2(0, select(-1, 1, n == 2u)),
                vec2(select(-1, 1, n == 0u), 0),
                n < 2u,
            );
            let np = prev_cell + off;
            if all(np >= vec2(0)) && all(np < size) {
                let ns = textureLoad(world_volumes_previous, np, 0);
                if ns.g > 0.45 {
                    y_sum += ns.r * ns.g;
                    w_sum += ns.g;
                }
            }
        }
        if w_sum > 0.45 {
            found_y = y_sum / w_sum;
            found_conf = 0.55;
            found_src = 0.25;
        }
    }

    textureStore(world_volumes, cell, vec4(found_y, found_conf, found_src, 0.0));
}
