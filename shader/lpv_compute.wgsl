// 3D Scrolled Colored Light Propagation Volume (LPV) Compute Pass (512x256 = 64x32x64 @ 2.0m/voxel):
// 1. Scrolls integer voxel cells cleanly with camera movement (zero blur)
// 2. Runs 26-neighbor Euclidean (L2) weighted propagation (exp(-alpha * d) over d in {1, sqrt(2), sqrt(3)})
//    with solid-wall blockers stored in .a — replacing L1 taxicab diamonds with true spherical pools
// 3. Injects live HDR emissive colors from screen-space glowing blocks & preserves off-screen emitters
fn compute(input: ComputeInput) {
    let atlas_size = vec2<i32>(i32(LPV_ATLAS_W), i32(LPV_ATLAS_H));
    if !input.in_bounds || any(vec2<i32>(input.cell.xy) >= atlas_size) {
        return;
    }
    let pixel = vec2<i32>(input.cell.xy);
    let store_px = vec2(pixel.x, pixel.y + LPV_ATLAS_Y_OFFSET);
    let rt_needs_lpv = blocklight_mode != 0u && frame.has_ray_tracing != 0u && frame.dimension == 0;
    if !colored_lpv && !rt_needs_lpv {
        textureStore(world_volumes, store_px, vec4(0.0));
        return;
    }

    let v = lpv_pixel_to_voxel(pixel);
    let curr_anc = lpv_anchor(frame.camera_position);
    let prev_anc = lpv_anchor(frame.camera_position - frame.camera_motion);
    let delta = vec3<i32>(curr_anc - prev_anc);
    let prev_v = v + delta;
    let teleport = dot(frame.camera_motion, frame.camera_motion) > 1024.0;

    // 1. Scroll & 26-neighbor isotropic L2 Euclidean propagation with .a solid wall occlusion.
    var diffused = vec3(0.0);
    var prev_self = vec3(0.0);
    var prev_wall = 0.0;
    if !teleport && all(prev_v >= vec3(0)) && all(prev_v < vec3(64, 32, 64)) {
        let self_tex = textureLoad(world_volumes_previous, lpv_voxel_to_pixel(prev_v), 0);
        prev_self = max(self_tex.rgb, vec3(0.0));
        prev_wall = clamp(self_tex.a, 0.0, 1.0);

        var n_sum = vec3(0.0);
        var w_sum = 0.0;
        var n_max = vec3(0.0);
        for (var dz = -1; dz <= 1; dz += 1) {
            let nz = prev_v.z + dz;
            if nz < 0 || nz > 63 {
                continue;
            }
            for (var dy = -1; dy <= 1; dy += 1) {
                let ny = prev_v.y + dy;
                if ny < 0 || ny > 31 {
                    continue;
                }
                for (var dx = -1; dx <= 1; dx += 1) {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let nx = prev_v.x + dx;
                    if nx < 0 || nx > 63 {
                        continue;
                    }
                    let n_tex = textureLoad(world_volumes_previous, lpv_voxel_to_pixel(vec3(nx, ny, nz)), 0);
                    let d2 = f32(dx * dx + dy * dy + dz * dz);
                    let dist = sqrt(d2);
                    // Exact L2 radial falloff: exp(-0.18 * d) -> 0.835 (face), 0.775 (edge), 0.732 (corner).
                    let radial = pow(0.835, dist);
                    let blocker = 1.0 - smoothstep(0.45, 0.92, clamp(n_tex.a, 0.0, 1.0)) * 0.78;
                    let energy = max(n_tex.rgb, vec3(0.0)) * (radial * blocker);
                    let iso_w = 1.0 / dist;
                    n_sum += energy * iso_w;
                    w_sum += iso_w;
                    n_max = max(n_max, energy);
                }
            }
        }
        let self_blocker = 1.0 - smoothstep(0.55, 0.95, prev_wall) * 0.45;
        let avg_26 = select(vec3(0.0), n_sum / max(w_sum, 1e-4), w_sum > 0.0);
        let propagated = mix(avg_26, n_max, 0.78) * self_blocker;
        diffused = max(prev_self * 0.86, propagated);
    }

    // 2. Inject genuine emissive light sources & stamp 3D solid wall blockers from scene_depth.
    let world_p = (curr_anc + vec3<f32>(v) - vec3(0.5 * LPV_XZ, 0.5 * LPV_Y, 0.5 * LPV_XZ) + 0.5) * LPV_CELL;
    let rel_p = world_p - frame.camera_position;
    let world_to_clip = frame.projection * frame.view;
    let clip = world_to_clip * vec4(rel_p, 1.0);
    var injected = vec3(0.0);
    var wall_occ = prev_wall * 0.985;
    var on_screen = false;
    var saw_emitter = false;
    if clip.w > 0.25 {
        let ndc = clip.xyz / clip.w;
        let uv = 0.5 + ndc.xy * vec2(0.5, -0.5);
        if all(uv > vec2(0.005)) && all(uv < vec2(0.995)) {
            on_screen = true;
            let d = depth_at(uv);
            if d > 0.0 {
                let hit = view_position(uv, d);
                let hit_dist = length(hit);
                let rel_dist = length(rel_p);
                if length(hit - rel_p) < LPV_CELL * 1.15 {
                    let hdr = max(textureSampleLevel(taa_history_previous, linear_sampler, uv, 0.0).rgb, vec3(0.0));
                    let luma = dot(hdr, frame.luminance);
                    let c_max = max(hdr.r, max(hdr.g, hdr.b));
                    let c_min = min(hdr.r, min(hdr.g, hdr.b));
                    let sat = (c_max - c_min) / max(c_max, 1e-4);

                    let full_size = vec2<f32>(textureDimensions(scene_depth));
                    let surf_cell = clamp(vec2<i32>(uv * full_size), vec2(0), vec2<i32>(full_size) - 1);
                    let surf = surface_at(surf_cell);
                    let packed = u32(surf.data);
                    let surf_solid = surf.present && !surface_is_lava(packed);
                    let direct_frac = select(0.0, surface_direct(packed), surf_solid);
                    let sky_lv = select(0.0, surface_sky(packed), surf_solid);

                    // Strict emissive detector: rejects sunlit white snow/sand outdoors while
                    // capturing torches, lava, soul lanterns, redstone, froglights, sea lanterns,
                    // glowing ores, Nether shroomlights, and End rods/crystals.
                    let day_sun = smoothstep(0.0, 0.25, frame.sun_direction.y) * select(0.0, 1.0, frame.dimension == 0);
                    let sun_thresh = mix(0.55, 2.65, day_sun * sky_lv * direct_frac);
                    let is_dark_emitter = (direct_frac < 0.15 || day_sun < 0.1) && luma > 0.42;
                    let is_bright_emitter = luma > sun_thresh && (sat > 0.18 || luma > sun_thresh * 1.45);
                    if is_dark_emitter || is_bright_emitter {
                        saw_emitter = true;
                        let norm_col = hdr / max(c_max, 1e-4);
                        let vivid = mix(vec3(dot(norm_col, frame.luminance)), norm_col, 1.25);
                        let intensity = clamp((luma - select(sun_thresh * 0.75, 0.28, is_dark_emitter)) * 0.95, 0.0, 3.5);
                        injected = max(vivid, vec3(0.0)) * intensity;
                        wall_occ = 0.0;
                    } else {
                        // Non-emissive solid surface inside this voxel acts as a light blocker.
                        wall_occ = min(prev_wall * 0.85 + 0.48, 0.95);
                    }
                } else if hit_dist - rel_dist > LPV_CELL * 0.75 {
                    // Open air in front of terrain: clear stale solid blockers.
                    wall_occ = prev_wall * 0.45;
                }
            } else {
                // Open sky line of sight: clear stale blockers.
                wall_occ = prev_wall * 0.45;
            }
        }
    }

    // Hold off-screen emitters gently (~3s persistence) so torches just behind or above
    // the camera continue feeding spherical L2 light & RT ReSTIR candidates without popping.
    let persistent_offscreen = select(vec3(0.0), prev_self * 0.985, !on_screen && max(prev_self.r, max(prev_self.g, prev_self.b)) > 0.35);
    let visible_decay = select(diffused, diffused * 0.90, on_screen && !saw_emitter && length(injected) < 0.01);
    let final_col = min(max(max(visible_decay, persistent_offscreen), injected), vec3(4.0));
    textureStore(world_volumes, store_px, vec4(final_col, clamp(wall_occ, 0.0, 1.0)));
}
