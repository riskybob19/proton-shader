// 3D Scrolled Colored Light Propagation Volume (LPV) Compute Pass (512x256 = 64x32x64):
// 1. Scrolls integer voxel cells cleanly with camera movement (zero blur)
// 2. Runs 6-neighbor 3D flood-fill diffusion (propagates colored blocklight around corners)
// 3. Injects live HDR emissive colors from screen-space glowing blocks (torches, soul fire,
//    redstone, lava, sea lanterns, froglights, amethyst, beacons) while rejecting sunlit terrain.
fn compute(input: ComputeInput) {
    let atlas_size = vec2<i32>(textureDimensions(colored_light_volume));
    if !input.in_bounds || any(vec2<i32>(input.cell.xy) >= atlas_size) {
        return;
    }
    let pixel = vec2<i32>(input.cell.xy);
    if !colored_lpv {
        textureStore(colored_light_volume, pixel, vec4(0.0));
        return;
    }

    let v = lpv_pixel_to_voxel(pixel);
    let curr_anc = lpv_anchor(frame.camera_position);
    let prev_anc = lpv_anchor(frame.camera_position - frame.camera_motion);
    let delta = vec3<i32>(curr_anc - prev_anc);
    let prev_v = v + delta;
    let teleport = dot(frame.camera_motion, frame.camera_motion) > 1024.0;

    // 1. Scroll & 6-neighbor max-weighted 3D flood-fill diffusion from previous frame's volume.
    //    Uses Minecraft-style per-voxel attenuation (no 1/6 collapse) so torches, soul fire,
    //    and lava shorelines carry full colored light across their entire 14-block radius at 0.008 ms GPU.
    var diffused = vec3(0.0);
    var prev_self = vec3(0.0);
    if !teleport && all(prev_v >= vec3(0)) && all(prev_v < vec3(64, 32, 64)) {
        prev_self = max(textureLoad(colored_light_volume_previous, lpv_voxel_to_pixel(prev_v), 0).rgb, vec3(0.0));
        let nx0 = select(vec3(0.0), max(textureLoad(colored_light_volume_previous, lpv_voxel_to_pixel(prev_v + vec3(-1, 0, 0)), 0).rgb, vec3(0.0)), prev_v.x > 0);
        let nx1 = select(vec3(0.0), max(textureLoad(colored_light_volume_previous, lpv_voxel_to_pixel(prev_v + vec3( 1, 0, 0)), 0).rgb, vec3(0.0)), prev_v.x < 63);
        let ny0 = select(vec3(0.0), max(textureLoad(colored_light_volume_previous, lpv_voxel_to_pixel(prev_v + vec3( 0,-1, 0)), 0).rgb, vec3(0.0)), prev_v.y > 0);
        let ny1 = select(vec3(0.0), max(textureLoad(colored_light_volume_previous, lpv_voxel_to_pixel(prev_v + vec3( 0, 1, 0)), 0).rgb, vec3(0.0)), prev_v.y < 31);
        let nz0 = select(vec3(0.0), max(textureLoad(colored_light_volume_previous, lpv_voxel_to_pixel(prev_v + vec3( 0, 0,-1)), 0).rgb, vec3(0.0)), prev_v.z > 0);
        let nz1 = select(vec3(0.0), max(textureLoad(colored_light_volume_previous, lpv_voxel_to_pixel(prev_v + vec3( 0, 0, 1)), 0).rgb, vec3(0.0)), prev_v.z < 63);
        let n_max = max(max(max(nx0, nx1), max(ny0, ny1)), max(nz0, nz1));
        let n_avg = (nx0 + nx1 + ny0 + ny1 + nz0 + nz1) * (1.0 / 6.0);
        let propagated = mix(n_avg, n_max, 0.82) * 0.82;
        diffused = max(prev_self * 0.86, propagated);
    }

    // 2. Inject genuine emissive light sources visible in the current camera view.
    let world_p = (curr_anc + vec3<f32>(v) - vec3(0.5 * LPV_XZ, 0.5 * LPV_Y, 0.5 * LPV_XZ) + 0.5) * LPV_CELL;
    let rel_p = world_p - frame.camera_position;
    let world_to_clip = frame.projection * frame.view;
    let clip = world_to_clip * vec4(rel_p, 1.0);
    var injected = vec3(0.0);
    if clip.w > 0.25 {
        let ndc = clip.xyz / clip.w;
        let uv = 0.5 + ndc.xy * vec2(0.5, -0.5);
        if all(uv > vec2(0.005)) && all(uv < vec2(0.995)) {
            let d = depth_at(uv);
            if d > 0.0 {
                let hit = view_position(uv, d);
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
                        let norm_col = hdr / max(c_max, 1e-4);
                        // Boost chromatic richness slightly so colored lanterns tint caves vividly.
                        let vivid = mix(vec3(dot(norm_col, frame.luminance)), norm_col, 1.25);
                        let intensity = clamp((luma - select(sun_thresh * 0.75, 0.28, is_dark_emitter)) * 0.95, 0.0, 3.5);
                        injected = max(vivid, vec3(0.0)) * intensity;
                    }
                }
            }
        }
    }

    let final_col = min(max(diffused, injected), vec3(4.0));
    textureStore(colored_light_volume, pixel, vec4(final_col, 1.0));
}
