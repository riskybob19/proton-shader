override underwater_marine_snow: bool = true;

fn marine_snow_hash3(p: vec3<f32>) -> vec3<f32> {
    var q = fract(p * vec3(0.1031, 0.1030, 0.0973));
    q += dot(q, q.yxz + 33.33);
    return fract((q.xxy + q.yxx) * q.zyx);
}

// Zero-VRAM 3D world-space parallax marine snow / drifting plankton illuminated
// by underwater sunlight and volumetric god-ray shafts.
fn underwater_marine_snow_glow(ray: vec3<f32>, max_dist: f32, shaft_tint: vec3<f32>) -> vec3<f32> {
    if !underwater_marine_snow || !use_underwater_fog() || max_dist < 0.8 {
        return vec3(0.0);
    }
    let access = smoothstep(0.02, 0.85, frame.eye_light_smooth.y);
    let sun_lit = direct_atmosphere_light() * (cloud_shadow(vec3(0.0)) * access);
    let amb_lit = photon_ambient_irradiance() * (0.28 * access)
        + vec3(0.14, 0.22, 0.26) * (frame.eye_light_smooth.x * frame.eye_light_smooth.x);
    let base_illum = sun_lit * 0.55 + amb_lit + shaft_tint * 1.35;
    if dot(base_illum, frame.luminance) < 1e-4 {
        return vec3(0.0);
    }

    // 3 concentric world-anchored parallax shells (3.2m, 6.8m, 12.5m)
    let shell_dists = array<f32, 3>(3.2, 6.8, 12.5);
    let cell_scales = array<f32, 3>(1.45, 1.05, 0.72);
    let radii = array<f32, 3>(0.055, 0.062, 0.070);
    var accum = 0.0;
    for (var s = 0u; s < 3u; s += 1u) {
        let d = shell_dists[s];
        if d >= max_dist - 0.25 {
            continue;
        }
        let drift = vec3(
            0.14 * sin(frame.time * 0.28 + f32(s) * 2.1),
            -0.065 * frame.time + 0.08 * cos(frame.time * 0.22 + f32(s)),
            0.14 * cos(frame.time * 0.24 + f32(s) * 1.7),
        );
        let world_p = frame.camera_position + ray * d + drift;
        let scaled = world_p * cell_scales[s];
        let cell = floor(scaled);
        let local = fract(scaled);
        let rnd = marine_snow_hash3(cell + vec3(f32(s) * 19.7));
        // Keep density tasteful (~38% of cells populated) so water feels alive, not noisy.
        if rnd.z > 0.38 {
            continue;
        }
        let center = 0.18 + 0.64 * rnd;
        let wobble = 0.06 * vec3(
            sin(frame.time * (0.8 + rnd.x) + rnd.y * TAU),
            cos(frame.time * (0.7 + rnd.y) + rnd.z * TAU),
            sin(frame.time * (0.9 + rnd.z) + rnd.x * TAU),
        );
        let dist_to_speck = length(local - (center + wobble));
        let speck = 1.0 - smoothstep(radii[s] * 0.35, radii[s], dist_to_speck);
        let depth_fade = exp(-0.095 * d) * smoothstep(d + 0.15, d + 1.2, max_dist);
        let twinkle = 0.55 + 0.45 * sin(frame.time * (1.8 + 2.0 * rnd.x) + rnd.y * TAU);
        accum += speck * depth_fade * twinkle * (0.55 + 0.45 * (1.0 - f32(s) * 0.25));
    }
    return base_illum * (accum * 0.26 * vec3(0.76, 0.96, 1.0));
}

fn post(input: PostInput) -> vec4<f32> {
    // Compute screen-space depth gradient in uniform control flow to guard the
    // hardware bilinear fog fast-path against silhouette depth cancellation.
    let depth_grad = abs(dpdx(input.depth)) + abs(dpdy(input.depth));
    // Horizon draws the End backdrop separately from its ordinary sky stage (dimension 1 = End).
    // Use translation-free infinity unprojection so view-bobbing eye translation never shifts the sky.
    if frame.dimension == 1 && input.depth <= 0.0 {
        let ndc = vec2(input.uv.x * 2.0 - 1.0, 1.0 - input.uv.y * 2.0);
        let direction = sky_world_ray(ndc);
        return vec4(end_sky(direction, input.color.rgb), 1.0);
    }
    let sky = input.depth <= 0.0;
    var base_input = input.color;
    // Composite full-resolution 1.0x screen-space clouds (cloud_screen_history) directly onto
    // Overworld open-sky pixels (all the way down to ray.y = -1.0 when above CLOUD_BASE),
    // preserving rain/snow/particles blended over sky.wgsl.
    if sky && frame.dimension == 0 && frame.eye_in < 2u && volumetric_clouds && cloud_cover > 0.0 {
        let sc_size = vec2<i32>(textureDimensions(cloud_screen_history));
        let sc_px = clamp(vec2<i32>(input.pixel), vec2(0), sc_size - 1);
        let sc_raw = unpack_u32_vec4(textureLoad(cloud_screen_history, sc_px, 0));
        if sc_raw.a >= 0.01 {
            let sc = vec4(unpack_hdr8(sc_raw.rgb), clamp((sc_raw.a - 0.02) * (1.0 / 0.98), 0.0, 1.0));
            let ndc = vec2(input.uv.x * 2.0 - 1.0, 1.0 - input.uv.y * 2.0);
            let ray = sky_world_ray(ndc);
            let above_clouds = frame.camera_position.y >= CLOUD_BASE - 8.0;
            if above_clouds || ray.y >= -0.05 {
                let base_atmo = atmosphere(ray);
                let aurora = aurora_volume(ray);
                let atmo_all = base_atmo + aurora;
                var celestial = vec3(0.0);
                if sc.a > 0.001 && ray.y > -0.05 {
                    celestial = sky_celestial(ray);
                }
                var sky_rainbow = vec3(0.0);
                let map_max = vec2<i32>(i32(ENV_WIDTH) - 1, i32(ENV_HEIGHT) - 1);
                if rainbows && frame.wetness > 0.0 && ray.y >= -0.072 {
                    let cell = clamp(vec2<i32>(cloud_map_texel(ray)), vec2(0), map_max);
                    let cloud_distance = textureLoad(cloud_aux, cell, 0).r;
                    let visible_distance = mix(cloud_distance, 2600.0, smoothstep(0.90, 1.0, sc.a));
                    sky_rainbow = rainbow_color(ray, visible_distance);
                }
                let flash = storm_flash();
                var sheet_lightning = vec3(0.0);
                if flash > 0.001 {
                    let strike_align = pow(clamp(dot(ray, storm_flash_dir()), 0.0, 1.0), 6.0);
                    sheet_lightning = vec3(0.62, 0.78, 1.15) * (flash * strike_align * (1.0 - sc.a) * 1.9);
                }
                let bg_sky = atmo_all + celestial;
                var native_sky = bg_sky * sc.a + sc.rgb + sheet_lightning + sky_rainbow;

                let sky_access = air_fog_sky_access();
                // At/above cloud altitude, ground-level horizon air fog sits physically BEHIND
                // foreground clouds (never painting a horizon stripe through opaque clouds).
                let cloud_above_fog = smoothstep(CLOUD_BASE - 36.0, CLOUD_BASE - 8.0, frame.camera_position.y);
                var haze = AirFog(vec3(0.0), vec3(1.0));
                if fog_strength > 0.0 && sky_access > 0.0 {
                    haze = raymarch_air_fog(ray, max(frame.render_distance, 32.0), 0.5, sky_access, true);
                    let bg_hazed = bg_sky * haze.transmittance + haze.scattering;
                    let cloud_front = bg_hazed * sc.a + sc.rgb + sheet_lightning + sky_rainbow;
                    let fog_front = native_sky * haze.transmittance + haze.scattering;
                    native_sky = mix(fog_front, cloud_front, cloud_above_fog);
                }
                var final_sky = native_sky;
                // Only extract rain/snow streaks from input.color during active precipitation,
                // clamping delta to prevent cubic-vs-bilinear cloud-edge mismatch from punching dark dots.
                if frame.rain > 0.001 {
                    var pano_cloud = vec3(0.0);
                    var pano_a = 1.0;
                    if ray.y >= -0.075 && !(above_clouds && ray.y <= -0.072) {
                        let pano_raw = sample_environment(ray);
                        let p_fade = select(0.0, 1.0 - smoothstep(-0.072, -0.032, ray.y), above_clouds);
                        pano_cloud = max(pano_raw.rgb - base_atmo * pano_raw.a, vec3(0.0)) * (1.0 - p_fade);
                        pano_a = mix(pano_raw.a, 1.0, p_fade);
                    }
                    let pano_cel = select(vec3(0.0), sky_celestial(ray), pano_a > 0.001 && ray.y > -0.05);
                    let pano_bg = base_atmo + pano_cel;
                    var pano_sky = pano_bg * pano_a + pano_cloud + sheet_lightning + sky_rainbow;
                    if fog_strength > 0.0 && sky_access > 0.0 {
                        let p_bg_hazed = pano_bg * haze.transmittance + haze.scattering;
                        let p_cloud_front = p_bg_hazed * pano_a + pano_cloud + sheet_lightning + sky_rainbow;
                        let p_fog_front = pano_sky * haze.transmittance + haze.scattering;
                        pano_sky = mix(p_fog_front, p_cloud_front, cloud_above_fog);
                    }
                    let rain_delta = max(input.color.rgb - pano_sky, vec3(0.0));
                    final_sky = native_sky + rain_delta;
                }
                base_input = vec4(final_sky, 1.0);
            }
        }
    }
    if sky && !use_underwater_fog() {
        return base_input;
    }
    let world = fog_position(input.uv, input.depth);
    let ray_length = length(world);
    var terrain = base_input;
    if !sky {
        terrain = terrain_effects(input, world, ray_length);
    }
    if !use_volume_composite() {
        return terrain;
    }
    let ray = world / max(ray_length, 1e-6);

    let size = vec2<i32>(textureDimensions(fog_transmittance));
    let size_f = vec2<f32>(size);
    let inv_size = 1.0 / size_f;
    let distance = fog_distance(ray_length, input.depth);
    let pos_half = input.uv * size_f - 0.5;
    let corner = vec2<i32>(floor(pos_half));
    let blend = fract(pos_half);

    // Single-check hardware bilinear fast-path validated directly against scene_depth at
    // the half-resolution texel centers (avoids missing alpha on Rg11b10Ufloat targets).
    var rainbow = vec3(0.0);
    if !sky && frame.eye_in == 0u && distance > 250.0 {
        rainbow = rainbow_color(ray, distance);
    }
    var tap_depths = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var all_match = sky || depth_grad < max(input.depth * 0.06, 1e-5);
    for (var tap = 0u; tap < 4u; tap += 1u) {
        let offset = vec2<i32>(i32(tap & 1u), i32(tap >> 1u));
        let cell = clamp(corner + offset, vec2(0), size - 1);
        let td = depth_at((vec2<f32>(cell) + 0.5) * inv_size);
        tap_depths[tap] = td;
        if (td <= 0.0) != sky || (!sky && abs(td - input.depth) > 0.08 * max(input.depth, 1e-8)) {
            all_match = false;
        }
    }

    var color = vec3(0.0);
    var scattering = vec3(0.0);
    var total = 0.0;
    var best_scattering = vec3(0.0);
    var resolved_scatter = vec3(0.0);
    var resolved_trans = vec3(1.0);
    if all_match {
        let bt = textureSampleLevel(fog_transmittance, linear_sampler, input.uv, 0.0).rgb;
        let bs = textureSampleLevel(fog_scattering, linear_sampler, input.uv, 0.0).rgb;
        let cloud_vis = clamp(dot(bt, vec3(0.333333)), 0.0, 1.0);
        color = terrain.rgb * bt + bs + rainbow * cloud_vis;
        best_scattering = bs;
        resolved_scatter = bs;
        resolved_trans = bt;
        total = -1.0;
    }
    if total == 0.0 {
        var transmittance = vec3(0.0);
        var best_diff = 1e30;
        var best_transmittance = vec3(1.0);
        for (var tap = 0u; tap < 4u; tap += 1u) {
            let td = tap_depths[tap];
            if (td <= 0.0) == sky {
                let offset = vec2<i32>(i32(tap & 1u), i32(tap >> 1u));
                let diff = select(abs(td - input.depth) / max(input.depth, 1e-8), 0.0, sky);
                let axis_weight = select(1.0 - blend, blend, offset == vec2(1));
                let depth_weight = select(1.0 - smoothstep(0.0, 0.16, diff), 1.0, sky);
                let weight = axis_weight.x * axis_weight.y * depth_weight;
                let is_best = diff < best_diff;
                if weight <= 0.0 && !is_best {
                    continue;
                }
                let cell = clamp(corner + offset, vec2(0), size - 1);
                let sample_t = textureLoad(fog_transmittance, cell, 0).rgb;
                let volume = textureLoad(fog_scattering, cell, 0).rgb;
                if is_best {
                    best_diff = diff;
                    best_transmittance = sample_t;
                    best_scattering = volume;
                }
                if weight <= 0.0 {
                    continue;
                }
                transmittance += weight * sample_t;
                scattering += weight * volume;
                total += weight;
            }
        }
        if total > 0.001 {
            let norm_t = transmittance / total;
            resolved_scatter = scattering / total;
            resolved_trans = norm_t;
            let cloud_vis = clamp(dot(norm_t, vec3(0.333333)), 0.0, 1.0);
            color = terrain.rgb * norm_t + resolved_scatter + rainbow * cloud_vis;
        } else if best_diff < 0.35 {
            resolved_scatter = best_scattering;
            resolved_trans = best_transmittance;
            let cloud_vis = clamp(dot(best_transmittance, vec3(0.333333)), 0.0, 1.0);
            color = terrain.rgb * best_transmittance + best_scattering + rainbow * cloud_vis;
        } else if use_underwater_fog() {
            let fog = raymarch_underwater(ray, distance, 0.5);
            resolved_trans = fog.transmittance;
            color = terrain.rgb * fog.transmittance + fog.scattering;
        } else if !sky && fog_strength > 0.0 && air_fog_sky_access() > 0.0 {
            let fog = raymarch_air_fog(ray, distance, 0.5, air_fog_sky_access(), true);
            resolved_trans = fog.transmittance;
            color = terrain.rgb * fog.transmittance + fog.scattering + rainbow;
        } else {
            color = terrain.rgb + rainbow;
        }
    }
    if use_underwater_fog() {
        let shaft_hint = select(best_scattering, scattering / max(total, 1e-4), total > 0.001);
        color += underwater_marine_snow_glow(ray, distance, shaft_hint);
    } else if !sky && frame.dimension == 0 && frame.eye_in == 0u
        && frame.blindness <= 0.0 && frame.darkness <= 0.0 {
        // Upstream Photon border fog (simple_fog.glsl + d4_deferred_shading.fsh):
        // Uses cubic_length squircle matching Minecraft's chunk loading grid with exp2(-8 * r^8)
        // falloff, tall-mountain preservation above the horizon, and horizon-clamped sky color.
        let sky_access = air_fog_sky_access();
        let end = min(max(frame.fog_render.y, 16.0), max(frame.render_distance, 32.0));
        let ax = abs(world.x);
        let az = abs(world.z);
        let cubic_xz = pow(ax * ax * ax + az * az * az, 1.0 / 3.0);
        let r_ratio = cubic_xz / end;
        let r2 = r_ratio * r_ratio;
        let r4 = r2 * r2;
        var border_vis = exp2(-8.0 * r4 * r4);
        border_vis = mix(border_vis, 1.0, 0.75 * photon_dampen(clamp(ray.y * 5.0, 0.0, 1.0)));
        border_vis *= clamp((world.y + end) * (1.0 / 16.0), 0.0, 1.0);
        let border = 1.0 - clamp(border_vis, 0.0, 1.0);
        if border > 0.002 && sky_access > 0.001 {
            let atmo_only = atmosphere(ray);
            let horizon_dir = normalize(vec3(ray.x, min(ray.y, -0.1), ray.z));
            let horizon_atmo = atmosphere(horizon_dir);
            let sun_twilight = clamp((exp(-75.0 * (frame.sun_direction.y + 0.0496) * (frame.sun_direction.y + 0.0496)) - 0.1) * (1.0 / 0.9), 0.0, 1.0);
            var horizon_factor = clamp(sun_twilight + select(0.0, 1.0, frame.rain > 0.01), 0.0, 1.0);
            horizon_factor = max(horizon_factor, photon_dampen(clamp((0.15 - ray.y) * 10.0, 0.0, 1.0)));
            let hf2 = horizon_factor * horizon_factor;
            var border_fog_color = mix(atmo_only, horizon_atmo, hf2);
            if fog_strength > 0.0 {
                let sky_dist = max(frame.render_distance, 32.0);
                let rim_haze = raymarch_air_fog(ray, sky_dist, 0.5, sky_access, true);
                border_fog_color = border_fog_color * rim_haze.transmittance + rim_haze.scattering;
            }
            let cave = (1.0 - smoothstep(49.0, 63.0, frame.camera_position.y))
                * (1.0 - smoothstep(0.03, 0.30, frame.eye_light_smooth.y));
            color = mix(color, border_fog_color * (1.0 - cave), border * sky_access);
        }
    }
    // Composite unified native 1.0x screen-space clouds cleanly in front of terrain + fog + border
    // whenever clouds sit between camera and terrain (eliminates terrain-behind-cloud seam & 9-step rings).
    if !sky && frame.dimension == 0 && frame.eye_in < 2u && volumetric_clouds && cloud_cover > 0.0 {
        let sc_size = vec2<i32>(textureDimensions(cloud_screen_history));
        let sc_px = clamp(vec2<i32>(input.pixel), vec2(0), sc_size - 1);
        let sc_raw = unpack_u32_vec4(textureLoad(cloud_screen_history, sc_px, 0));
        if sc_raw.a >= 0.01 {
            let sc = vec4(unpack_hdr8(sc_raw.rgb), clamp((sc_raw.a - 0.02) * (1.0 / 0.98), 0.0, 1.0));
            color = color * sc.a + sc.rgb;
        }
    }
    return vec4(color, 1.0);
}
