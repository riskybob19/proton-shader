// The sky stage runs without terrain shadow bindings.
fn fog_shadow(_position: vec3<f32>) -> vec3<f32> {
    return vec3(1.0);
}

fn cloud_coverage(world: vec2<f32>) -> f32 {
    return cumulus_coverage(world);
}

// 4-tap cubic B-spline reconstruction over the 1024x512 upper-hemisphere cloud map.
// Sampled with Horizon's unbobbed SkyInput.direction so clouds stay 100% level while walking.
fn sky_environment(direction: vec3<f32>) -> vec4<f32> {
    let map_size = vec2<f32>(textureDimensions(cloud_environment_map));
    let texel = cloud_map_texel(direction) - 0.5;
    let base = floor(texel) + 0.5;
    let f = fract(texel);
    let f2 = f * f;
    let f3 = f2 * f;
    let inv = 1.0 - f;
    let w0 = inv * inv * inv / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let near_weight = w0 + w1;
    let far_weight = w2 + w3;
    let near = base - w0 / near_weight;
    let far = base + 1.0 + w3 / far_weight;
    let uv00 = vec2(near.x, near.y) / map_size;
    let uv10 = vec2(far.x, near.y) / map_size;
    let uv01 = vec2(near.x, far.y) / map_size;
    let uv11 = vec2(far.x, far.y) / map_size;
    let s00 = textureSampleLevel(cloud_environment_map, scattering_atlas_sampler, uv00, 0.0);
    let s10 = textureSampleLevel(cloud_environment_map, scattering_atlas_sampler, uv10, 0.0);
    let s01 = textureSampleLevel(cloud_environment_map, scattering_atlas_sampler, uv01, 0.0);
    let s11 = textureSampleLevel(cloud_environment_map, scattering_atlas_sampler, uv11, 0.0);
    let top = s00 * (near_weight.x * near_weight.y) + s10 * (far_weight.x * near_weight.y);
    let bot = s01 * (near_weight.x * far_weight.y) + s11 * (far_weight.x * far_weight.y);
    let cubic = max(top + bot, vec4(0.0));
    // Adaptive cloud-edge denoiser: zero extra fetches on clear sky or solid
    // cloud cores; on semi-transparent silhouettes, sample 4 wider diagonal taps
    // to dissolve remaining dither grain into silky-smooth cloud edges.
    let min_t = min(min(s00.a, s10.a), min(s01.a, s11.a));
    let max_t = max(max(s00.a, s10.a), max(s01.a, s11.a));
    let edge_spread = max_t - min_t;
    let semi_transparent = cubic.a > 0.06 && cubic.a < 0.94;
    if edge_spread < 0.045 && !semi_transparent {
        return cubic;
    }
    let center_uv = (texel + 0.5) / map_size;
    // Motion-damped R2 temporal dither (RDR2 / Decima): dissolve cloud-map texel steps
    // via TAA when stationary, and damp jitter smoothly to 0 during camera panning/motion
    // so moving cloud silhouettes stay rock-solid instead of boiling.
    let clip = frame.projection * frame.view * vec4(direction, 0.0);
    let prev_clip = frame.previous_view_projection * vec4(direction, 0.0);
    let screen_uv = 0.5 + clip.xy / max(abs(clip.w), 1e-4) * vec2(0.5, -0.5);
    let prev_uv = 0.5 + prev_clip.xy / max(abs(prev_clip.w), 1e-4) * vec2(0.5, -0.5);
    let screen_px = screen_uv * frame.resolution;
    let rot_px = length((screen_uv - prev_uv) * frame.resolution);
    let move_damp = 1.0 - smoothstep(0.4, 3.5, rot_px + length(frame.camera_motion) * 18.0);
    let ign = fract(52.9829189 * fract(dot(floor(screen_px), vec2(0.06711056, 0.00583715))));
    let r2_t = f32(frame.frame_index & 63u);
    let dither = fract(vec2(ign, fract(ign * 1.61803399 + 0.3719)) + r2_t * vec2(0.7548777, 0.5698403)) - 0.5;
    let jitter_uv = dither * (vec2(1.15, 1.05) * move_damp / map_size);

    // 4-tap Dual-Kawase diagonal cloud-edge denoiser reusing the cubic B-spline center
    // (saves a full-res texture fetch on every cloud silhouette pixel vs 5-tap Gaussian).
    let step_uv = vec2(1.65, 1.45) / map_size;
    let d0 = textureSampleLevel(cloud_environment_map, scattering_atlas_sampler, center_uv + jitter_uv + vec2(-step_uv.x, -step_uv.y), 0.0);
    let d1 = textureSampleLevel(cloud_environment_map, scattering_atlas_sampler, center_uv + jitter_uv + vec2(step_uv.x, -step_uv.y), 0.0);
    let d2 = textureSampleLevel(cloud_environment_map, scattering_atlas_sampler, center_uv + jitter_uv + vec2(-step_uv.x, step_uv.y), 0.0);
    let d3 = textureSampleLevel(cloud_environment_map, scattering_atlas_sampler, center_uv + jitter_uv + vec2(step_uv.x, step_uv.y), 0.0);
    let inner_avg = 0.25 * (s00 + s10 + s01 + s11);
    let outer_avg = 0.25 * (d0 + d1 + d2 + d3);
    let kawase = 0.40 * cubic + 0.30 * inner_avg + 0.30 * outer_avg;
    let edge_mask = max(
        smoothstep(0.015, 0.20, edge_spread),
        smoothstep(0.03, 0.25, cubic.a) * (1.0 - smoothstep(0.75, 0.97, cubic.a)),
    );
    return max(mix(cubic, kawase, edge_mask * 0.92), vec4(0.0));
}

fn sky(input: SkyInput) -> vec4<f32> {
    let ray = input.direction;
    if frame.dimension == -1 {
        // Volcanic ash-drifting Nether sky with lower-horizon molten lava glow.
        let horizon = 1.0 - smoothstep(0.0, 0.72, abs(ray.y));
        let fog_tint = pow(max(frame.fog_color, vec3(0.0)), vec3(2.2));
        let wisp_uv = ray.xz / (abs(ray.y) + 0.35) * 0.14 + frame.time * vec2(0.004, -0.003);
        let wisp = textureSampleLevel(wave_noise, wave_noise_sampler, wisp_uv, 0.0).g;
        let under_glow = smoothstep(0.18, -0.35, ray.y) * vec3(0.14, 0.036, 0.008);
        let ember = mix(fog_tint * 0.42, fog_tint * 1.12 + vec3(0.075, 0.022, 0.007) * (0.75 + 0.5 * wisp), horizon) + under_glow;
        return vec4(ember, 1.0);
    }
    if frame.dimension == 1 {
        let base_end = mix(vec3(0.008, 0.006, 0.018), vec3(0.032, 0.020, 0.055),
            1.0 - smoothstep(0.0, 0.65, abs(ray.y)));
        return vec4(end_sky(ray, base_end), 1.0);
    }
    if frame.dimension != 0 {
        return input.color;
    }

    // Keep volumetric clouds visible when eye_in == 1u (water / flipped mirror camera);
    // only skip clouds inside opaque lava or powder snow (eye_in >= 2u).
    if frame.eye_in >= 2u {
        return vec4(atmosphere(ray) + sky_celestial(ray), 1.0);
    }
    // Sample with Horizon's unbobbed sky ray remapped to the physical
    // render-distance horizon (y = 0 sits cleanly on the furthest loaded terrain).
    let sky_ray = sky_dome_ray(ray);
    // Horizon draws sky across all 1080p pixels before opaque terrain; rays below
    // y = -0.05 have no clouds/celestials, so skip the 4-8 cloud fetches while still
    // applying analytic horizon haze (zero seam in skyblock/void/high-altitude flight).
    var sky_color: vec3<f32>;
    if sky_ray.y < -0.05 {
        sky_color = atmosphere(sky_ray);
    } else {
        let environment = sky_environment(sky_ray);
        var celestial = vec3(0.0);
        if environment.a > 0.001 {
            celestial = sky_celestial(sky_ray);
        }
        var rainbow = vec3(0.0);
        if rainbows && frame.wetness > 0.0 {
            let cell = clamp(vec2<i32>(cloud_map_texel(sky_ray)), vec2(0), vec2(i32(ENV_WIDTH) - 1, i32(ENV_HEIGHT) - 1));
            let cloud_distance = textureLoad(cloud_aux, cell, 0).r;
            let visible_distance = mix(cloud_distance, 2600.0, smoothstep(0.90, 1.0, environment.a));
            rainbow = rainbow_color(sky_ray, visible_distance);
        }
        // Localized 3D in-cloud sheet lightning evaluated at full framerate (zero cloud-history stutter).
        let flash = storm_flash();
        var sheet_lightning = vec3(0.0);
        if flash > 0.001 {
            let strike_align = pow(clamp(dot(sky_ray, storm_flash_dir()), 0.0, 1.0), 6.0);
            sheet_lightning = vec3(0.62, 0.78, 1.15) * (flash * strike_align * (1.0 - environment.a) * 1.9);
        }
        sky_color = environment.rgb + sheet_lightning + celestial * environment.a + rainbow;
    }
    let sky_access = air_fog_sky_access();
    if fog_strength <= 0.0 || sky_access <= 0.0 {
        return vec4(sky_color, 1.0);
    }
    // Evaluate analytic open-sky haze directly along the physical horizon-aligned ray.
    let haze = raymarch_air_fog(sky_ray, frame.far, 0.5, sky_access, true);
    return vec4(sky_color * haze.transmittance + haze.scattering, 1.0);
}
