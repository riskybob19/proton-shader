// The sky stage runs without terrain shadow bindings.
fn fog_shadow(_position: vec3<f32>) -> vec3<f32> {
    return vec3(1.0);
}

fn cloud_coverage(world: vec2<f32>) -> f32 {
    return cumulus_coverage(world);
}

// Fast 4-tap cubic B-spline reconstruction over the 768x768 octahedral cloud dome.
// Open sky is rendered at 1:1 native screen resolution via cloud_screen_history in
// composite.wgsl, so sky.wgsl stays ultra-lean for translucent glass/water refraction.
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
    return max(top + bot, vec4(0.0));
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
    let above_clouds = frame.camera_position.y >= CLOUD_BASE - 8.0;
    let below_fade = select(0.0, 1.0 - smoothstep(-0.072, -0.032, ray.y), above_clouds);

    let base_atmo = atmosphere(ray);
    var bg_sky = base_atmo;
    var cloud_rgb = vec3(0.0);
    var cloud_a = 1.0;
    var extra_glow = vec3(0.0);
    if ray.y < -0.05 || (above_clouds && ray.y <= -0.072) {
        bg_sky += select(vec3(0.0), sky_celestial(ray), ray.y > -0.05);
    } else {
        let raw_env = sky_environment(ray);
        let cloud_only = max(raw_env.rgb - base_atmo * raw_env.a, vec3(0.0));
        cloud_rgb = cloud_only * (1.0 - below_fade);
        cloud_a = mix(raw_env.a, 1.0, below_fade);
        if cloud_a > 0.001 && ray.y > -0.05 {
            bg_sky += sky_celestial(ray);
        }
        if rainbows && frame.wetness > 0.0 && below_fade < 0.999 {
            let cell = clamp(vec2<i32>(cloud_map_texel(ray)), vec2(0), vec2(i32(ENV_WIDTH) - 1, i32(ENV_HEIGHT) - 1));
            let cloud_distance = textureLoad(cloud_aux, cell, 0).r;
            let visible_distance = mix(cloud_distance, 2600.0, smoothstep(0.90, 1.0, cloud_a));
            extra_glow += rainbow_color(ray, visible_distance) * (1.0 - below_fade);
        }
        // Localized 3D in-cloud sheet lightning evaluated at full framerate (zero cloud-history stutter).
        let flash = storm_flash();
        if flash > 0.001 {
            let strike_align = pow(clamp(dot(ray, storm_flash_dir()), 0.0, 1.0), 6.0);
            extra_glow += vec3(0.62, 0.78, 1.15) * (flash * strike_align * (1.0 - cloud_a) * 1.9);
        }
    }
    var sky_color = bg_sky * cloud_a + cloud_rgb + extra_glow;
    let sky_access = air_fog_sky_access();
    if fog_strength <= 0.0 || sky_access <= 0.0 {
        return vec4(sky_color, 1.0);
    }
    // Evaluate analytic open-sky haze out to render_distance (matching upstream Photon's `far`),
    // keeping ground-level haze behind foreground clouds when flying at/above cloud altitude.
    let haze = raymarch_air_fog(ray, max(frame.render_distance, 32.0), 0.5, sky_access, true);
    let cloud_above_fog = smoothstep(CLOUD_BASE - 36.0, CLOUD_BASE - 8.0, frame.camera_position.y);
    let bg_hazed = bg_sky * haze.transmittance + haze.scattering;
    let cloud_front = bg_hazed * cloud_a + cloud_rgb + extra_glow;
    let fog_front = sky_color * haze.transmittance + haze.scattering;
    return vec4(mix(fog_front, cloud_front, cloud_above_fog), 1.0);
}
