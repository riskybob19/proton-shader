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
    // Use direction-only unprojection (z=0.0005, w=0) so view-bobbing eye translation never swings the sky.
    if frame.dimension == 1 && input.depth <= 0.0 {
        let ndc = vec2(input.uv.x * 2.0 - 1.0, 1.0 - input.uv.y * 2.0);
        let clip_far = frame.projection_inverse * vec4(ndc, 0.0005, 1.0);
        let view_dir = clip_far.xyz / max(abs(clip_far.w), 1e-6);
        let direction = normalize((frame.view_inverse * vec4(view_dir, 0.0)).xyz);
        return vec4(end_sky(direction, input.color.rgb), 1.0);
    }
    if input.depth <= 0.0 && !use_underwater_fog() {
        return input.color;
    }
    let world = fog_position(input.uv, input.depth);
    let ray_length = length(world);
    let terrain = terrain_effects(input, world, ray_length);
    if !use_volume_composite() {
        return terrain;
    }
    let ray = world / max(ray_length, 1e-6);

    let size = vec2<i32>(textureDimensions(fog_transmittance));
    let distance = fog_distance(ray_length, input.depth);
    let sky = input.depth <= 0.0;
    let guide = select(distance, -1.0, sky);
    let tolerance = max(0.3, 0.025 * distance);

    // Single-check hardware bilinear fast-path on continuous terrain/sky (2 fetches instead
    // of 9 manual loads); falls back to 4-tap bilateral resolve only on geometric silhouettes.
    var rainbow = vec3(0.0);
    if !sky && frame.eye_in == 0u && distance > 250.0 {
        rainbow = rainbow_color(ray, distance);
    }
    var color = vec3(0.0);
    var scattering = vec3(0.0);
    var total = 0.0;
    var best_scattering = vec3(0.0);
    let bt = textureSampleLevel(fog_transmittance, linear_sampler, input.uv, 0.0);
    let smooth_depth = sky || depth_grad < max(input.depth * 0.06, 1e-5);
    let b_ok = smooth_depth && (bt.a < 0.0) == sky && (sky || abs(guide - bt.a) < 0.45 * tolerance);
    if b_ok {
        let bs = textureSampleLevel(fog_scattering, linear_sampler, input.uv, 0.0);
        color = terrain.rgb * bt.rgb + bs.rgb + rainbow * bs.a;
        best_scattering = bs.rgb;
        total = -1.0;
    }
    if total == 0.0 {
        let position = input.uv * vec2<f32>(size) - 0.5;
        let corner = vec2<i32>(floor(position));
        let blend = fract(position);
        var transmittance = vec3(0.0);
        var cloud_transmission = 0.0;
        var best_diff = 1e30;
        var best_transmittance = vec3(1.0);
        var best_cloud = 1.0;
        for (var tap = 0u; tap < 4u; tap += 1u) {
            let offset = vec2<i32>(i32(tap & 1u), i32(tap >> 1u));
            let cell = clamp(corner + offset, vec2(0), size - 1);
            let sample_value = textureLoad(fog_transmittance, cell, 0);
            let sample_sky = sample_value.a < 0.0;
            if sample_sky == sky {
                let diff = abs(guide - sample_value.a);
                let axis_weight = select(1.0 - blend, blend, offset == vec2(1));
                let depth_weight = 1.0 - smoothstep(0.0, tolerance, diff);
                let weight = axis_weight.x * axis_weight.y * depth_weight;
                let is_best = diff < best_diff;
                if weight <= 0.0 && !is_best {
                    continue;
                }
                let volume = textureLoad(fog_scattering, cell, 0);
                if is_best {
                    best_diff = diff;
                    best_transmittance = sample_value.rgb;
                    best_scattering = volume.rgb;
                    best_cloud = volume.a;
                }
                if weight <= 0.0 {
                    continue;
                }
                transmittance += weight * sample_value.rgb;
                scattering += weight * volume.rgb;
                cloud_transmission += weight * volume.a;
                total += weight;
            }
        }
        if total > 0.001 {
            color = (terrain.rgb * transmittance + scattering + rainbow * cloud_transmission) / total;
        } else if best_diff < 1e29 {
            color = terrain.rgb * best_transmittance + best_scattering + rainbow * best_cloud;
        } else if use_underwater_fog() {
            let fog = raymarch_underwater(ray, distance, 0.5);
            color = terrain.rgb * fog.transmittance + fog.scattering;
        } else if fog_strength > 0.0 && air_fog_sky_access() > 0.0 {
            let fog = raymarch_air_fog(ray, distance, 0.5, air_fog_sky_access(), true);
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
        // Physically based render-distance border dissolve: distant terrain/oceans
        // meet the horizon-aligned sky dome seamlessly with identical atmospheric haze.
        let sky_access = air_fog_sky_access();
        let end = min(max(frame.fog_render.y, 16.0), max(frame.render_distance, 32.0));
        let cyl_ratio = clamp(length(world.xz) / end, 0.0, 1.0);
        // Strictly 0.0 inside 84% of render distance so the sky/horizon never bleeds
        // through foreground or midground terrain, then dissolves smoothly at the rim.
        let t = smoothstep(0.84, 1.0, cyl_ratio);
        let border = t * t * (3.0 - 2.0 * t)
            * smoothstep(0.0, 0.14, 1.0 - max(-ray.y, 0.0));
        if border > 0.002 && sky_access > 0.001 {
            let sky_ray = sky_dome_ray(ray);
            let env = sample_environment(sky_ray);
            let cel = select(vec3(0.0), sky_celestial(sky_ray), env.a > 0.001);
            var sky_backdrop = env.rgb + env.a * cel;
            if fog_strength > 0.0 {
                let sky_haze = raymarch_air_fog(sky_ray, frame.far, 0.5, sky_access, true);
                sky_backdrop = sky_backdrop * sky_haze.transmittance + sky_haze.scattering;
            }
            let cave = (1.0 - smoothstep(49.0, 63.0, frame.camera_position.y))
                * (1.0 - smoothstep(0.03, 0.30, frame.eye_light_smooth.y));
            color = mix(color, sky_backdrop * (1.0 - cave), border * sky_access);
        }
    }
    return vec4(color, 1.0);
}
