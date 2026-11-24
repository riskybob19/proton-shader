override bloom_strength: f32 = 0.14;
override cas_strength: f32 = 0.45;
override auto_exposure: bool = false;
override underwater_distortion: f32 = 0.5;
override heat_distortion: f32 = 0.65;
override sun_lens_halo: f32 = 0.45;

// Physically based long-distance atmospheric optical turbulence (Chimitt & Chan Tilt-then-Blur
// + Monin-Obukhov h^-4/3 convective boundary layer + 2-octave Kolmogorov buoyancy eddies).
// Runs post-TAA inside bloom_combine with zero extra passes/VRAM and instant early-out on
// cold biomes, rain, night, caves, sky, and foreground geometry (< 38m Overworld / < 22m Nether).
struct HeatResult {
    color: vec3<f32>,
    uv: vec2<f32>,
    turb: f32,
}

fn atmospheric_heat_distortion(uv: vec2<f32>, depth: f32, center_sample: vec4<f32>,
    size: vec2<i32>) -> HeatResult {
    if heat_distortion <= 0.001 || depth <= 0.0 || frame.eye_in != 0u || frame.blindness > 0.0 {
        return HeatResult(center_sample.rgb, uv, 0.0);
    }

    var heat_env = 0.0;
    var dry_desert = 0.0;
    let is_nether = frame.dimension == -1;
    if frame.dimension == 0 {
        let sun_heat = smoothstep(0.08, 0.52, frame.sun_direction.y)
            * (1.0 - frame.rain) * (1.0 - 0.55 * frame.wetness);
        let sky_open = smoothstep(0.25, 0.85, frame.eye_light_smooth.y);
        if sun_heat * sky_open <= 0.005 {
            return HeatResult(center_sample.rgb, uv, 0.0);
        }
        dry_desert = smoothstep(0.68, 1.0, frame.temperature)
            * (1.0 - smoothstep(0.05, 0.48, frame.downfall));
        let warm_day = smoothstep(0.45, 0.90, frame.temperature) * 0.28;
        heat_env = heat_distortion * sun_heat * sky_open * clamp(warm_day + 0.88 * dry_desert, 0.0, 1.0);
    } else if is_nether {
        heat_env = heat_distortion * 0.90;
    }
    if heat_env <= 0.008 {
        return HeatResult(center_sample.rgb, uv, 0.0);
    }

    let taa_has_dist = center_sample.a > 0.0;
    let start_dist = select(38.0, 22.0, is_nether);
    if taa_has_dist && center_sample.a <= start_dist {
        return HeatResult(center_sample.rgb, uv, 0.0);
    }

    let world_pos = view_position(uv, depth);
    let dist = select(length(world_pos), center_sample.a, taa_has_dist);
    if dist <= start_dist {
        return HeatResult(center_sample.rgb, uv, 0.0);
    }

    let ray = world_pos / max(dist, 1e-4);
    let dist_ramp = smoothstep(start_dist, start_dist + 130.0, dist);
    // Monin-Obukhov convective boundary layer C_n^2(h) ~ h^(-4/3) & grazing sightline weight.
    var cbl = 1.0;
    var grazing = 1.0;
    if is_nether {
        let h_mid = max(frame.camera_position.y + 0.5 * world_pos.y - 31.0, 3.0);
        cbl = clamp(20.0 / h_mid, 0.35, 1.35);
        grazing = 1.0 - smoothstep(0.22, 0.72, abs(ray.y));
    } else {
        let h_mid = max(frame.camera_position.y + 0.5 * world_pos.y - 60.0, 2.5);
        let h_norm = clamp(h_mid * (1.0 / 14.0), 0.35, 16.0);
        cbl = clamp(pow(h_norm, -1.333333), 0.08, 1.45);
        grazing = 1.0 - smoothstep(0.06, 0.42, abs(ray.y));
    }
    let turb = clamp(heat_env * sqrt(dist_ramp) * cbl * grazing, 0.0, 1.0);
    if turb <= 0.012 {
        return HeatResult(center_sample.rgb, uv, 0.0);
    }

    // World-ray-anchored 2-octave Kolmogorov buoyancy eddies (Taylor's frozen flow):
    // camera panning never drags heat waves like a screen-space filter.
    let az = atan2(ray.x, ray.z) * (1.0 / TAU);
    let uv1 = vec2(az * 14.0 + frame.time * 0.045, ray.y * 18.0 - frame.time * 0.22);
    let uv2 = vec2(az * 34.0 - frame.time * 0.075, ray.y * 46.0 - frame.time * 0.58);
    let n1 = textureSampleLevel(wave_noise, wave_noise_sampler, uv1, 0.0).rg * 2.0 - 1.0;
    let n2 = textureSampleLevel(wave_noise, wave_noise_sampler, uv2, 0.0).rg * 2.0 - 1.0;
    let tilt = (n1 * 0.68 + n2 * 0.32) * vec2(0.55, 1.0);

    // Subtle upward-bent inferior mirage along grazing hot desert horizons.
    let mirage_w = dry_desert * heat_env * smoothstep(72.0, 170.0, dist)
        * (1.0 - smoothstep(0.0, 0.075, abs(ray.y + 0.015)));
    let edge_fade = smoothstep(0.0, 0.04, min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)));
    let uv_shift = (tilt * (0.0021 * turb) + vec2(0.0, -0.0011 * mirage_w)) * edge_fade;
    var heat_uv = clamp(uv + uv_shift, vec2(0.001), vec2(0.999));

    // Depth-guarded foreground silhouette protection: never pull closer blocks into distant shimmer.
    var shifted = textureSampleLevel(taa_history, linear_sampler, heat_uv, 0.0);
    var hit_fg = shifted.a > 0.0 && shifted.a < dist * 0.86;
    if !taa_has_dist {
        let tap_px = clamp(vec2<i32>(heat_uv * vec2<f32>(size)), vec2(0), size - 1);
        hit_fg = textureLoad(scene_depth, tap_px, 0) > depth * 1.08;
    }
    if hit_fg {
        heat_uv = uv;
        shifted = center_sample;
    }

    // Chimitt & Chan Tilt-then-Blur: 2-tap depth-guarded higher-order PSF softening + micro-scintillation.
    var out_color = shifted.rgb;
    if turb > 0.12 {
        let blur_step = vec2(0.45, 1.25) * (turb / vec2<f32>(size));
        let uv_a = clamp(heat_uv + blur_step, vec2(0.001), vec2(0.999));
        let uv_b = clamp(heat_uv - blur_step, vec2(0.001), vec2(0.999));
        let s_a = textureSampleLevel(taa_history, linear_sampler, uv_a, 0.0);
        let s_b = textureSampleLevel(taa_history, linear_sampler, uv_b, 0.0);
        let ok_a = !taa_has_dist || s_a.a <= 0.0 || s_a.a >= dist * 0.84;
        let ok_b = !taa_has_dist || s_b.a <= 0.0 || s_b.a >= dist * 0.84;
        let w_blur = smoothstep(0.12, 0.85, turb) * 0.24;
        var sum = out_color;
        var w_tot = 1.0;
        if ok_a {
            sum += s_a.rgb * w_blur;
            w_tot += w_blur;
        }
        if ok_b {
            sum += s_b.rgb * w_blur;
            w_tot += w_blur;
        }
        out_color = sum / w_tot;
    }
    let scint = 1.0 + 0.032 * turb * (n2.y * 0.7 + n1.x * 0.3);
    return HeatResult(max(out_color * scint, vec3(0.0)), heat_uv, turb);
}

// 5-tap AMD FidelityFX Contrast-Adaptive Sharpening (zero extra render passes)
// restores crisp pixel-art texture detail after TAA without ringing.
fn cas_sharpen(pixel: vec2<i32>, size: vec2<i32>, center: vec3<f32>) -> vec3<f32> {
    if cas_strength <= 0.001 {
        return center;
    }
    let n = textureLoad(taa_history, clamp(pixel + vec2(0, -1), vec2(0), size - 1), 0).rgb;
    let s = textureLoad(taa_history, clamp(pixel + vec2(0, 1), vec2(0), size - 1), 0).rgb;
    let w = textureLoad(taa_history, clamp(pixel + vec2(-1, 0), vec2(0), size - 1), 0).rgb;
    let e = textureLoad(taa_history, clamp(pixel + vec2(1, 0), vec2(0), size - 1), 0).rgb;

    // Map HDR luminance into perceptual [0, 1] before computing CAS contrast weight
    // so sharpening stays active on sunlit snow, bright stone, and clouds (HDR > 2.0).
    // Since x / (1 + x) is strictly monotonic for x >= 0, compute min/max first (2 divs vs 5).
    let l_c = max(dot(center, frame.luminance), 0.0);
    let l_n = max(dot(n, frame.luminance), 0.0);
    let l_s = max(dot(s, frame.luminance), 0.0);
    let l_w = max(dot(w, frame.luminance), 0.0);
    let l_e = max(dot(e, frame.luminance), 0.0);
    let raw_min = min(l_c, min(min(l_n, l_s), min(l_w, l_e)));
    let raw_max = max(l_c, max(max(l_n, l_s), max(l_w, l_e)));
    let l_min = raw_min / (1.0 + raw_min);
    let l_max = raw_max / (1.0 + raw_max);

    // Adaptive weight peaks on medium-contrast detail and backs off on high-contrast edges.
    let amp = clamp(min(l_min, 1.0 - l_max) / max(l_max, 1e-4), 0.0, 1.0);
    let weight = -sqrt(amp) * mix(0.0, 0.14, clamp(cas_strength, 0.0, 1.0));
    let sharpened = (center + weight * (n + s + w + e)) / max(1.0 + 4.0 * weight, 1e-4);
    let low = min(center, min(min(n, s), min(w, e)));
    let high = max(center, max(max(n, s), max(w, e)));
    return clamp(sharpened, max(low, vec3(0.0)), high);
}

// Subtle RDR2 / Horizon Forbidden West physical sun-corona diffraction halo:
// auto-occluded by both terrain/leaves (scene_depth) and volumetric clouds (taa_history brightness).
fn physical_sun_halo(uv: vec2<f32>, aspect: f32) -> vec3<f32> {
    if sun_lens_halo <= 0.001 || frame.dimension != 0 || frame.eye_in != 0u || frame.sun_direction.y < -0.02 {
        return vec3(0.0);
    }
    let clip = frame.projection * frame.view * vec4(frame.sun_direction, 0.0);
    if clip.w <= 0.0 {
        return vec3(0.0);
    }
    let ndc = clip.xy / clip.w;
    let sun_uv = 0.5 + ndc * vec2(0.5, -0.5);
    if any(sun_uv < vec2(-0.25)) || any(sun_uv > vec2(1.25)) {
        return vec3(0.0);
    }
    // 5-tap visibility probe around the solar disk (checks terrain depth, exact 1080p
    // cloud transmittance, and high solar-disk luminance so opaque clouds never trigger corona).
    let sc_size = vec2<i32>(textureDimensions(cloud_screen_history));
    let tap_r = vec2(0.007, 0.007 * aspect);
    var sun_vis = 0.0;
    var sun_col = vec3(0.0);
    for (var i = 0u; i < 5u; i += 1u) {
        let off = select(
            select(
                select(vec2(0.0, -tap_r.y), vec2(0.0, tap_r.y), i == 3u),
                select(vec2(-tap_r.x, 0.0), vec2(tap_r.x, 0.0), i == 1u),
                i < 3u,
            ),
            vec2(0.0),
            i == 0u,
        );
        let p_uv = clamp(sun_uv + off, vec2(0.002), vec2(0.998));
        let d = depth_at(p_uv);
        if d <= 0.0 {
            let sc_px = clamp(vec2<i32>(p_uv * vec2<f32>(sc_size)), vec2(0), sc_size - 1);
            let sc_raw = unpack_u32_vec4(textureLoad(cloud_screen_history, sc_px, 0));
            let cloud_vis = select(1.0, clamp((sc_raw.a - 0.02) * (1.0 / 0.98), 0.0, 1.0), sc_raw.a >= 0.01);
            let c = max(textureSampleLevel(taa_history, linear_sampler, p_uv, 0.0).rgb, vec3(0.0));
            let l = dot(c, frame.luminance);
            let w = smoothstep(1.8, 5.5, l) * cloud_vis;
            sun_vis += w;
            sun_col += c * w;
        }
    }
    sun_vis *= 0.20;
    if sun_vis <= 0.005 {
        return vec3(0.0);
    }
    let sun_tint = normalize(sun_col / max(sun_vis * 5.0, 1e-4) + vec3(0.05)) * vec3(1.12, 0.98, 0.82);
    let delta = vec2((uv.x - sun_uv.x) * aspect, uv.y - sun_uv.y);
    let dist = length(delta);
    let angle = atan2(delta.y, delta.x);
    // Soft atmospheric corona + subtle 6-blade aperture diffraction star.
    let corona = exp(-dist * 7.5) * 0.14 + exp(-dist * 24.0) * 0.22;
    let rays = pow(abs(cos(angle * 3.0 + 0.35)), 10.0) * exp(-dist * 11.5) * 0.11;
    // Faint warm-cool chromatic ghost along the opposite optical axis.
    let ghost_uv = vec2(0.5) - (sun_uv - vec2(0.5)) * 0.55;
    let g_delta = vec2((uv.x - ghost_uv.x) * aspect, uv.y - ghost_uv.y);
    let g_dist = length(g_delta);
    let ghost = exp(-pow((g_dist - 0.085) * 26.0, 2.0)) * 0.032 * vec3(0.85, 0.96, 1.15);
    let edge_damp = 1.0 - smoothstep(0.75, 1.20, max(abs(ndc.x), abs(ndc.y)));
    let weather_damp = (1.0 - 0.85 * frame.rain) * smoothstep(-0.02, 0.14, frame.sun_direction.y);
    return (sun_tint * (corona + rays) + ghost) * (sun_vis * sun_lens_halo * edge_damp * weather_damp);
}

fn post(input: PostInput) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(taa_history));
    let pixel = clamp(vec2<i32>(input.pixel), vec2(0), size - 1);
    let submerged = frame.dimension == 0 && frame.eye_in == 1u;

    var sample_uv = input.uv;
    var resolved: vec3<f32>;
    if submerged && underwater_distortion > 0.001 {
        // Subtle post-TAA underwater optical wave refraction (safe for TAA motion vectors)
        // + gentle peripheral diver-mask RGB Cauchy chromatic dispersion.
        let dist_amt = clamp(underwater_distortion, 0.0, 1.0);
        let t = frame.time * 1.15;
        let w1 = vec2(
            sin(input.uv.y * 14.0 + t) + 0.45 * cos(input.uv.x * 11.0 - t * 0.78),
            cos(input.uv.x * 13.0 + t * 0.92) + 0.45 * sin(input.uv.y * 10.5 + t * 0.71),
        );
        let edge_fade = smoothstep(0.0, 0.06, min(min(input.uv.x, 1.0 - input.uv.x), min(input.uv.y, 1.0 - input.uv.y)));
        let uv_shift = w1 * (0.00115 * dist_amt * edge_fade);
        sample_uv = clamp(input.uv + uv_shift, vec2(0.001), vec2(0.999));

        let radial = sample_uv - vec2(0.5);
        let r2 = dot(radial, radial);
        let ca_offset = radial * (0.0024 * dist_amt * r2);
        let uv_r = clamp(sample_uv + ca_offset, vec2(0.001), vec2(0.999));
        let uv_b = clamp(sample_uv - ca_offset, vec2(0.001), vec2(0.999));
        let col_r = textureSampleLevel(taa_history, linear_sampler, uv_r, 0.0).r;
        let col_g = textureSampleLevel(taa_history, linear_sampler, sample_uv, 0.0).g;
        let col_b = textureSampleLevel(taa_history, linear_sampler, uv_b, 0.0).b;
        let refracted = vec3(col_r, col_g, col_b);
        // Relax CAS sharpening underwater so distant geometry softens naturally with depth.
        let sharp = cas_sharpen(pixel, size, textureLoad(taa_history, pixel, 0).rgb);
        resolved = mix(refracted, sharp, 0.25 * (1.0 - dist_amt * 0.65));
    } else {
        let center_sample = textureLoad(taa_history, pixel, 0);
        let heat = atmospheric_heat_distortion(input.uv, input.depth, center_sample, size);
        if heat.turb > 0.012 {
            sample_uv = heat.uv;
            if heat.turb < 0.22 {
                let sharp = cas_sharpen(pixel, size, center_sample.rgb);
                resolved = mix(sharp, heat.color, smoothstep(0.012, 0.22, heat.turb));
            } else {
                resolved = heat.color;
            }
        } else if input.depth > 0.0 {
            resolved = cas_sharpen(pixel, size, center_sample.rgb);
        } else {
            resolved = center_sample.rgb;
        }
    }

    var auto_exp = 1.0;
    if auto_exposure {
        let stored_exp = textureLoad(sky_lighting, EXPOSURE_CELL, 0).x;
        auto_exp = select(1.0, clamp(stored_exp, 0.60, 1.85), stored_exp > 0.2);
    }
    if bloom_strength <= 0.0 {
        return vec4(resolved * auto_exp, 1.0);
    }

    let near_glow = textureSampleLevel(bloom_near, linear_sampler, sample_uv, 0.0).rgb;
    let wide_glow = textureSampleLevel(bloom_wide, linear_sampler, sample_uv, 0.0).rgb;
    let far_glow = textureSampleLevel(bloom_far, linear_sampler, sample_uv, 0.0).rgb;
    let halo_glow = 0.45 * wide_glow + 0.55 * far_glow;
    // Upstream Photon Bloomy Fog above water + Jaffe-McGlamery small-angle forward-scatter
    // optical PSF underwater (softens distant submerged blocks & wraps sea lanterns/shafts in halos).
    if frame.dimension == 0 {
        if frame.eye_in == 0u && input.depth > 0.0 {
            // Depth-guarded nearest half-res fog transmittance lookup (prevents sky/cloud
            // alpha from bleeding across silhouette edges and creating dark-scene halos).
            let fog_size = vec2<i32>(textureDimensions(fog_transmittance));
            let fog_cell = clamp(pixel / 2, vec2(0), fog_size - 1);
            let fog_uv = (vec2<f32>(fog_cell) + 0.5) / vec2<f32>(fog_size);
            if depth_at(fog_uv) > 0.0 {
                let fog_sample = textureLoad(fog_transmittance, fog_cell, 0);
                let fog_opacity = clamp(1.0 - dot(fog_sample.rgb, vec3(0.333333)), 0.0, 1.0);
                let day_mask = smoothstep(-0.04, 0.18, frame.sun_direction.y);
                let luma_guard = smoothstep(0.015, 0.14, dot(resolved, frame.luminance));
                let bloomy_w = fog_opacity * fog_opacity * fog_opacity * 0.11 * day_mask * luma_guard;
                resolved = mix(resolved, halo_glow, bloomy_w);
            }
        } else if submerged {
            let fog_trans = textureSampleLevel(fog_transmittance, linear_sampler, sample_uv, 0.0).rgb;
            let uw_opacity = clamp(1.0 - dot(fog_trans, vec3(0.333333)), 0.0, 1.0);
            let psf_strength = mix(0.10, 0.26, clamp(underwater_distortion, 0.0, 1.0));
            resolved = mix(resolved, halo_glow, uw_opacity * uw_opacity * psf_strength);
        }
    }
    let uw_bloom_boost = select(1.0, 1.35, submerged);
    let multi_bloom = 0.42 * near_glow + 0.34 * wide_glow + 0.24 * far_glow;
    let lens_halo = physical_sun_halo(sample_uv, frame.resolution.x / max(frame.resolution.y, 1.0));
    return vec4((resolved + bloom_strength * uw_bloom_boost * multi_bloom + lens_halo) * auto_exp, 1.0);
}
