override soft_shadows: bool = false;

fn shadow_disc(index: u32, count: u32) -> vec2<f32> {
    let angle = 2.3999632 * f32(index) + 0.43;
    return vec2(cos(angle), sin(angle)) * sqrt((f32(index) + 0.5) / f32(count));
}

fn shadow_blocker_gap(pixel: vec2<i32>, depth: f32, ray_receiver: RayShadowReceiver) -> f32 {
    if frame.has_shadow != 0u {
        return depth - pack_shadow_depth(pixel, false);
    }
    let blocker = ray_shadow_sample(pixel).x;
    let gap = select(ray_shadow_receiver_at(ray_receiver, pixel, true) - blocker, 0.0, blocker >= RAY_SHADOW_DEPTH);
    return gap * (0.5 / RAY_SHADOW_DEPTH);
}

fn shadow_filter_radius(uv: vec2<f32>, depth: f32, cloud: f32, ray_receiver: RayShadowReceiver) -> f32 {
    let texel = 1.0 / f32(pack_shadow_size().x);
    if !soft_shadows {
        return 1.35 * texel;
    }

    let size = pack_shadow_size();
    var separation = 0.0;
    var blockers = 0.0;
    for (var tap = 0u; tap < 9u; tap += 1u) {
        let offset = select(shadow_disc(tap, 9u) * (8.0 * texel), vec2(0.0), tap == 0u);
        if frame.has_shadow == 0u {
            // Weight the blocker search in texel space too, so a moving
            // edge cannot abruptly change the whole filter radius.
            let cell = (uv + offset) * vec2<f32>(size) - 0.5;
            for (var i = 0u; i < 4u; i += 1u) {
                let filter_tap = shadow_bilinear_tap(cell, i);
                let gap = shadow_blocker_gap(filter_tap.pixel, depth, ray_receiver);
                let weight = filter_tap.weight * smoothstep(0.00015, 0.00065, gap);
                separation += max(gap, 0.0) * weight;
                blockers += weight;
            }
            continue;
        }
        let pixel = clamp(vec2<i32>((uv + offset) * vec2<f32>(size)), vec2(0), size - 1);
        let gap = shadow_blocker_gap(pixel, depth, ray_receiver);
        if gap > 0.00015 {
            separation += gap;
            blockers += 1.0;
        }
    }

    let depth_per_block = pack_shadow_depth_per_block();
    let distance = separation / max(select(max(blockers, 1.0), blockers, frame.has_shadow != 0u) * depth_per_block, 1e-6);
    let penumbra = distance * mix(0.095, 0.035, cloud) / max(pack_shadow_texel(), 0.01);
    return clamp(1.35 + penumbra, 1.35, 10.0) * texel;
}

struct TerrainShadow {
    visibility: vec3<f32>,
    thickness: f32,
    fade: f32,
}

fn terrain_blocker_depth(uv: vec2<f32>, depth: f32, ray_receiver: RayShadowReceiver, dist_sq: f32) -> f32 {
    let size = pack_shadow_size();
    let depth_per_block = max(pack_shadow_depth_per_block(), 1e-6);
    if dist_sq > 900.0 {
        let pixel = clamp(vec2<i32>(uv * vec2<f32>(size)), vec2(0), size - 1);
        let gap = max(shadow_blocker_gap(pixel, depth, ray_receiver), 0.0);
        return min(gap / depth_per_block, 5.0);
    }
    let texel = uv * vec2<f32>(size) - 0.5;
    var separation = 0.0;
    // Bilinearly filter blocker depth at the biased shadow position so
    // shadow-map texel boundaries never form diagonal stripes on leaves or ice.
    for (var tap = 0u; tap < 4u; tap += 1u) {
        let filter_tap = shadow_bilinear_tap(texel, tap);
        let pixel = clamp(filter_tap.pixel, vec2(0), size - 1);
        let gap = max(shadow_blocker_gap(pixel, depth, ray_receiver), 0.0);
        separation += min(gap / depth_per_block, 5.0) * filter_tap.weight;
    }
    return separation;
}

struct ShadowFilterResult {
    visibility: vec3<f32>,
    blocker_depth: f32,
    has_depth: bool,
}

fn terrain_shadow_filter(center: vec2<f32>, depth: f32, cloud: f32, ray_receiver: RayShadowReceiver, pixel_cell: vec2<f32>, dist_sq: f32) -> ShadowFilterResult {
    let native = frame.has_shadow != 0u;
    if !native && !soft_shadows {
        return ShadowFilterResult(ray_shadow_antialiased(center, ray_receiver), 0.0, false);
    }
    if native {
        let size = pack_shadow_size();
        let fsize = vec2<f32>(size);
        let texel = 1.0 / fsize.x;
        // Single IGN + golden-ratio temporal rotation shared across blocker search & PCF.
        let ign = fract(52.9829189 * fract(dot(pixel_cell, vec2(0.06711056, 0.00583715))));
        let base_angle = (ign + f32(frame.frame_index & 7u) * 0.61803399) * TAU;
        let ca = cos(base_angle);
        let sa = sin(base_angle);
        let rot_unit = mat2x2<f32>(vec2(ca, sa), vec2(-sa, ca));

        // Beyond 84 blocks, 1 shadow texel is < 0.9 screen pixel and 8-frame TAA integrates
        // 16 effective taps from a 2-tap temporally rotated Vogel disk.
        if dist_sq > 7056.0 {
            let rot = rot_unit * (1.45 * texel);
            let uv0 = clamp(center + rot * vec2(0.50, 0.0), vec2(0.0), vec2(1.0));
            let uv1 = clamp(center + rot * vec2(-0.64, 0.58), vec2(0.0), vec2(1.0));
            let solid = 0.5 * (pack_shadow_compare(uv0, depth, false)
                + pack_shadow_compare(uv1, depth, false));
            return ShadowFilterResult(shadow_transmission(center, depth, solid), 0.0, false);
        }

        // Beyond 40 blocks, 1 shadow texel projects to < 1.8 screen pixels so variable PCSS
        // penumbra is sub-pixel: use 3-tap temporally rotated Vogel PCF.
        if !soft_shadows || dist_sq > 1600.0 {
            let rot = rot_unit * (1.5525 * texel);
            let uv0 = clamp(center + rot * vec2(0.4082483, 0.0), vec2(0.0), vec2(1.0));
            let uv1 = clamp(center + rot * vec2(-0.5214019, 0.4776401), vec2(0.0), vec2(1.0));
            let uv2 = clamp(center + rot * vec2(0.0797876, -0.9093593), vec2(0.0), vec2(1.0));
            let solid = (pack_shadow_compare(uv0, depth, false)
                + pack_shadow_compare(uv1, depth, false)
                + pack_shadow_compare(uv2, depth, false)) * (1.0 / 3.0);
            return ShadowFilterResult(shadow_transmission(center, depth, solid), 0.0, false);
        }

        // Near-field (0..36m) temporally-rotated 4-tap PCSS blocker search (center + 3 equilateral
        // taps, zero trig/sqrt per tap) + open-sun & deep-shadow early-outs.
        let search = rot_unit * (6.8 * texel);
        var separation = 0.0;
        var blockers = 0.0;
        let g0 = depth - pack_shadow_depth(clamp(vec2<i32>(center * fsize), vec2(0), size - 1), false);
        if g0 > 0.00015 { separation += g0; blockers += 1.0; }
        let g1 = depth - pack_shadow_depth(clamp(vec2<i32>((center + search * vec2(0.82, 0.0)) * fsize), vec2(0), size - 1), false);
        if g1 > 0.00015 { separation += g1; blockers += 1.0; }
        let g2 = depth - pack_shadow_depth(clamp(vec2<i32>((center + search * vec2(-0.41, 0.71)) * fsize), vec2(0), size - 1), false);
        if g2 > 0.00015 { separation += g2; blockers += 1.0; }
        let g3 = depth - pack_shadow_depth(clamp(vec2<i32>((center + search * vec2(-0.41, -0.71)) * fsize), vec2(0), size - 1), false);
        if g3 > 0.00015 { separation += g3; blockers += 1.0; }

        if blockers == 0.0 {
            // Fast 2-tap rotated PCF confirms thin 1-2 texel edges (fences, leaf gaps) without hard ringing.
            let uv1 = clamp(center + rot_unit * vec2(1.35 * texel, 0.0), vec2(0.0), vec2(1.0));
            let solid = 0.5 * (pack_shadow_compare(center, depth, false)
                + pack_shadow_compare(uv1, depth, false));
            return ShadowFilterResult(shadow_transmission(center, depth, solid), 0.0, false);
        }

        let depth_per_block = pack_shadow_depth_per_block();
        let distance = separation / max(blockers * depth_per_block, 1e-6);
        let local_depth = min(max(g0, 0.0) / max(depth_per_block, 1e-6), 5.0);
        let use_local_depth = dist_sq > 900.0 && g0 > 0.00015;
        let min_gap = min(min(g0, g1), min(g2, g3));
        if blockers >= 4.0 && min_gap > 0.002 {
            // Completely inside solid shadow across the full search disk: skip PCF + translucent fetch.
            return ShadowFilterResult(vec3(0.0), local_depth, use_local_depth);
        }

        // Smoothly taper PCSS penumbra into the 1.35-texel PCF baseline across 28..40m (zero pop at 40m).
        let pcss_fade = 1.0 - smoothstep(784.0, 1600.0, dist_sq);
        let penumbra = distance * mix(0.095, 0.035, cloud) * pcss_fade / max(pack_shadow_texel(), 0.01);
        let radius = clamp(1.35 + penumbra, 1.35, 10.0) * texel;
        let rot = rot_unit * radius;

        // Unrolled temporally rotated 4-tap Vogel-disk PCF (32 effective samples across 8 TAA frames).
        let uv0 = clamp(center + rot * vec2(0.353553, 0.0), vec2(0.0), vec2(1.0));
        let uv1 = clamp(center + rot * vec2(-0.451547, 0.413648), vec2(0.0), vec2(1.0));
        let uv2 = clamp(center + rot * vec2(0.069116, -0.787542), vec2(0.0), vec2(1.0));
        let uv3 = clamp(center + rot * vec2(0.569148, 0.742340), vec2(0.0), vec2(1.0));
        let solid = (pack_shadow_compare(uv0, depth, false)
            + pack_shadow_compare(uv1, depth, false)
            + pack_shadow_compare(uv2, depth, false)
            + pack_shadow_compare(uv3, depth, false)) * 0.25;
        return ShadowFilterResult(shadow_transmission(center, depth, solid), local_depth, use_local_depth);
    }

    let radius = shadow_filter_radius(center, depth, cloud, ray_receiver);
    let softness = smoothstep(1.35, 3.5, radius * f32(pack_shadow_size().x));
    if softness <= 0.0 {
        return ShadowFilterResult(ray_shadow_antialiased(center, ray_receiver), 0.0, false);
    }
    let taps = select(4u, 8u, soft_shadows);
    var transmitted = vec3(0.0);
    for (var tap = 0u; tap < taps; tap += 1u) {
        var offset: vec2<f32>;
        if soft_shadows {
            offset = shadow_disc(tap, taps);
        } else {
            let t = tap & 3u;
            offset = select(select(select(vec2(-0.85, -0.40), vec2(0.40, -0.85), t == 1u),
                vec2(0.85, 0.40), t == 2u), vec2(-0.40, 0.85), t == 3u);
        }
        let uv = clamp(center + offset * radius, vec2(0.0), vec2(1.0));
        transmitted += ray_shadow_filtered(uv, ray_receiver, true);
    }
    transmitted /= f32(taps);
    if softness < 1.0 {
        return ShadowFilterResult(mix(ray_shadow_antialiased(center, ray_receiver), transmitted, softness), 0.0, false);
    }
    return ShadowFilterResult(transmitted, 0.0, false);
}

fn terrain_shadow_biased(position: vec3<f32>, normal: vec3<f32>, skylight: f32, subsurface: f32,
    cloud: f32, pixel_cell: vec2<f32>, max_bias: f32) -> TerrainShadow {
    if !pack_shadow_active() {
        return TerrainShadow(vec3(1.0), 0.0, 1.0);
    }
    // In Minecraft, any block beneath even a 1-block-thick leaf canopy has skylight <= 14/15;
    // keep 14/15 shadowed so trees never pop brighter when crossing outside the shadow map.
    let distant = vec3(smoothstep(14.1 / 15.0, 14.9 / 15.0, skylight));
    let n_dot_l = dot(normal, frame.light_direction);
    let grazing = 1.0 - max(n_dot_l, 0.0);
    let dist_sq = dot(position, position);
    // Unified voxel normal-offset + minimal slope-scaled depth bias:
    // keeps contact anchored near the camera without peter-panning while eliminating acne out to 144m.
    let dist_scale = clamp(0.08 + 0.0068 * sqrt(dist_sq), 0.08, 1.0);
    let classic_bias = min(max(pack_shadow_texel() * 1.10, 0.14) * dist_scale * (0.75 + 1.15 * grazing), max_bias);
    let normal_bias = select(0.0, select(0.025 * grazing, classic_bias,
        frame.has_shadow != 0u), n_dot_l > 0.0);
    let p = pack_shadow_position(position + normal * normal_bias);
    if any(p.xy < vec2(0.0)) || any(p.xy > vec2(1.0)) || p.z <= 0.0 || p.z >= 1.0 {
        return TerrainShadow(distant, 0.0, 1.0);
    }
    var fade = pack_shadow_edge_fade(position, p);
    var ray_receiver: RayShadowReceiver;
    if frame.has_shadow == 0u {
        ray_receiver = ray_shadow_surface(p, normal);
        fade = max(fade, ray_receiver.grazing_fade);
    }
    if fade >= 1.0 {
        return TerrainShadow(distant, 0.0, 1.0);
    }
    let depth = p.z - (0.00005 + 0.00012 * grazing);
    let filtered = terrain_shadow_filter(p.xy, depth, cloud, ray_receiver, pixel_cell, dist_sq);
    let transmitted = filtered.visibility;

    var thickness = 0.0;
    if subsurface > 0.0 && min(transmitted.r, min(transmitted.g, transmitted.b)) < 0.99 {
        var raw_depth = filtered.blocker_depth;
        if !filtered.has_depth {
            raw_depth = terrain_blocker_depth(p.xy, depth, ray_receiver, dist_sq);
        }
        let shadow_weight = 1.0 - dot(transmitted, vec3(0.333333));
        thickness = raw_depth * shadow_weight;
    }
    let near = transmitted * smoothstep(0.0, 2.0 / 15.0, skylight);
    return TerrainShadow(mix(near, distant, fade), thickness, fade);
}

fn terrain_shadow(position: vec3<f32>, normal: vec3<f32>, skylight: f32, subsurface: f32, cloud: f32, pixel_cell: vec2<f32>) -> TerrainShadow {
    return terrain_shadow_biased(position, normal, skylight, subsurface, cloud, pixel_cell, 1.0);
}

// Unified leaf shadow: keep normal bias tight (<= 0.13 blocks) so medium-distance
// porous leaf cubes continue self-shadowing inside tree crowns instead of jumping
// to 100% sunlit across 50..120 blocks.
fn leaf_shadow(position: vec3<f32>, geometric_normal: vec3<f32>, skylight: f32,
    subsurface: f32, cloud: f32, pixel_cell: vec2<f32>) -> TerrainShadow {
    let light = frame.light_direction;
    let facing = dot(geometric_normal, light);
    let grazing = 1.0 - smoothstep(0.035, 0.22, abs(facing));
    let near_only = 1.0 - smoothstep(576.0, 2304.0, dot(position, position));
    let offset = select(vec3(0.0), light * (grazing * near_only * max(pack_shadow_texel() * 0.45, 0.05)), facing > 0.0);
    return terrain_shadow_biased(position + offset, geometric_normal, skylight, subsurface, cloud, pixel_cell, 0.13);
}

fn terrain_skylight(normal: vec3<f32>, sky_level: f32, ao: f32, subsurface: f32, leaves: bool) -> vec3<f32> {
    let falloff = sky_level * sky_level;
    let up = photon_ambient_irradiance();
    var irradiance = up;
    if directional_skylight && falloff > 0.0 {
        let plane = normal.xz / (abs(normal.x) + abs(normal.y) + abs(normal.z));
        let oct = select(plane, octahedral_fold(plane), normal.y < 0.0) * 0.5 + 0.5;
        let uv = skylight_atlas_uv(oct, vec2<f32>(textureDimensions(cloud_aux)));
        let directional = textureSampleLevel(cloud_aux, scattering_atlas_sampler, uv, 0.0).rgb;
        irradiance = mix(up, directional * PHOTON_SKY_IRRADIANCE_SCALE, falloff);
    }
    let transmitted_sky = select(0.5 * up * ao, 0.55 * up + 0.20 * irradiance, leaves);
    let scattered = mix(irradiance, transmitted_sky, subsurface);
    return scattered * (ao * falloff);
}

fn surface_blocklight(position: vec3<f32>, light_levels: vec2<f32>, ao: f32, directional: f32) -> vec3<f32> {
    let block = light_levels.x;
    if block <= 0.0 {
        return vec3(0.0);
    }
    let b2 = block * block;
    let b3 = b2 * block;
    let b4 = b2 * b2;
    let b8 = b4 * b4;
    let rt_restir_active = blocklight_mode == 1u && frame.has_ray_tracing != 0u && frame.dimension == 0;
    // Physical inverse-square decay curve without a high low-end linear shelf
    // (eliminates Minecraft's jagged level-1..2 Manhattan diamond contours on terrain).
    let outer_fade = select(smoothstep(0.04, 0.26, block), smoothstep(0.28, 0.64, block), rt_restir_active);
    var falloff = (b8 + 0.36 * b4 + 0.22 * b3 + 0.04 * b2) * outer_fade;
    falloff *= mix(ao * ao, 1.0, clamp(falloff, 0.0, 1.0));
    if frame.darkness > 0.0 {
        let pulse = ao * photon_dampen(abs(cos(2.0 * frame.time))) * 0.67 + 0.2;
        falloff *= mix(1.0, pulse, frame.darkness);
    }
    falloff *= 1.0 - (0.2 + 0.2 * photon_daylight().noon) * light_levels.y;
    falloff += min(2.7 * b8 * b4, 0.9) * outer_fade;

    const WARM_BLOCKLIGHT = vec3(1.0, 0.56, 0.24);
    let base_luma = dot(WARM_BLOCKLIGHT, frame.luminance);
    var color = WARM_BLOCKLIGHT;
    var lpv_boost = vec3(0.0);
    let use_lpv_bounce = colored_lpv && !rt_restir_active;
    if use_lpv_bounce {
        // 100% GPU-driven 128x64x128 26-neighbor spherical L2 LPV with solid-wall blockers.
        let lpv = lpv_sample_trilinear(world_volumes_previous,
            frame.camera_position + position, lpv_anchor(frame.camera_position));
        let lpv_max = max(lpv.r, max(lpv.g, lpv.b));
        if lpv_max > 0.012 {
            let lpv_unit = lpv / lpv_max;
            let lpv_tint = lpv_unit * clamp(base_luma / max(dot(lpv_unit, frame.luminance), 0.18), 0.72, 1.38);
            let blend_w = smoothstep(0.012, 0.22, lpv_max);
            color = mix(color, lpv_tint, blend_w * 0.92);
            // Smooth spherical outer glow driven by L2 LPV so villages & open caves stay round, never diamond.
            let spherical_gate = smoothstep(0.02, 0.45, block + lpv_max * 0.45);
            let boost_scale = select(0.14, 0.26, blocklight_mode == 0u);
            lpv_boost = lpv * (boost_scale * ao * directional * spherical_gate);
        }
    }
    return color * (6.0 * falloff * directional) + lpv_boost;
}

fn terrain_lift(x: f32, amount: f32) -> f32 {
    return x * (1.0 + amount) / (1.0 + x * amount);
}

fn surface_fog(color: vec4<f32>, position: vec3<f32>) -> Surface {
    if use_underwater_fog() {
        return Surface(color);
    }
    var amount = vanilla_fog(position);
    if frame.dimension == 0 && frame.eye_in == 0u && frame.blindness <= 0.0 && frame.darkness <= 0.0 {
        // The post-process composite pass integrates outdoor volumetric haze AND
        // blends the render-distance border seamlessly into the live sky + cloud map.
        // Only short-range gameplay fog needs an early surface blend here.
        let local_fog = clamp((length(position) - frame.fog_environment.x)
            / max(frame.fog_environment.y - frame.fog_environment.x, 0.001), 0.0, 1.0);
        let short_range = 1.0 - smoothstep(32.0, 80.0, frame.fog_environment.y);
        amount = min(amount, short_range * local_fog);
        if amount <= 0.0 {
            return Surface(color);
        }
        let direction = position / max(length(position), 1e-4);
        let horizon = normalize(vec3(direction.x, clamp(direction.y, -0.10, 0.12), direction.z)
            + vec3(0.0, 0.0, 1e-6));
        let sky_fog = atmosphere(horizon);
        let cave = (1.0 - smoothstep(49.0, 63.0, frame.camera_position.y))
            * (1.0 - smoothstep(0.03, 0.30, frame.eye_light_smooth.y));
        return Surface(vec4(mix(color.rgb, sky_fog * (1.0 - cave), amount), color.a));
    }

    if frame.dimension == -1 && frame.eye_in == 0u && frame.blindness <= 0.0 && frame.darkness <= 0.0 {
        // Long-range height-stratified Nether volcanic ash haze + subtle lava under-glow.
        // Uses render_distance instead of vanilla_fog() (which Minecraft hard-caps at 96 blocks in the Nether).
        let dist = length(position);
        let world_p = frame.camera_position + position;
        let ash = 0.5 + 0.5 * sin(world_p.x * 0.042 + frame.time * 0.14)
            * cos(world_p.z * 0.038 - frame.time * 0.11 + world_p.y * 0.025);
        let lava_prox = 1.0 - smoothstep(28.0, 68.0, world_p.y);
        let density = mix(0.0026, 0.0046, ash) * (1.0 + 0.25 * lava_prox);
        let haze = (1.0 - exp(-dist * density)) * 0.78;
        let far_d = max(frame.render_distance, 64.0);
        let border = smoothstep(far_d * 0.72, far_d * 0.98, dist);
        let fog_factor = max(border, haze);
        let base_tint = pow(max(frame.fog_color, vec3(0.06, 0.02, 0.01)), vec3(2.2)) * 0.92;
        let ember_glow = vec3(0.14, 0.038, 0.009) * (0.25 + 0.85 * lava_prox) * (0.75 + 0.50 * ash);
        let fog_tint = base_tint + ember_glow;
        return Surface(vec4(mix(color.rgb, fog_tint, clamp(fog_factor, 0.0, 1.0)), color.a));
    }
    if frame.dimension == 1 && frame.eye_in == 0u && frame.blindness <= 0.0 && frame.darkness <= 0.0 {
        // Subtle cosmic purple-indigo End haze anchored to render distance so islands & obsidian
        // pillars stay crisp and high-contrast while distant void fades cleanly.
        let dist = length(position);
        let dir = position / max(dist, 1e-4);
        let end_sun_dir = normalize(vec3(-0.83, 0.31, 0.46));
        let cos_es = clamp(dot(dir, end_sun_dir), -1.0, 1.0);
        let haze = (1.0 - exp(-dist * 0.0014)) * 0.58;
        let far_d = max(frame.render_distance, 64.0);
        let border = smoothstep(far_d * 0.78, far_d * 0.99, dist);
        let fog_factor = max(border, haze);
        let base_end = mix(vec3(0.012, 0.008, 0.026), vec3(0.024, 0.013, 0.046),
            1.0 - smoothstep(0.0, 0.55, abs(dir.y)));
        let sun_scatter = vec3(0.065, 0.032, 0.055) * photon_phase(cos_es, 0.58) * smoothstep(48.0, 180.0, dist);
        return Surface(vec4(mix(color.rgb, base_end + sun_scatter, clamp(fog_factor, 0.0, 1.0)), color.a));
    }

    if amount <= 0.0 {
        return Surface(color);
    }
    let encoded = pow(max(color.rgb, vec3(0.0)), vec3(1.0 / 2.2));
    let fog = pow(max(frame.fog_color, vec3(0.0)), vec3(1.0 / 2.2));
    return Surface(vec4(pow(mix(encoded, fog, amount), vec3(2.2)), color.a));
}

