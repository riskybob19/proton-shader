override water_reflections: bool = true;
override native_water_mirror: bool = false;

// Photon's three-octave squared-sine profile, with Belmu's crest shape (unrolled: zero const array).
fn water_wave_height(coord: vec2<f32>, wave_direction: vec2<f32>, time: f32, flowing: bool) -> f32 {
    let sine = select(-0.675490, 0.0, flowing);
    let cosine = select(-0.737369, 1.0, flowing);
    let wave_rotation = mat2x2<f32>(vec2(cosine, sine), vec2(-sine, cosine));
    var direction = wave_direction;
    var noise_coord = (coord + vec2(0.0, 0.25 * time)) * 0.007;

    let n0 = textureSampleLevel(wave_noise, wave_noise_sampler, noise_coord, 0.0).r;
    let c0 = 0.5 + 0.5 * sin(7.8469878 * time - 6.2831853 * (0.7 * dot(direction, coord) + 2.0 * n0));
    var height = c0 * c0;

    direction = wave_rotation * direction;
    noise_coord *= 2.5;
    let n1 = textureSampleLevel(wave_noise, wave_noise_sampler, noise_coord, 0.0).r;
    let c1 = 0.5 + 0.5 * sin(6.4070387 * time - 4.1887902 * (1.19 * dot(direction, coord) + 2.0 * n1));
    height += 0.5 * c1 * c1;

    direction = wave_rotation * direction;
    noise_coord *= 2.5;
    let n2 = textureSampleLevel(wave_noise, wave_noise_sampler, noise_coord, 0.0).r;
    let c2 = 0.5 + 0.5 * sin(5.2313252 * time - 2.7925268 * (2.023 * dot(direction, coord) + 2.0 * n2));
    height += 0.25 * c2 * c2;

    let swell_uv = (coord + vec2(0.0, 0.1 * time)) * 0.001;
    let swell = textureSampleLevel(wave_noise, wave_noise_sampler, swell_uv, 0.0).r;
    return height * (1.0 / 1.75) * max(0.4, 2.0 * swell - 0.5);
}

// Analytic spatial gradient of Photon's three-octave squared-sine profile (unrolled: zero const array).
fn water_wave_slope_analytic(coord: vec2<f32>, wave_direction: vec2<f32>, time: f32, flowing: bool) -> vec2<f32> {
    let sine = select(-0.675490, 0.0, flowing);
    let cosine = select(-0.737369, 1.0, flowing);
    let wave_rotation = mat2x2<f32>(vec2(cosine, sine), vec2(-sine, cosine));
    var direction = wave_direction;
    var noise_coord = (coord + vec2(0.0, 0.25 * time)) * 0.007;

    let n0 = textureSampleLevel(wave_noise, wave_noise_sampler, noise_coord, 0.0).r;
    let p0 = 7.8469878 * time - 6.2831853 * (0.7 * dot(direction, coord) + 2.0 * n0);
    let c0 = 0.5 + 0.5 * sin(p0);
    var grad = direction * (-4.3982297 * c0 * cos(p0));

    direction = wave_rotation * direction;
    noise_coord *= 2.5;
    let n1 = textureSampleLevel(wave_noise, wave_noise_sampler, noise_coord, 0.0).r;
    let p1 = 6.4070387 * time - 4.1887902 * (1.19 * dot(direction, coord) + 2.0 * n1);
    let c1 = 0.5 + 0.5 * sin(p1);
    grad += direction * (-2.4923302 * c1 * cos(p1));

    direction = wave_rotation * direction;
    noise_coord *= 2.5;
    let n2 = textureSampleLevel(wave_noise, wave_noise_sampler, noise_coord, 0.0).r;
    let p2 = 5.2313252 * time - 2.7925268 * (2.023 * dot(direction, coord) + 2.0 * n2);
    let c2 = 0.5 + 0.5 * sin(p2);
    grad += direction * (-1.4123204 * c2 * cos(p2));

    let swell_uv = (coord + vec2(0.0, 0.1 * time)) * 0.001;
    let swell = textureSampleLevel(wave_noise, wave_noise_sampler, swell_uv, 0.0).r;
    return grad * (max(0.4, 2.0 * swell - 0.5) * (1.0 / 1.75));
}

// Warp broad wave crests in world space with wind-aligned non-commensurate
// wavelengths so distant water rolls organically without egg-carton spots.
fn water_swell_slope(position: vec2<f32>, time: f32) -> vec2<f32> {
    let broad = cloud_noise(position * 0.032);
    let local = cloud_noise(position * 0.071 + vec2(17.3, -9.7));
    let warp = vec2(broad.x - 0.5, local.x - 0.5);
    let broad_gradient = broad.yz * 0.032;
    let local_gradient = local.yz * 0.071;
    let a = vec2(0.46, 0.24);
    let b = vec2(-0.18, 0.62);
    let c = vec2(0.31, -0.19);
    let d = vec2(-0.26, -0.35);
    let phase = vec4(dot(a, position), dot(b, position), dot(c, position), dot(d, position))
        - time * vec4(0.95, 1.21, 0.74, 0.83)
        + vec4(5.4 * warp.x, 4.1 * warp.y, 3.2 * (warp.x - warp.y), 2.8 * (warp.x + warp.y));
    let crest = cos(phase) * vec4(0.75, 0.52, 0.42, 0.34);
    return crest.x * (a + 5.4 * broad_gradient)
        + crest.y * (b + 4.1 * local_gradient)
        + crest.z * (c + 3.2 * (broad_gradient - local_gradient))
        + crest.w * (d + 2.8 * (broad_gradient + local_gradient));
}

// Animated concentric raindrop impact rings on water surfaces during rain.
fn water_rain_ripple_slope(world_xz: vec2<f32>) -> vec2<f32> {
    let scaled = world_xz * 1.25;
    let cell = floor(scaled);
    let local = fract(scaled) - 0.5;
    let h1 = fract(sin(dot(cell, vec2(127.1, 311.7))) * 43758.5453);
    let h2 = fract(sin(dot(cell, vec2(269.5, 183.3))) * 43758.5453);
    let phase = fract(frame.time * 1.45 + h1);
    let center = (vec2(h1, h2) - 0.5) * 0.45;
    let delta = local - center;
    let dist = length(delta);
    let radius = phase * 0.46;
    let dr = (dist - radius) * 24.0;
    let om_phase = 1.0 - phase;
    let ring = exp(-dr * dr) * om_phase * om_phase;
    let wave = sin((dist - radius) * 36.0) * ring;
    return delta / max(dist, 0.02) * (wave * 0.14);
}

fn water_wave_normal(input: MaterialInput, view_ray: vec3<f32>, distance: f32,
    detail: vec2<f32>, coastal_damp: f32) -> vec3<f32> {
    let flat = input.normal;
    let incidence = abs(dot(view_ray, flat));
    let fade = detail.x;
    if wave_strength <= 0.0 {
        return flat;
    }
    let world = input.position + frame.camera_position;
    let sky = clamp(input.light.y, 0.0, 1.0);
    // Beaufort wind scaling + TMA finite-depth coastal wave attenuation (calm shallows,
    // full rolling swells in deep open water).
    let wind_scale = mix(0.55, 0.65 + 0.75 * weather_wind() + 0.35 * frame.rain, sky * (2.0 - sky))
        * mix(0.45, 1.0, coastal_damp);
    // Keep broad ripples after the fine wave detail falls below screen resolution.
    var swell_slope = vec3(0.0);
    if abs(flat.y) > 0.99 && detail.y > 0.0 {
        let slope = water_swell_slope(world.xz, frame.time);
        swell_slope = vec3(slope.x, 0.0, slope.y) * sign(flat.y) * wave_strength
            * 0.020 * wind_scale * detail.y;
    }
    if fade <= 0.0 {
        return normalize(flat + swell_slope);
    }
    let side = input.face >= 2u;
    let axis_u = select(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, -1.0), input.face >= 4u);
    let axis_v = select(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), side);
    var coord = -vec2(dot(world, axis_u), dot(world, axis_v));
    let flowing = side || abs(flat.y) < 0.99;
    var direction = vec2(0.8660254, 0.5);
    var normal_sign = select(-1.0, 1.0, flat.y >= 0.0);
    if side {
        direction = vec2(0.0, 1.0);
        let outward = cross(axis_u, axis_v) * select(-1.0, 1.0, input.face % 2u == 1u);
        normal_sign = select(-1.0, 1.0, dot(flat, outward) >= 0.0);
    } else if flowing {
        // Horizon's geometric normal faces the eye. Orient it up to find downhill.
        // Wave coordinates are negated, so negate downhill in wave space as well.
        direction = normalize(flat.xz) * -normal_sign;
    }

    let time = frame.time * select(0.5, 0.7, flowing);
    if water_parallax && distance < 18.0 && incidence > 0.09 {
        let parallax_fade = 1.0 - smoothstep(10.0, 18.0, distance);
        let tangent_view = vec2(dot(view_ray, axis_u), dot(view_ray, axis_v));
        let step_uv = -tangent_view * (0.046 * wave_strength * parallax_fade / max(incidence, 0.15));
        var previous_gap = -water_wave_height(coord, direction, time, flowing);
        for (var step = 1u; step <= 2u; step += 1u) {
            let next = coord + step_uv;
            let gap = f32(step) * 0.5 - water_wave_height(next, direction, time, flowing);
            if gap >= 0.0 {
                coord = mix(coord, next, clamp(-previous_gap / max(gap - previous_gap, 1e-4), 0.0, 1.0));
                break;
            }
            coord = next;
            previous_gap = gap;
        }
    }
    let slope = water_wave_slope_analytic(coord, direction, time, flowing);
    let still_strength = mix(0.01, (0.036 + 0.12 * frame.rain) * wind_scale, sky * (2.0 - sky));
    let influence = select(still_strength, 0.1, flowing) * (2.0 * wave_strength * fade);
    var ripple = axis_u * slope.x + axis_v * slope.y;
    // Subtle close-range capillary micro-ripples (non-axis-aligned, Nyquist-faded by detail.x
    // so they never form a square grid or alias into moire patterns at mid-range).
    var extra_slope = vec3(0.0);
    if !side && distance < 12.0 && fade > 0.05 {
        let cap_fade = (1.0 - smoothstep(4.0, 12.0, distance)) * fade * wave_strength
            * (0.35 + 0.65 * weather_wind()) * mix(0.45, 1.0, coastal_damp);
        let d1 = vec2(0.877, 0.480);
        let d2 = vec2(-0.515, 0.857);
        let c1 = cos(dot(world.xz, d1) * 7.5 - frame.time * 3.4);
        let c2 = cos(dot(world.xz, d2) * 10.2 + frame.time * 4.1);
        let cap_grad = (d1 * c1 + d2 * (0.72 * c2)) * (0.016 * cap_fade);
        extra_slope += axis_u * cap_grad.x + axis_v * cap_grad.y;
    }
    if !side && frame.rain > 0.0 && sky > 0.75 && distance < 26.0 {
        let rain_fade = frame.rain * (1.0 - smoothstep(14.0, 26.0, distance)) * smoothstep(0.75, 0.95, sky);
        let splash = water_rain_ripple_slope(world.xz) + water_rain_ripple_slope(world.xz + vec2(13.7, 8.3));
        extra_slope += (axis_u * splash.x + axis_v * splash.y) * (rain_fade * 1.4);
    }
    let tangent_ripple = ripple - flat * dot(flat, ripple);
    let tangent_extra = extra_slope - flat * dot(flat, extra_slope);
    let normal = normalize(flat + swell_slope + (tangent_ripple * influence + tangent_extra) * normal_sign);
    // Extreme grazing angles can place the perturbed surface behind the eye ray.
    return select(flat, normal, dot(-view_ray, normal) > 0.001);
}

const WATER_IOR: f32 = 1.333;
const WATER_MAX_THICKNESS: f32 = 28.0;
const WATER_UV_TO_NDC = vec2(2.0, -2.0);

fn water_opaque_depth(pixel: vec2<i32>) -> f32 {
    return textureLoad(opaque_depth, pixel, 0);
}

fn water_opaque_color(pixel: vec2<i32>) -> vec3<f32> {
    return textureLoad(opaque_color, pixel, 0).rgb;
}

fn water_mirror_sample(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(reflection, reflection_sampler, uv, 0.0);
}

fn water_view_position(ndc: vec2<f32>, depth: f32) -> vec3<f32> {
    let position = frame.projection_inverse * vec4(ndc, depth, 1.0);
    return position.xyz / max(abs(position.w), 0.00001);
}

fn water_pixel(uv: vec2<f32>, size: vec2<i32>) -> vec2<i32> {
    return clamp(vec2<i32>(uv * vec2<f32>(size)), vec2(0), size - 1);
}

fn water_ndc(pixel: vec2<i32>, size: vec2<i32>) -> vec2<f32> {
    return ((vec2<f32>(pixel) + 0.5) / vec2<f32>(size) - 0.5) * WATER_UV_TO_NDC;
}

fn water_screen_edge(uv: vec2<f32>, margin: f32) -> f32 {
    let edge = min(uv, 1.0 - uv);
    return smoothstep(0.003, margin, min(edge.x, edge.y));
}

fn water_limit_offset(offset: vec2<f32>, limit: f32) -> vec2<f32> {
    return offset / max(1.0, length(offset) / limit);
}

fn water_fresnel(cosine: f32, eta: f32) -> f32 {
    let sin2_transmitted = eta * eta * (1.0 - cosine * cosine);
    if sin2_transmitted >= 1.0 {
        return 1.0;
    }
    let transmitted = sqrt(1.0 - sin2_transmitted);
    let perpendicular = (eta * cosine - transmitted) / (eta * cosine + transmitted);
    let parallel = (cosine - eta * transmitted) / (cosine + eta * transmitted);
    return 0.5 * (perpendicular * perpendicular + parallel * parallel);
}

fn water_light_visibility(position: vec3<f32>, normal: vec3<f32>) -> f32 {
    if !pack_shadow_active() {
        return 1.0;
    }
    let bias = select(0.025, pack_shadow_texel() * 0.45, frame.has_shadow != 0u);
    let shadow = pack_shadow_position(position + normal * bias);
    if any(shadow < vec3(0.0)) || any(shadow > vec3(1.0)) {
        return 1.0;
    }
    if frame.has_shadow != 0u {
        return pack_shadow_compare(shadow.xy, shadow.z - 0.00024, false);
    }
    let receiver = ray_shadow_surface(shadow, normal);
    let fade = max(pack_shadow_edge_fade(position, shadow), receiver.grazing_fade);
    if fade >= 1.0 {
        return 1.0;
    }
    let visible = ray_shadow_compare(shadow.xy, receiver, false, true);
    return mix(visible, 1.0, fade);
}

fn water_refracted_sample(input: MaterialInput, pixel: vec2<i32>, size: vec2<i32>, bottom: vec3<f32>,
    thickness: f32, view_ray: vec3<f32>, normal: vec3<f32>, eta: f32, unwarped: vec3<f32>) -> vec4<f32> {
    let original = vec4(unwarped, thickness);
    let edge = water_screen_edge(input.screen, 0.045);
    if thickness <= 0.025 || edge <= 0.0 || dot(normal, input.normal) > 0.999999 {
        return original;
    }

    // Like Photon, only the ripple displaces the background. Comparing the
    // two refracted rays keeps the air/water index and removes the flat bend.
    let flat_ray = refract(view_ray, input.normal, eta);
    let wavy_ray = refract(view_ray, normal, eta);
    if dot(flat_ray, flat_ray) < 0.5 || dot(wavy_ray, wavy_ray) < 0.5 {
        return original;
    }
    let bend = (frame.view * vec4(wavy_ray - flat_ray, 0.0)).xyz;
    let shifted = frame.projection * vec4(bottom + bend * min(thickness, 6.0), 1.0);
    if shifted.w <= 0.01 {
        return original;
    }
    let viewport = vec2<f32>(size);
    let shifted_ndc = shifted.xy / shifted.w;
    let displacement = (shifted_ndc - water_ndc(pixel, size)) / WATER_UV_TO_NDC * viewport;
    let offset = water_limit_offset(displacement, 0.009 * viewport.y) * edge;
    if dot(offset, offset) < 0.16 {
        return original;
    }
    let sample_at = vec2<f32>(pixel) + offset;
    let corner = vec2<i32>(floor(sample_at));
    let blend = fract(sample_at);
    let water_view = (frame.view * vec4(input.position, 1.0)).xyz;
    let view_normal = (frame.view * vec4(input.normal, 0.0)).xyz;
    let depth_limit = min(0.85, 0.20 + 0.05 * thickness + 0.002 * length(bottom));
    var filtered = vec4(0.0);
    var used = 0.0;

    // Filter only samples from the same depth layer. Rejected taps keep their
    // share of the undistorted image, so the silhouette fades instead of jumping.
    for (var tap = 0u; tap < 4u; tap += 1u) {
        let cell = vec2<i32>(i32(tap & 1u), i32(tap >> 1u));
        let sample_pixel = corner + cell;
        if any(sample_pixel < vec2(0)) || any(sample_pixel >= size) {
            continue;
        }
        let depth = water_opaque_depth(sample_pixel);
        if depth <= 0.0 {
            continue;
        }
        let position = water_view_position(water_ndc(sample_pixel, size), depth);
        let below = dot(view_normal, water_view - position);
        let behind = water_view.z - position.z;
        let gap = abs(position.z - bottom.z);
        let coverage = smoothstep(0.015, 0.16, min(below, behind));
        let continuity = 1.0 - smoothstep(0.4 * depth_limit, depth_limit, gap);
        let kernel = select(1.0 - blend, blend, cell == vec2(1));
        let weight = kernel.x * kernel.y * coverage * continuity;
        let path = min(length(position - water_view), WATER_MAX_THICKNESS);
        filtered += vec4(water_opaque_color(sample_pixel), path) * weight;
        used += weight;
    }
    return filtered + original * max(0.0, 1.0 - used);
}

fn mirror_soft_clamp(rgb: vec3<f32>) -> vec3<f32> {
    let safe = max(rgb, vec3(0.0));
    let peak = max(safe.r, max(safe.g, safe.b));
    return safe * (1.0 / (1.0 + 0.22 * peak));
}

fn mirror_terrain_weight(sample_rgb: vec3<f32>, sample_a: f32, clear_atmo: vec3<f32>, vanilla_sky: vec3<f32>) -> f32 {
    let diff_atmo = length(sample_rgb - clear_atmo) / max(length(clear_atmo), 0.08);
    let diff_vanilla = length(sample_rgb - vanilla_sky) / max(length(vanilla_sky), 0.08);
    let sky_diff = min(diff_atmo, diff_vanilla);
    return smoothstep(0.02, 0.95, sample_a) * smoothstep(0.06, 0.18, sky_diff);
}

fn water_plane_reflection(input: MaterialInput, ray: vec3<f32>, surface_distance: f32, lost_variance: f32) -> vec4<f32> {
    if frame.has_reflection == 0u || input.face != 1u || input.normal.y <= 0.90 {
        return vec4(0.0);
    }
    let plane = frame.water_plane - frame.camera_position.y;
    let height_error = abs(plane - input.position.y);
    // Scale plane-height tolerance with distance + wave displacement so distant rivers
    // and displaced vertex waves never pop in/out of Horizon's planar mirror.
    let tol_lo = 0.14 + surface_distance * 0.0025;
    let tol_hi = 0.42 + surface_distance * 0.0060;
    let coverage = (1.0 - smoothstep(tol_lo, tol_hi, height_error))
        * smoothstep(0.90, 0.985, input.normal.y) * smoothstep(-0.18, -0.02, ray.y);
    if coverage <= 0.0 {
        return vec4(0.0);
    }

    // Horizon projects the flipped world about this plane. Anchor the mirror lookup
    // directly on the mathematical water plane so wave displacement never bounces UVs.
    let mirror_position = vec3(input.position.x, plane, input.position.z);
    let direction = ray * vec3(1.0, -1.0, 1.0);
    let probe = mirror_position + direction * max(length(input.position), 1.0);
    let clip = frame.projection * frame.view * vec4(probe, 1.0);
    var uv = input.screen;
    if clip.w > 0.01 {
        let displaced = 0.5 + clip.xy / clip.w / WATER_UV_TO_NDC;
        let max_offset = mix(0.024, 0.004, smoothstep(14.0, 95.0, surface_distance));
        let offset = water_limit_offset(displaced - input.screen, max_offset);
        uv += offset * water_screen_edge(input.screen, 0.075);
    }

    let lo = vec2(0.002, 0.002);
    let hi = vec2(0.998, 0.998);
    let sky_ray = normalize(vec3(ray.x, max(ray.y, 0.005), ray.z));
    let clear_atmo = atmosphere(sky_ray);
    let vanilla_sky = pow(max(frame.fog_color, vec3(0.0)), vec3(2.2));

    // Unrolled 3/5-tap anisotropic elliptical Cox-Munk filter (zero local array spill in [material]).
    let rough_blur = clamp(sqrt(lost_variance) * 2.2 + smoothstep(16.0, 110.0, surface_distance) * 0.55, 0.0, 1.0);
    let dy = mix(0.0022, 0.0075, rough_blur);
    let dx = dy * 0.38;

    let t0 = water_mirror_sample(clamp(uv, lo, hi));
    let wk0 = mirror_terrain_weight(t0.rgb, t0.a, clear_atmo, vanilla_sky);
    let t1 = water_mirror_sample(clamp(uv + vec2(0.0, -dy), lo, hi));
    let wk1 = mirror_terrain_weight(t1.rgb, t1.a, clear_atmo, vanilla_sky);
    let t2 = water_mirror_sample(clamp(uv + vec2(0.0, dy), lo, hi));
    let wk2 = mirror_terrain_weight(t2.rgb, t2.a, clear_atmo, vanilla_sky);
    var alpha_avg = 0.44 * wk0 + 0.28 * (wk1 + wk2);
    var color_sum = mirror_soft_clamp(t0.rgb) * (0.44 * wk0)
        + mirror_soft_clamp(t1.rgb) * (0.28 * wk1)
        + mirror_soft_clamp(t2.rgb) * (0.28 * wk2);
    var weight_norm = 1.0;
    if rough_blur > 0.30 {
        let t3 = water_mirror_sample(clamp(uv + vec2(-dx, 0.0), lo, hi));
        let wk3 = mirror_terrain_weight(t3.rgb, t3.a, clear_atmo, vanilla_sky);
        let t4 = water_mirror_sample(clamp(uv + vec2(dx, 0.0), lo, hi));
        let wk4 = mirror_terrain_weight(t4.rgb, t4.a, clear_atmo, vanilla_sky);
        alpha_avg += 0.14 * (wk3 + wk4);
        color_sum += mirror_soft_clamp(t3.rgb) * (0.14 * wk3) + mirror_soft_clamp(t4.rgb) * (0.14 * wk4);
        weight_norm = 1.28;
    }
    let norm_alpha = alpha_avg / weight_norm;
    if norm_alpha <= 0.002 {
        return vec4(0.0);
    }
    return vec4(color_sum / max(alpha_avg, 1e-4), coverage * clamp(norm_alpha, 0.0, 1.0));
}

fn reflection_from_scene(input: MaterialInput, ray: vec3<f32>, view_ray: vec3<f32>,
    size: vec2<i32>, reach: f32, surface_distance: f32, lost_variance: f32) -> vec4<f32> {
    if surface_distance > 550.0 {
        return vec4(0.0);
    }
    // Clamp grazing back-projected rays that cause severe stretched-cliff artifacts.
    let back_fade = 1.0 - smoothstep(0.62, 0.92, dot(ray, -view_ray));
    if back_fade <= 0.001 {
        return vec4(0.0);
    }
    let origin = input.position + input.normal * clamp(0.055 + surface_distance * 0.0012, 0.055, 0.35);
    let view_projection = frame.projection * frame.view;
    let start_clip = view_projection * vec4(origin, 1.0);
    let step_clip = view_projection * vec4(ray, 0.0);
    // Static spatial IGN dither (no temporal frame_index strobe) breaks up step
    // boundaries while preventing thin islands/cliffs from flickering HIT/MISS every frame.
    let ign = fract(52.9829189 * fract(dot(floor(input.screen * vec2<f32>(size)), vec2(0.06711056, 0.00583715))));
    var previous = 0.0;
    let far_water = surface_distance > 84.0;
    let max_steps = select(11, 8, far_water);
    let max_refine = select(3, 2, far_water);
    let step_grow = select(1.44, 1.52, far_water);
    var step_len = clamp(0.42 + surface_distance * 0.012, 0.42, 3.2);
    for (var step = 1; step <= max_steps; step += 1) {
        if previous >= reach {
            break;
        }
        let next_t = min(reach, previous + step_len);
        let probe_t = mix(previous, next_t, 0.55 + 0.45 * ign);
        let clip = start_clip + probe_t * step_clip;
        if clip.w <= 0.01 {
            break;
        }
        let ndc = clip.xyz / clip.w;
        if any(abs(ndc.xy) >= vec2(1.0)) || ndc.z <= 0.0 || ndc.z >= 1.0 {
            break;
        }
        let screen = 0.5 + ndc.xy / WATER_UV_TO_NDC;
        let pixel = water_pixel(screen, size);
        let solid = water_opaque_depth(pixel);
        if solid > ndc.z {
            var near = previous;
            var far = probe_t;
            var hit_uv = screen;
            var hit_pixel = pixel;
            var hit_depth = solid;
            for (var refine = 0; refine < max_refine; refine += 1) {
                let middle = 0.5 * (near + far);
                let probe = start_clip + step_clip * middle;
                if probe.w <= 0.01 || any(abs(probe.xy) >= vec2(probe.w)) {
                    return vec4(0.0);
                }
                let projected = probe.xyz / probe.w;
                let uv = 0.5 + projected.xy / WATER_UV_TO_NDC;
                let cell = water_pixel(uv, size);
                let depth = water_opaque_depth(cell);
                if depth > projected.z {
                    far = middle;
                    hit_uv = uv;
                    hit_pixel = cell;
                    hit_depth = depth;
                } else {
                    near = middle;
                }
            }

            let hit_view = water_view_position(water_ndc(hit_pixel, size), hit_depth);
            let sample_position = (frame.view_inverse * vec4(hit_view, 1.0)).xyz;
            let ray_position = origin + ray * far;
            let error = distance(sample_position, ray_position);
            let step_span = max(probe_t - previous, 0.25);
            let tolerance = clamp(0.55 + 0.038 * length(hit_view) + 0.50 * step_span, 0.55, 26.0);
            // Discard false hits that jump across a depth edge or lie across the water plane.
            let side = dot(sample_position - input.position, input.normal);
            if side < -0.25 || error > tolerance || far < 0.14 {
                // Keep marching past foreground islands/hills so distant mountains behind them still hit.
                previous = probe_t;
                step_len *= step_grow;
                continue;
            }
            let confidence = water_screen_edge(hit_uv, 0.08)
                * (1.0 - smoothstep(0.68 * tolerance, tolerance, error))
                * smoothstep(-0.25, 0.12, side) * back_fade;
            let current = water_opaque_color(hit_pixel);
            var filtered = current;
            if !far_water && far > 6.0 {
                // Unrolled 2-tap vertical anisotropic water blur (zero local array spill).
                let aniso_weight = smoothstep(6.0, 28.0, far) * clamp(0.45 + 2.2 * sqrt(lost_variance), 0.0, 1.0);
                if aniso_weight > 0.12 {
                    let inv_hd = 1.0 / max(hit_depth, 1e-5);
                    let w1 = 0.34 * aniso_weight;
                    let p_up = clamp(hit_pixel + vec2(0, -1), vec2(0), size - 1);
                    let p_dn = clamp(hit_pixel + vec2(0, 1), vec2(0), size - 1);
                    let w_up = w1 * (1.0 - smoothstep(0.02, 0.07, abs(water_opaque_depth(p_up) - hit_depth) * inv_hd));
                    let w_dn = w1 * (1.0 - smoothstep(0.02, 0.07, abs(water_opaque_depth(p_dn) - hit_depth) * inv_hd));
                    filtered = (filtered + water_opaque_color(p_up) * w_up + water_opaque_color(p_dn) * w_dn)
                        / (1.0 + w_up + w_dn);
                }
            }
            let color = select(filtered, reflected_history(sample_position, filtered, 0.0), !far_water);
            var aerial: vec3<f32>;
            if frame.eye_in == 1u {
                // Underwater TIR reflection: attenuate reflected seabed/coral by water extinction
                // instead of applying above-water air aerial perspective.
                let uw_trans = exp(-vec3(0.24, 0.10, 0.052) * min(far, 42.0));
                aerial = color * uw_trans;
            } else {
                aerial = reflected_aerial_color(color, ray, far, input.light.y);
            }
            // Soft firefly knee suppresses single-pixel grazing HDR spikes before Fresnel blending.
            let peak = max(aerial.r, max(aerial.g, aerial.b));
            let stable = aerial * (1.0 / (1.0 + 0.08 * max(peak - 1.5, 0.0)));
            return vec4(stable, confidence);
        }
        previous = probe_t;
        step_len *= step_grow;
    }
    return vec4(0.0);
}

// Fast 6-step Decima heightfield fallback for distant off-screen mountain silhouettes in water
// (keeps [material] 100% free of SPV_KHR_ray_query register spill while reflecting 4..460m peaks).
fn water_heightfield_reflection(origin: vec3<f32>, ray: vec3<f32>, start_dist: f32, sky_access: f32) -> vec4<f32> {
    if ray.y < -0.03 || ray.y > 0.34 {
        return vec4(0.0);
    }
    let anchor = hf_anchor(frame.camera_position.xz);
    let world_origin = frame.camera_position + origin;
    let max_d = min(460.0, max(start_dist + 88.0, (290.0 - world_origin.y) / max(ray.y, 0.05)));
    let span = max(max_d - start_dist, 1.0);
    const STEPS: u32 = 6u;
    let hf_size = vec2<i32>(i32(HF_SIZE), i32(HF_SIZE));
    for (var i = 0u; i < STEPS; i += 1u) {
        let t = (f32(i) + 0.5) / f32(STEPS);
        let d = start_dist + span * (0.22 * t + 0.78 * t * t);
        let p_w = world_origin + ray * d;
        let uv = hf_world_to_uv(p_w.xz, anchor);
        if any(uv <= vec2(0.01)) || any(uv >= vec2(0.99)) {
            break;
        }
        let hf_cell = clamp(vec2<i32>(uv * vec2<f32>(hf_size)), vec2(0), hf_size - 1);
        let hf = textureLoad(world_volumes_previous, hf_cell, 0);
        if hf.g > 0.35 && hf.r > p_w.y + 1.2 {
            let snow_w = smoothstep(116.0, 142.0, hf.r);
            let mtn_albedo = mix(vec3(0.09, 0.13, 0.08), vec3(0.62, 0.66, 0.72), snow_w);
            let sun_term = direct_atmosphere_light() * max(frame.light_direction.y, 0.0) * 0.45;
            let sky_term = photon_ambient_irradiance() * 0.78;
            let lit = (mtn_albedo / PHOTON_PI) * (sun_term + sky_term);
            let fogged = reflected_aerial_color(lit, ray, d, sky_access);
            let conf = smoothstep(0.35, 0.70, hf.g) * (1.0 - smoothstep(360.0, 460.0, d));
            return vec4(fogged, conf * 0.90);
        }
    }
    return vec4(0.0);
}

fn caustic_ray_offset(depth: f32, inv_ior: f32) -> vec2<f32> {
    let ray = refract(-frame.light_direction, vec3(0.0, 1.0, 0.0), inv_ior);
    return ray.xz * (depth / max(-ray.y, 0.22));
}

// Single-pass sub-pixel Lagrangian wave-folding caustic kernel with true directional
// Snell's-law RGB Cauchy chromatic dispersion (4.2x fewer trig ops than 3x full loops).
fn caustic_web_rgb(coord: vec2<f32>, disp_r: vec2<f32>, disp_b: vec2<f32>, t: f32, sharpness: f32) -> vec3<f32> {
    let rot1 = mat2x2<f32>(vec2(0.9563, 0.2924), vec2(-0.2924, 0.9563));
    let p1 = fract((rot1 * coord) * 0.145) * TAU;
    let dp_r = (rot1 * disp_r) * (0.145 * TAU);
    let dp_b = (rot1 * disp_b) * (0.145 * TAU);
    var i1 = p1;
    var sum_r = 1.0;
    var sum_g = 1.0;
    var sum_b = 1.0;
    for (var n = 0u; n < 4u; n += 1u) {
        let tn = t * (1.0 - 3.2 / f32(n + 1u));
        i1 = p1 + vec2(cos(tn - i1.x) + sin(tn + i1.y), sin(tn - i1.y) + cos(tn + i1.x));
        let sxg = sin(i1.x + tn);
        let cyg = cos(i1.y + tn);
        sum_g += abs(sxg * cyg) / (1.18 * sqrt(sxg * sxg + cyg * cyg + 1e-5));
        let sxr = sin(i1.x + dp_r.x + tn);
        let cyr = cos(i1.y + dp_r.y + tn);
        sum_r += abs(sxr * cyr) / (1.18 * sqrt(sxr * sxr + cyr * cyr + 1e-5));
        let sxb = sin(i1.x + dp_b.x + tn);
        let cyb = cos(i1.y + dp_b.y + tn);
        sum_b += abs(sxb * cyb) / (1.18 * sqrt(sxb * sxb + cyb * cyb + 1e-5));
    }
    let c1 = clamp(vec3(1.16) - pow(vec3(sum_r, sum_g, sum_b) * 0.25, vec3(1.38)), vec3(0.0), vec3(1.0));
    let web1 = pow(c1, vec3(sharpness));

    // Secondary finer capillary web (2-iteration shared shimmer to break tiling).
    let rot2 = mat2x2<f32>(vec2(0.4540, -0.8910), vec2(0.8910, 0.4540));
    let p2 = fract((rot2 * coord) * 0.265) * TAU;
    var i2 = p2;
    var sum2 = 1.0;
    for (var m = 0u; m < 2u; m += 1u) {
        let tm = t * 1.18 * (1.0 - 2.6 / f32(m + 1u));
        i2 = p2 + vec2(cos(tm - i2.x) + sin(tm + i2.y), sin(tm - i2.y) + cos(tm + i2.x));
        let sx = sin(i2.x + tm);
        let cy = cos(i2.y + tm);
        sum2 += abs(sx * cy) / (1.18 * sqrt(sx * sx + cy * cy + 1e-5));
    }
    let c2 = clamp(1.15 - pow(sum2 * 0.45, 1.35), 0.0, 1.0);
    let web2 = pow(c2, sharpness * 0.85);

    return web1 * 0.72 + vec3(web2 * 0.36);
}

// Photorealistic analytic water caustics with true Snell's-law RGB Cauchy chromatic
// dispersion, depth-dependent focal sharpening/defocusing, and Beaufort wind coupling.
fn caustic_pattern(position: vec3<f32>, depth: f32) -> vec3<f32> {
    let contrast = wave_strength * smoothstep(0.03, 0.28, depth)
        * (1.0 - smoothstep(8.5, 13.5, depth)) * (1.0 - smoothstep(22.0, 56.0, length(position)));
    if contrast <= 0.0 {
        return vec3(0.0);
    }
    let world = frame.camera_position + position;
    let wind_speed = mix(0.82, 1.28, weather_wind());
    let t = frame.time * 0.48 * wind_speed;
    // Sharpest filaments in the 0.3..3.0m shallow focal zone; softly defocus deeper down.
    let sharpness = mix(7.2, 2.8, smoothstep(2.2, 11.0, depth));

    // Physical RGB Cauchy dispersion (red bends least, blue bends most along the sun ray).
    let disp_depth = min(depth, 4.5);
    let base_g = world.xz + caustic_ray_offset(depth, 1.0 / 1.334);
    let disp_r = (caustic_ray_offset(disp_depth, 1.0 / 1.320) - caustic_ray_offset(disp_depth, 1.0 / 1.334)) * 1.35;
    let disp_b = (caustic_ray_offset(disp_depth, 1.0 / 1.352) - caustic_ray_offset(disp_depth, 1.0 / 1.334)) * 1.35;

    let raw = caustic_web_rgb(base_g, disp_r, disp_b, t, sharpness);
    let gain = clamp(raw * 2.35 - vec3(0.28), vec3(-0.32), vec3(1.85));
    let wind_contrast = mix(0.85, 1.18, weather_wind());
    let atten = min(1.35 * contrast * wind_contrast, 1.15) * exp(-0.11 * depth) * (1.0 - 0.75 * frame.rain);
    return gain * atten;
}

fn refracted_water(input: MaterialInput) -> vec4<f32> {
    let surface_distance = length(input.position);
    let view_ray = input.position / max(surface_distance, 0.00001);
    let submerged = frame.eye_in == 1u;
    let sun_side = !submerged && input.normal.y > 0.0;
    let size = vec2<i32>(textureDimensions(opaque_depth));
    let pixel = water_pixel(input.screen, size);
    let background_depth = water_opaque_depth(pixel);
    var bottom_view = vec3(0.0);
    var water_depth = WATER_MAX_THICKNESS;
    var floor_world = vec3(0.0);
    var vertical_depth = WATER_MAX_THICKNESS;
    if background_depth > 0.0 {
        bottom_view = water_view_position(water_ndc(pixel, size), background_depth);
        water_depth = clamp(length(bottom_view) - surface_distance, 0.0, WATER_MAX_THICKNESS);
        floor_world = (frame.view_inverse * vec4(bottom_view, 1.0)).xyz;
        vertical_depth = max(input.position.y - floor_world.y, 0.0);
    }
    let coastal_damp = select(1.0, smoothstep(0.08, 3.2, vertical_depth), sun_side && background_depth > 0.0);

    // Estimate pixel footprint on the water plane with an elevation-independent distance
    // floor so high-frequency ripples never re-alias and strobe when flying high above water.
    let view_depth = abs((frame.view * vec4(input.position, 1.0)).z);
    let height_pixels = f32(size.y);
    let proj_footprint = 2.0 * view_depth / max(height_pixels * abs(frame.projection[1].y)
        * max(abs(dot(view_ray, input.normal)), 0.002), 0.001);
    let footprint = max(proj_footprint, surface_distance * 0.018);
    // Wide, overlapping LOD rolloff so fine ripples dissolve seamlessly into broad swells.
    let detail = 1.0 - smoothstep(vec2(0.12, 0.55), vec2(1.35, 3.8), vec2(footprint));
    let slope_rms = wave_strength * vec2(0.11 + 0.20 * frame.rain, 0.038);
    let lost_variance = dot(slope_rms * slope_rms, 1.0 - detail * detail);
    let normal = water_wave_normal(input, view_ray, surface_distance, detail, coastal_damp);
    let eta = select(1.0 / WATER_IOR, WATER_IOR, submerged);
    let incidence = clamp(dot(-view_ray, normal), 0.0, 1.0);
    let optical_fresnel = water_fresnel(incidence, eta);
    // Rough-surface Fresnel attenuation (energy-conserving microfacet grazing):
    // choppy/distant unresolved ripples soften extreme grazing mirror reflection
    // so windy oceans retain rich deep-water color instead of turning flat grey.
    let rough_fresnel_damp = 1.0 - clamp(sqrt(lost_variance) * 1.4, 0.0, 0.38);
    let fresnel = select(water_fresnel(incidence, 1.0 / WATER_IOR), optical_fresnel, !submerged || snells_window)
        * select(rough_fresnel_damp, 1.0, submerged);
    let reflected_ray = reflect(view_ray, normal);

    let sky_access = smoothstep(0.04, 0.90, input.light.y);
    let sky_normal = select(input.normal, vec3(0.0, 1.0, 0.0), submerged);
    let sky_light = terrain_skylight(sky_normal, input.light.y, input.ao, 0.0, false);
    let block_light = surface_blocklight(input.position, input.light, input.ao, 1.0);
    let ambient = (sky_light + block_light + vec3(1.5 * frame.night_vision)) / PHOTON_PI;
    let sunlight = direct_atmosphere_light() * cloud_shadow(input.position);
    // Physically-based Jerlov water absorption (red extinguishes ~5.5x faster than blue)
    // coupled cleanly to Minecraft biome water tint (Warm Ocean turquoise -> Swamp tannins).
    let absorption = vec3(0.41, 0.145, 0.075);
    let biome_tint = clamp(input.tint.rgb, vec3(0.02), vec3(1.0));
    let biome_delta = log(vec3(0.1245, 0.1797, 0.7108) / biome_tint);
    let extinction = max(absorption + select(vec3(0.0), 0.14 * biome_delta, biome_water_tint), vec3(0.022)) + 0.014;
    var behind = water_opaque_color(pixel);
    var optical_depth = water_depth;
    if background_depth > 0.0 && fresnel < 0.88 && surface_distance < 34.0 {
        let refracted_sample = water_refracted_sample(input, pixel, size, bottom_view, water_depth, view_ray, normal, eta, behind);
        let refract_fade = 1.0 - smoothstep(22.0, 34.0, surface_distance);
        behind = mix(behind, refracted_sample.rgb, refract_fade);
        optical_depth = mix(optical_depth, refracted_sample.a, refract_fade);
    }
    let medium_distance = select(optical_depth, min(surface_distance, WATER_MAX_THICKNESS), submerged);
    let transmission = exp(-extinction * medium_distance);
    let solar_energy = dot(sunlight, frame.luminance);
    var lit_water = 0.0;
    if sun_side && sky_access > 0.0 && solar_energy > 0.001 {
        lit_water = sky_access * water_light_visibility(input.position, input.normal);
    }
    // Henyey-Greenstein forward scattering in water (hydrosols, g = 0.72) + wave-crest
    // subsurface glow when sunlight passes through tilted crests toward the viewer.
    let cos_vl = clamp(dot(view_ray, frame.light_direction), -1.0, 1.0);
    let hg_d = max(1.0 + 0.5184 - 1.44 * cos_vl, 0.05);
    let water_phase = mix(0.28, 0.4816 / (4.0 * PHOTON_PI * hg_d * sqrt(hg_d)), 0.55);
    let crest_glow = clamp(dot(normal - input.normal, -view_ray) * 4.5, 0.0, 1.0) * max(cos_vl, 0.0);
    // scatter_albedo already represents the single-scattering reflectance (sigma_s / sigma_t).
    let scatter_albedo = mix(vec3(0.020, 0.072, 0.135),
        clamp(biome_tint * vec3(0.10, 0.16, 0.14), vec3(0.012), vec3(0.16)),
        select(0.0, 0.45, biome_water_tint));
    // Monzón et al. vertical downwelling spectral attenuation: sunlight traveling down
    // through the water column loses red wavelengths before scattering back to the eye.
    let downwelling = exp(-extinction * (0.42 * min(max(vertical_depth, 0.0), 16.0)));
    let sun_scatter = sunlight * downwelling * (lit_water * (0.22 * water_phase + 0.14 * crest_glow));
    let scatter = scatter_albedo * (ambient + sun_scatter);
    // Above water the visible surface supplies the bed depth, even if the
    // shadow atlas has no water caster for the corresponding light ray.
    if water_caustics && lit_water > 0.0 && background_depth > 0.0 {
        let depth = vertical_depth;
        if depth > 0.02 && depth < 13.5 {
            let focus = caustic_pattern(floor_world, depth);
            let direct_fraction = sunlight / max(ambient + sunlight, vec3(0.001));
            behind *= max(vec3(0.18), vec3(1.0) + direct_fraction * focus * lit_water);
        }
    }

    var reflection_color = scatter;
    if water_reflections && sun_side && input.normal.y > 0.65 {
        // Clamp sky reflection ray slightly above the horizon instead of hard-branching
        // on reflected_ray.y > 0.0, which caused a bright white/dark strobe on distant
        // shoreline waves whenever ripples crossed y = 0.
        let sky_ray = normalize(vec3(reflected_ray.x, max(reflected_ray.y, 0.005), reflected_ray.z));
        var sky = sample_environment(sky_ray);
        if lost_variance > 0.002 {
            // A small normal tilt rotates the reflected ray by about twice that angle.
            let axis = select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), sky_ray.y > 0.98);
            let spread = normalize(cross(axis, sky_ray)) * (2.0 * sqrt(lost_variance));
            let s_pos = normalize(sky_ray + spread);
            let s_neg = normalize(sky_ray - spread);
            sky = 0.5 * sky + 0.25 * (sample_environment(normalize(vec3(s_pos.x, max(s_pos.y, 0.005), s_pos.z)))
                + sample_environment(normalize(vec3(s_neg.x, max(s_neg.y, 0.005), s_neg.z))));
        }
        let openness = smoothstep(0.0, 0.75, input.light.y);
        let celestial_damp = 1.0 - clamp(sqrt(lost_variance) * 3.5, 0.0, 0.85);
        let sky_horiz_fade = smoothstep(-0.18, 0.02, reflected_ray.y);
        let sky_refl = (sky.rgb + sky.a * sky_celestial(sky_ray) * celestial_damp) * openness;
        reflection_color = mix(scatter, sky_refl, sky_horiz_fade);
        if native_water_mirror && (!rt_reflections || frame.has_ray_tracing == 0u) {
            let mirror = water_plane_reflection(input, reflected_ray, surface_distance, lost_variance);
            let mirror_alpha = mirror.a * sky_access;
            // Exposure- and mood-grade Horizon's native mirror to match our shaded atmosphere.
            let sky_mood = clamp(dot(direct_atmosphere_light() + ambient_atmosphere_light(), frame.luminance), 0.06, 1.6);
            let mirror_grade = mix(vec3(0.26, 0.32, 0.42), vec3(0.72, 0.71, 0.70), smoothstep(0.08, 0.90, sky_mood));
            reflection_color = mix(reflection_color, mirror.rgb * mirror_grade, mirror_alpha);
            // Smoothly fade into SSR fallback only where Horizon's mirror texture is
            // missing/culled (avoids a hard binary pop at mirror_alpha == 0.5).
            let fb_mask = 1.0 - smoothstep(0.15, 0.65, mirror_alpha);
            if fb_mask > 0.01 && reflected_ray.y < 0.38 {
                let fallback = reflection_from_scene(input, reflected_ray, view_ray, size, 550.0, surface_distance, lost_variance);
                reflection_color = mix(reflection_color, fallback.rgb, fallback.a * fb_mask);
            }
        } else {
            // 4-Tier Hybrid Water Reflections:
            //   Tier 1: Fast screen-space raymarch resolves on-screen shorelines/cliffs with zero RT cost.
            //   Tier 2: Hardware ray query (VK_KHR_ray_query) fires ONLY on SSR misses/disocclusions
            //           (off-screen mountains, overhangs, waterfalls, entities, indoor pools).
            //   Tier 3: Decima 512x512 world-space heightfield catches far mountain silhouettes (96..450m).
            var hit_alpha = 0.0;
            if screen_space_reflections && fresnel > 0.025 {
                let scene_hit = reflection_from_scene(input, reflected_ray, view_ray, size, 550.0, surface_distance, lost_variance);
                hit_alpha = scene_hit.a;
                reflection_color = mix(reflection_color, scene_hit.rgb, hit_alpha);
            }
            if hit_alpha < 0.72 && fresnel > 0.03 && reflected_ray.y < 0.28 {
                let miss_w = 1.0 - hit_alpha;
                let hf_hit = water_heightfield_reflection(input.position + input.normal * 0.08, reflected_ray, 4.0, sky_access);
                if hf_hit.a > 0.01 {
                    reflection_color = mix(reflection_color, hf_hit.rgb, hf_hit.a * miss_w);
                }
            }
        }
    } else if submerged && water_reflections && input.normal.y < -0.65 {
        // Underside Total Internal Reflection (TIR) liquid-mirror: reflect the underwater
        // seabed, coral reefs, kelp, and light sources outside the 48.6-deg Snell's Window.
        let deep_base = scatter * mix(0.65, 1.15, clamp(sky_access, 0.0, 1.0));
        reflection_color = deep_base;
        if fresnel > 0.06 {
            let tir_hit = reflection_from_scene(input, reflected_ray, view_ray, size, 42.0, surface_distance, lost_variance);
            reflection_color = mix(deep_base, tir_hit.rgb + deep_base * 0.25, tir_hit.a * 0.92);
        }
    }

    // Dual-lobe Cox-Munk sun/moon glitter path weighted by optical Fresnel:
    // crisp dazzling sun core at grazing angles + soft wave-slope sheen.
    var glint = vec3(0.0);
    if lit_water > 0.0 {
        let half_vector = normalize(frame.light_direction - view_ray + vec3(0.0, 0.00001, 0.0));
        let normal_half = clamp(dot(normal, half_vector), 0.0, 1.0);
        let normal_light = clamp(dot(normal, frame.light_direction), 0.0, 1.0);
        let core_w = 0.024 + 0.035 * frame.rain;
        let broad_w = 0.095 + 0.07 * frame.rain;
        let dist_rough = 0.0005 * smoothstep(20.0, 95.0, surface_distance);
        let a2_core = core_w * core_w + lost_variance + dist_rough;
        let a2_broad = broad_w * broad_w + lost_variance + dist_rough;
        let d_core = 1.0 + (a2_core - 1.0) * normal_half * normal_half;
        let d_broad = 1.0 + (a2_broad - 1.0) * normal_half * normal_half;
        let dist_core = a2_core / max(0.5 * TAU * d_core * d_core, 0.0001);
        let dist_broad = a2_broad / max(0.5 * TAU * d_broad * d_broad, 0.0001);
        let glint_fresnel = mix(0.25, 1.0, optical_fresnel);
        let spec = min(0.022 * dist_core + 0.016 * dist_broad, 1.8);
        glint = sunlight * (spec * normal_light * lit_water * glint_fresnel);
    }

    var refracted = behind * transmission + scatter * (1.0 - transmission);
    if water_texture_mode != 0u {
        let texture_luma = dot(input.albedo.rgb, frame.luminance);
        let highlight = smoothstep(0.32, 0.70, texture_luma) * 0.22 + 0.025 * texture_luma;
        let exposed = 1.0 - smoothstep(0.13, 0.63, input.light.y);
        let texture_strength = select(exposed, 1.0, water_texture_mode == 2u);
        refracted += vec3(0.11, 0.20, 0.28) * (ambient + sunlight * (0.15 * lit_water)) * highlight * texture_strength;
        if water_texture_mode == 3u {
            refracted = mix(refracted, input.albedo.rgb * input.tint.rgb * (ambient + sunlight * (0.15 * lit_water)), 0.55);
        }
    }
    var color = mix(refracted, reflection_color, fresnel) + glint;
    if submerged {
        var window_color = behind;
        if input.normal.y < -0.65 {
            let openness = smoothstep(0.0, 0.70, input.light.y);
            if snells_window && fresnel < 0.995 {
                // Inside Snell's Window: wave-lensed sky, clouds, and sun/moon disk through
                // the 48.6-deg critical cone, plus physical RGB Cauchy chromatic dispersion
                // along the shimmering critical-angle rim (R=1.328, G=1.333, B=1.341).
                if background_depth <= 0.0 {
                    let air_g = refract(view_ray, normal, WATER_IOR);
                    if dot(air_g, air_g) > 0.5 {
                        let sky_ray_g = normalize(vec3(air_g.x, max(air_g.y, 0.015), air_g.z));
                        let env_g = sample_environment(sky_ray_g);
                        let cel_g = env_g.a * sky_celestial(sky_ray_g) * 0.85;
                        var lensed = (env_g.rgb + cel_g) * openness;

                        // Subtle prismatic dispersion concentrated along the critical-angle rim.
                        let rim_band = smoothstep(0.56, 0.67, incidence) * (1.0 - smoothstep(0.71, 0.84, incidence));
                        if rim_band > 0.02 {
                            let air_r = refract(view_ray, normal, 1.328);
                            let air_b = refract(view_ray, normal, 1.341);
                            let r_ch = select(lensed.r,
                                sample_environment(normalize(vec3(air_r.x, max(air_r.y, 0.015), air_r.z))).r * openness,
                                dot(air_r, air_r) > 0.5);
                            let b_ch = select(lensed.b,
                                sample_environment(normalize(vec3(air_b.x, max(air_b.y, 0.015), air_b.z))).b * openness,
                                dot(air_b, air_b) > 0.5);
                            lensed = mix(lensed, vec3(r_ch, lensed.g, b_ch), rim_band * 0.70);
                        }
                        window_color = lensed;
                    }
                }
                // Subtle sunlit sparkling rim right at the TIR boundary where refracted grazing rays compress.
                let rim_sparkle = smoothstep(0.62, 0.68, incidence) * (1.0 - smoothstep(0.68, 0.78, incidence));
                let sun_rim = pow(max(dot(view_ray, frame.light_direction), 0.0), 4.0);
                window_color += sunlight * (vec3(0.42, 0.68, 0.85) * rim_sparkle * (0.08 + 0.35 * sun_rim) * openness);
            } else if !snells_window {
                // Snell's Window disabled: keep the overhead water surface clearly visible
                // with wave-rippled sky lensing, sunlit wave-slope shimmer, and glassy sheen.
                let slope_tilt = normal - input.normal;
                if background_depth <= 0.0 {
                    let rippled_ray = normalize(view_ray + slope_tilt * 0.75);
                    let sky_ray = normalize(vec3(rippled_ray.x, max(rippled_ray.y, 0.015), rippled_ray.z));
                    let env = sample_environment(sky_ray);
                    let cel = env.a * sky_celestial(sky_ray) * 0.85;
                    window_color = (env.rgb + cel) * openness;
                }
                let ripple_shimmer = clamp(length(slope_tilt) * 3.8, 0.0, 1.0);
                let sun_align = 0.25 + 0.75 * pow(max(dot(view_ray, frame.light_direction), 0.0), 3.0);
                window_color += (ambient * 0.22 + sunlight * (0.28 * sun_align))
                    * vec3(0.48, 0.78, 0.92) * (0.08 + 0.38 * ripple_shimmer) * openness;
            }
        }
        // Per-channel Fresnel near the critical angle adds a delicate physical prism fringe
        // where blue hits Total Internal Reflection slightly before red.
        if snells_window && input.normal.y < -0.65 && incidence > 0.58 && incidence < 0.76 {
            let fr_r = water_fresnel(incidence, 1.328);
            let fr_b = water_fresnel(incidence, 1.341);
            let fresnel_rgb = vec3(fr_r, fresnel, fr_b);
            color = mix(window_color, reflection_color, fresnel_rgb);
        } else {
            let eff_fresnel = select(fresnel,
                max(fresnel, mix(0.16, 0.58, pow(1.0 - incidence, 2.2))),
                !snells_window && input.normal.y < -0.65);
            color = mix(window_color, reflection_color, eff_fresnel);
        }
        if !use_underwater_fog() {
            color = color * transmission + scatter * (1.0 - transmission);
        }
    } else if input.normal.y > 0.65 && background_depth > 0.0 && vertical_depth > 0.0 {
        // Animated shoreline capillary meniscus & gentle lapping micro-foam (near-field only
        // so distant shorelines never flash white).
        let foam_dist_fade = 1.0 - smoothstep(14.0, 32.0, surface_distance);
        if foam_dist_fade > 0.001 {
            let world_xz = (input.position + frame.camera_position).xz;
            let foam_noise = textureSampleLevel(wave_noise, wave_noise_sampler,
                world_xz * 0.22 + vec2(0.03, -0.02) * frame.time, 0.0).r;
            let lap = 0.06 * sin(frame.time * 2.1 + dot(world_xz, vec2(1.3, 0.9)));
            let shore_band = (1.0 - smoothstep(0.04, 0.42 + lap, vertical_depth)) * smoothstep(0.0, 0.035, vertical_depth);
            let foam_mask = smoothstep(0.42, 0.80, foam_noise) * shore_band * (1.0 - fresnel * 0.75) * foam_dist_fade;
            let foam_light = ambient * 0.45 + sunlight * (0.14 * lit_water);
            color = mix(color, foam_light * vec3(0.88, 0.94, 0.98), clamp(foam_mask * 0.35, 0.0, 0.35));
        }
        // Soften the razor-sharp 0-depth geometry seam right at the shoreline contact edge.
        let contact_blend = smoothstep(0.0, 0.075, vertical_depth);
        color = mix(behind, color, contact_blend);
    }
    return vec4(color, 1.0);
}


// Use the water caster and light-space thickness for wavelength
// attenuation and projected convergence on submerged opaque receivers.
fn water_receiver_light(position: vec3<f32>, surface_normal: vec3<f32>) -> vec3<f32> {
    if !pack_shadow_active() || frame.light_direction.y <= 0.07 {
        return vec3(1.0);
    }
    let p = pack_shadow_position(position);
    if any(p <= vec3(0.0)) || any(p >= vec3(1.0)) {
        return vec3(1.0);
    }
    var thickness: f32;
    var coverage = 1.0;
    if frame.has_shadow != 0u {
        // 1-fetch hardware shadow compare early-out: dry land blocks under trees/overhangs
        // have no translucent water closer to the sun, skipping all 4 shadow_color + depth loads.
        if pack_shadow_compare(p.xy, p.z - 0.00015, true) >= 0.995 {
            return vec3(1.0);
        }
        let size = pack_shadow_size();
        let fpix = p.xy * vec2<f32>(size) - vec2(0.5);
        let base = vec2<i32>(floor(fpix));
        let frac = fract(fpix);
        var depth_sum = 0.0;
        var water_w = 0.0;
        for (var tap = 0u; tap < 4u; tap += 1u) {
            let cell = vec2<i32>(i32(tap & 1u), i32(tap >> 1u));
            let q = clamp(base + cell, vec2(0), size - 1);
            let k = select(1.0 - frac, frac, cell == vec2(1));
            let w = k.x * k.y;
            if pack_shadow_color(q).a >= 0.995 {
                depth_sum += pack_shadow_depth(q, true) * w;
                water_w += w;
            }
        }
        if water_w <= 0.001 {
            return vec3(1.0);
        }
        let depth = depth_sum / water_w;
        coverage = water_w;
        thickness = clamp((p.z - depth) / max(pack_shadow_depth_per_block(), 1e-5), 0.0, 24.0);
    } else {
        coverage = 1.0 - pack_shadow_edge_fade(position, p);
        if coverage <= 0.0 {
            return vec3(1.0);
        }
        let water = ray_shadow_water(p);
        thickness = water.x;
        coverage *= water.y;
    }
    if thickness <= 0.045 {
        return vec3(1.0);
    }
    let attenuation = exp(-vec3(0.20, 0.08, 0.04) * thickness);
    // Above water the surface pass applies focusing after refraction.
    // Submerged views see the bed directly, so apply normal-weighted focusing here.
    if frame.eye_in != 1u || !water_caustics || thickness >= 18.0 {
        return mix(vec3(1.0), attenuation, coverage);
    }
    let sun_y = clamp(frame.light_direction.y, 0.08, 1.0);
    let refr_sun = normalize(vec3(
        frame.light_direction.x * 0.75,
        sqrt(max(0.08, 1.0 - 0.5625 * (1.0 - sun_y * sun_y))),
        frame.light_direction.z * 0.75,
    ));
    // Project caustics smoothly onto sloped reefs, coral, and sunlit vertical walls.
    let slope_facing = smoothstep(-0.18, 0.52, dot(surface_normal, refr_sun));
    let height = thickness * frame.light_direction.y;
    let deep_fade = 1.0 - smoothstep(12.0, 18.0, thickness);
    let focus = caustic_pattern(position, min(height, 12.5)) * (slope_facing * deep_fade);
    return mix(vec3(1.0), attenuation * max(vec3(0.18), vec3(1.0) + focus), coverage);
}
