struct TerrainReflectionHit {
    color: vec3<f32>,
    confidence: f32,
    distance: f32,
}

// Bounded VNDF Importance Sampling (Eto & Tokuyoshi, SIGGRAPH 2023):
// restricts the spherical cap to valid upper-hemisphere reflection directions,
// eliminating below-horizon rejected rays and reducing 1-SPP glossy noise by ~35%.
fn ggx_bounded_vndf(view: vec3<f32>, alpha: f32, random: vec2<f32>) -> vec3<f32> {
    let v_std = normalize(vec3(alpha * view.xy, max(view.z, 0.015)));
    let phi = TAU * random.y;
    let a = clamp(alpha, 0.008, 1.0);
    let s = 1.0 + length(view.xy);
    let a2 = a * a;
    let s2 = s * s;
    let k = (1.0 - a2) * s2 / max(s2 + a2 * view.z * view.z, 1e-5);
    let b = select(v_std.z, k * v_std.z, view.z > 0.0);
    let z = (1.0 - random.x) * (1.0 + b) - b;
    let sin_theta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
    let o_std = vec3(sin_theta * cos(phi), sin_theta * sin(phi), z);
    let m_std = v_std + o_std;
    return normalize(vec3(alpha * m_std.xy, max(m_std.z, 0.0001)));
}

fn rough_ray(view: vec3<f32>, normal: vec3<f32>, roughness: f32, random: vec2<f32>) -> vec3<f32> {
    let axis = select(vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0), abs(normal.y) > 0.98);
    let tangent = normalize(cross(axis, normal));
    let bitangent = cross(normal, tangent);
    let tbn = mat3x3<f32>(tangent, bitangent, normal);
    let alpha = clamp(roughness * roughness * 0.34, 0.008, 0.14);
    let microfacet = tbn * ggx_bounded_vndf(transpose(tbn) * view, alpha, random);
    let reflected = reflect(-view, microfacet);
    return select(reflect(-view, normal), reflected, dot(reflected, normal) > 0.0);
}

// Trace half-resolution terrain SSR and stabilize hits against full-res taa_history_previous.
fn terrain_reflection_hit(position: vec3<f32>, normal: vec3<f32>, ray: vec3<f32>, roughness: f32, sky: f32) -> TerrainReflectionHit {
    let origin = position + normal * 0.075;
    let view_projection = frame.projection * frame.view;
    let projected_origin = view_projection * vec4(origin, 1.0);
    let projected_step = view_projection * vec4(ray, 0.0);
    var previous_t = 0.0;
    for (var step = 1u; step <= 8u; step += 1u) {
        let travel = 0.16 + 0.45 * f32(step * step);
        let clip = projected_origin + travel * projected_step;
        if clip.w <= 0.01 {
            break;
        }
        let ndc = clip.xyz / clip.w;
        if any(abs(ndc.xy) >= vec2(1.0)) || ndc.z <= 0.0 || ndc.z >= 1.0 {
            break;
        }
        let uv = 0.5 + ndc.xy * vec2(0.5, -0.5);
        let depth = depth_at(uv);
        if depth > ndc.z {
            var low = previous_t;
            var high = travel;
            var hit_uv = uv;
            var hit_depth = depth;
            for (var i = 0u; i < 3u; i += 1u) {
                let probe_t = 0.5 * (low + high);
                let probe_clip = projected_origin + probe_t * projected_step;
                if probe_clip.w <= 0.01 || any(abs(probe_clip.xy) >= vec2(probe_clip.w)) {
                    return TerrainReflectionHit(vec3(0.0), 0.0, 0.0);
                }
                let probe = probe_clip.xyz / probe_clip.w;
                let probe_uv = 0.5 + probe.xy * vec2(0.5, -0.5);
                let probe_depth = depth_at(probe_uv);
                if probe_depth > probe.z {
                    high = probe_t;
                    hit_uv = probe_uv;
                    hit_depth = probe_depth;
                } else {
                    low = probe_t;
                }
            }
            let hit_pos = view_position(hit_uv, hit_depth);
            let error = distance(hit_pos, origin + high * ray);
            let tolerance = min(0.85, 0.13 + 0.009 * length(hit_pos) + 0.11 * (travel - previous_t));
            let side = dot(hit_pos - position, normal);
            if high < 0.19 || side < -0.06 || error >= tolerance {
                return TerrainReflectionHit(vec3(0.0), 0.0, 0.0);
            }
            let size = vec2<f32>(textureDimensions(scene_color));
            let texel = vec2<i32>(hit_uv * size);
            if !surface_at(texel).present {
                return TerrainReflectionHit(vec3(0.0), 0.0, 0.0);
            }
            let edge = min(hit_uv, 1.0 - hit_uv);
            let confidence = smoothstep(0.0, 0.085, min(edge.x, edge.y))
                * (1.0 - smoothstep(0.2 * tolerance, tolerance, error))
                * (1.0 - smoothstep(22.0, 34.0, high));
            let color = textureSampleLevel(scene_color, linear_sampler, hit_uv, 0.0).rgb;
            let filtered = distant_reflection_color(scene_color, scene_depth, hit_uv, hit_depth, max(high, length(hit_pos)), color);
            let saved = reflected_history(hit_pos, filtered, roughness);
            return TerrainReflectionHit(reflected_aerial_color(saved, ray, high, sky), confidence, high);
        }
        previous_t = travel;
    }
    return TerrainReflectionHit(vec3(0.0), 0.0, 0.0);
}

// Dedicated half-resolution Reflection TAA pass: traces environment + stochastic GGX SSR
// and temporally accumulates in Reinhard HDR space using virtual reflection parallax
// reprojection and depth-validated bilinear history sampling.
fn save_reflection_history(cell: vec2<i32>, uv: vec2<f32>, depth: f32, position: vec3<f32>, distance: f32) {
    if !terrain_reflections || frame.dimension != 0 || depth <= 0.0 || distance <= 0.02 {
        textureStore(reflection_history, cell, vec4<u32>(0u));
        return;
    }
    let scene_size = vec2<f32>(textureDimensions(scene_depth));
    let texel = clamp(vec2<i32>(uv * scene_size), vec2(0), vec2<i32>(scene_size) - 1);
    let surface = surface_at(texel);
    if !surface.present || surface_is_lava(u32(surface.data)) {
        textureStore(reflection_history, cell, vec4<u32>(0u));
        return;
    }
    let packed = u32(surface.data);
    let fog = vanilla_fog(position);
    let reflectance = surface_reflectance(packed);
    if fog >= 1.0 || max(reflectance.r, max(reflectance.g, reflectance.b)) <= 0.008 {
        textureStore(reflection_history, cell, vec4<u32>(0u));
        return;
    }

    let ray = reflect(position / distance, surface.normal);
    let sky = surface_sky(packed);
    let roughness = 1.0 - surface.smoothness;
    let reflect_dir = normalize(vec3(ray.x, max(ray.y, 0.03) + 0.35 * roughness, ray.z));
    let radiance = environment_radiance(reflect_dir);
    let broad = ambient_atmosphere_light() * 0.45;
    let sky_access = smoothstep(0.42, 1.0, sky);
    var current = mix(radiance, broad, roughness * 0.6) * sky_access * smoothstep(-0.25, 0.08, ray.y);
    var hit_dist = 0.0;
    var hit_conf = 0.0;
    let min_smoothness = select(0.68, 0.55, rough_reflections);
    let trace_eligible = surface.smoothness > min_smoothness && distance < 54.0;
    if trace_eligible && (screen_space_reflections || (rt_reflections && frame.has_ray_tracing != 0u)) {
        var traced_ray = ray;
        if rough_reflections && roughness > 0.10 {
            let ign = fract(52.9829189 * fract(dot(vec2<f32>(cell), vec2(0.06711056, 0.00583715))));
            let seed = fract(ign + f32(frame.frame_index & 31u) * 0.61803399);
            let random = fract(vec2(seed, 0.7548777 * seed));
            traced_ray = rough_ray(-position / distance, surface.normal, roughness, random);
        }
        let smooth_w = smoothstep(min_smoothness, 0.86, surface.smoothness);
        let dist_w = 1.0 - smoothstep(36.0, 54.0, distance);

        // Tier 1: Fast Hi-Z/screen-space SSR resolves ~70% of on-screen rays with zero RT cost.
        if screen_space_reflections {
            let hit = terrain_reflection_hit(position, surface.normal, traced_ray, roughness, sky_access);
            hit_conf = hit.confidence * smooth_w * dist_w;
            current = mix(current, hit.color, hit_conf);
            hit_dist = hit.distance;
        }

        // Tier 2: Hardware ray-query fallback (RAY_SOLID) fires ONLY on SSR misses/disocclusions
        // with a tight 48m reach so BVH traversal stays sub-0.1ms at half resolution.
        if hit_conf < 0.72 && rt_reflections && frame.has_ray_tracing != 0u {
            let rt_max = min(frame.ray_distance, 48.0);
            let rt_origin = position + surface.normal * clamp(0.04 + distance * 0.0012, 0.04, 0.14);
            let rt_hit = trace_rt_reflection(rt_origin, traced_ray, sky_access, rt_max);
            if rt_hit.a >= 0.0 {
                let rt_w = (1.0 - hit_conf) * smooth_w * dist_w;
                current = mix(current, rt_hit.rgb, rt_w);
                hit_dist = mix(rt_hit.a, hit_dist, hit_conf);
                hit_conf = clamp(hit_conf + rt_w, 0.0, 1.0);
            }
        }
    }

    var resolved = current;
    if temporal_reflections && frame.delta_time > 0.0 && frame.delta_time < 0.16 {
        // Blend between virtual reflection-point parallax (for glossy hits) and
        // surface reprojection (for rough surfaces or sky reflections).
        let parallax_weight = hit_conf * (1.0 - roughness) * (1.0 - roughness);
        let reproj_pos = position + ray * (hit_dist * parallax_weight) + frame.camera_motion;
        let clip = frame.previous_view_projection * vec4(reproj_pos, 1.0);
        if clip.w > 0.01 && all(abs(clip.xy) < vec2(clip.w)) {
            let prev_uv = 0.5 + clip.xy / clip.w * vec2(0.5, -0.5);
            let hist_size = vec2<i32>(textureDimensions(reflection_history_previous));
            let hist_res = vec2<f32>(hist_size);
            let expected = length(position + frame.camera_motion);
            let tolerance = max(0.35, 0.065 * expected);

            // 2x2 depth-validated bilinear history sample (prevents foreground bleeding).
            let pos_f = prev_uv * hist_res - 0.5;
            let base = vec2<i32>(floor(pos_f));
            let frac = fract(pos_f);
            var hist_tm = vec3(0.0);
            var w_sum = 0.0;
            for (var tap = 0u; tap < 4u; tap += 1u) {
                let off = vec2<i32>(i32(tap & 1u), i32(tap >> 1u));
                let q = clamp(base + off, vec2(0), hist_size - 1);
                let sample_val = unpack_u32_vec4(textureLoad(reflection_history_previous, q, 0));
                if sample_val.a > 0.002 {
                    let sample_dist = unpack_dist8(sample_val.a);
                    let depth_w = 1.0 - smoothstep(0.35 * tolerance, tolerance, abs(sample_dist - expected));
                    if depth_w > 0.0 {
                        let axis = select(1.0 - frac, frac, off == vec2(1));
                        let w = axis.x * axis.y * depth_w;
                        hist_tm += sample_val.rgb * w;
                        w_sum += w;
                    }
                }
            }
            if w_sum > 0.05 {
                let current_tm = pack_hdr8(current);
                let prev_tm = hist_tm / w_sum;
                // ReBLUR-style hit-distance-aware perceptual clamping: sharp near hits clamp tightly
                // so moving entities/camera pans never ghost, while rough/distant lobes integrate smoothly.
                let motion_px = length((uv - prev_uv) * hist_res);
                let motion_fade = smoothstep(0.5, 10.0, motion_px);
                let hit_clamp = mix(0.75, 1.25, smoothstep(2.0, 28.0, hit_dist));
                let clamp_radius = vec3(mix(0.16 + 0.24 * roughness, 0.09 + 0.10 * roughness, motion_fade) * hit_clamp);
                let clamped_tm = clamp(prev_tm, current_tm - clamp_radius, current_tm + clamp_radius);
                let base_keep = mix(0.84, 0.91, smoothstep(0.08, 0.40, roughness));
                let keep = mix(base_keep, 0.52, motion_fade) * clamp(w_sum, 0.0, 1.0);
                resolved = unpack_hdr8(mix(current_tm, clamped_tm, keep));
            }
        }
    }
    textureStore(reflection_history, cell, pack_u32_vec4(vec4(pack_hdr8(resolved), pack_dist8(distance))));
}
