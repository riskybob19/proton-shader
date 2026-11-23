const GTAO_SLICES: u32 = 2u;
const GTAO_STEPS: u32 = 2u;
const GTAO_RADIUS: f32 = 2.0;

fn gtao_sky_luma(normal: vec3<f32>) -> f32 {
    let plane = normal.xz / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    let oct = 0.5 + 0.5 * select(plane, octahedral_fold(plane), normal.y < 0.0);
    let uv = skylight_atlas_uv(oct, vec2<f32>(textureDimensions(cloud_aux)));
    let irradiance = textureSampleLevel(cloud_aux, scattering_atlas_sampler, uv, 0.0).rgb;
    return dot(irradiance, frame.luminance);
}

// EGSR 2023 Visibility Bitmask AO (Therrien & Levesque) with cosine-weighted
// 32-sector hemisphere encoding and finite voxel thickness so light passes
// naturally behind thin grass blades, flowers, fences, and tree trunks.
fn gtao_bitmask_dir(center: vec3<f32>, toward_eye: vec3<f32>, tangent: vec3<f32>,
    normal: vec3<f32>, cos_gamma: f32, sin_gamma: f32, uv: vec2<f32>, step_uv: vec2<f32>,
    size: vec2<f32>, radius: f32, thickness: f32, jitter: f32, steps: u32) -> u32 {
    let step_pixels = length(step_uv * size);
    if step_pixels < 0.18 {
        return 0u;
    }
    var bitmask = 0u;
    let f_steps = f32(steps);
    let inv_steps = 1.0 / f_steps;
    let base_off = 1.0 / step_pixels;
    let radius2 = radius * radius;
    let inv_size = 1.0 / size;
    for (var tap = 0u; tap < steps; tap += 1u) {
        // Sample densely near the receiver so narrow grass blades reliably occlude the ground.
        let t = (f32(tap) + jitter) * inv_steps;
        let sample_uv = uv + step_uv * (t * t * f_steps + base_off);
        if any(sample_uv <= vec2(0.0)) || any(sample_uv >= vec2(1.0)) {
            break;
        }
        let cell = vec2<i32>(sample_uv * size);
        let depth = textureLoad(scene_depth, cell, 0);
        if depth > 0.0 {
            let sample_center = (vec2<f32>(cell) + 0.5) * inv_size;
            let separation = view_position(sample_center, depth) - center;
            let squared = dot(separation, separation);
            let height = dot(normal, separation);
            // Reject samples outside the radius and below the surface bias.
            if squared > 0.0001 && squared < radius2
                && height > 0.012 && height * height > 0.000225 * squared {
                let dist = sqrt(squared);
                let falloff = smoothstep(0.68 * radius, radius, dist);
                // Exact algebraic equivalence to sin(clamp(atan2(s_t, s_v) - gamma, -pi/2, pi/2))
                // without any atan2() or sin() calls. Since toward_eye is orthogonal to tangent,
                // back_sep = separation - toward_eye * thickness only shifts s_v by -thickness.
                let s_t = dot(separation, tangent);
                let s_v = dot(separation, toward_eye);
                let sn_f = s_v * cos_gamma + s_t * sin_gamma;
                let sp_f = s_t * cos_gamma - s_v * sin_gamma;
                let sin_f = select(select(-1.0, 1.0, sp_f >= 0.0),
                    sp_f * inverseSqrt(max(sn_f * sn_f + sp_f * sp_f, 1e-12)), sn_f > 0.0);
                let bv = s_v - thickness;
                let sn_b = bv * cos_gamma + s_t * sin_gamma;
                let sp_b = s_t * cos_gamma - bv * sin_gamma;
                let sin_b = select(select(-1.0, 1.0, sp_b >= 0.0),
                    sp_b * inverseSqrt(max(sn_b * sn_b + sp_b * sp_b, 1e-12)), sn_b > 0.0);
                let u_f = 0.5 + 0.5 * sin_f;
                let u_b = 0.5 + 0.5 * sin_b;
                let lo = min(u_f, u_b);
                let hi = max(u_f, u_b);
                let span = (hi - lo) * (1.0 - falloff);
                if span > 0.002 {
                    let mid = 0.5 * (lo + hi);
                    let a = clamp(mid - 0.5 * span, 0.0, 1.0);
                    let b = clamp(mid + 0.5 * span, 0.0, 1.0);
                    let start_bit = u32(clamp(floor(a * 32.0), 0.0, 31.0));
                    let bit_count = u32(clamp(ceil((b - a) * 32.0), 0.0, 32.0 - f32(start_bit)));
                    if bit_count > 0u {
                        let mask = select(((1u << bit_count) - 1u) << start_bit, 0xffffffffu, bit_count >= 32u);
                        bitmask |= mask;
                    }
                }
            }
        }
    }
    return bitmask;
}

// Fast scalar reversed-Z to positive linear view-space depth (sign-proof).
fn ssrt_linear_z(depth: f32) -> f32 {
    let p = frame.projection_inverse;
    let z = p[2].z * depth + p[3].z;
    let w = p[2].w * depth + p[3].w;
    return abs(z) / max(abs(w), 1e-6);
}

// Bend Studio / UE5-style near-field Screen-Space Contact Shadows (SSCS) with
// finite surface-thickness slabs: catches fine grass blades, fences, rails, stairs,
// and block steps inside 0..28 blocks where the 1024 shadow map cannot resolve them.
fn near_contact_shadow(position: vec3<f32>, normal: vec3<f32>, distance: f32,
    world_to_clip: mat4x4<f32>, size: vec2<f32>, jitter: f32) -> f32 {
    let light = frame.light_direction;
    if dot(normal, light) <= 0.02 || light.y <= 0.025 || distance >= 22.0 {
        return 0.0;
    }
    let dist_fade = 1.0 - smoothstep(14.0, 22.0, distance);
    let max_world_len = mix(0.65, 1.35, clamp(distance / 22.0, 0.0, 1.0));
    let origin_world = position + normal * clamp(0.028 + distance * 0.0022, 0.028, 0.095);
    let clip0 = world_to_clip * vec4(origin_world, 1.0);
    if clip0.w <= 0.01 {
        return 0.0;
    }
    var clip1 = world_to_clip * vec4(origin_world + light * max_world_len, 1.0);
    if clip1.w < 0.05 {
        let clip_t = (clip0.w - 0.05) / max(clip0.w - clip1.w, 1e-5);
        clip1 = mix(clip0, clip1, clamp(clip_t, 0.0, 1.0));
    }
    let ndc0 = clip0.xyz / clip0.w;
    let ndc1 = clip1.xyz / clip1.w;
    let origin_screen = vec3(0.5 + ndc0.xy * vec2(0.5, -0.5), ndc0.z);
    let end_screen = vec3(0.5 + ndc1.xy * vec2(0.5, -0.5), ndc1.z);
    var delta = end_screen - origin_screen;
    let screen_len = length(delta.xy);
    if screen_len <= 1e-5 {
        return 0.0;
    }
    // Cap screen-space travel so grazing rays stay cache-coherent and local.
    if screen_len > 0.065 {
        delta *= 0.065 / screen_len;
    }
    let min_bias = clamp(0.032 + distance * 0.0032, 0.032, 0.14);
    let slab_thickness = clamp(0.18 + distance * 0.016, 0.18, 0.58);
    var blocked = 0.0;
    const STEPS: u32 = 4u;
    for (var i = 0u; i < STEPS; i += 1u) {
        let t = (f32(i) + 0.35 + 0.65 * jitter) / f32(STEPS);
        let probe = origin_screen + delta * t;
        if any(probe.xy <= vec2(0.0)) || any(probe.xy >= vec2(1.0)) || probe.z <= 0.0 || probe.z >= 1.0 {
            break;
        }
        let cell = clamp(vec2<i32>(probe.xy * size), vec2(0), vec2<i32>(size) - 1);
        let sample_depth = textureLoad(scene_depth, cell, 0);
        if sample_depth > probe.z {
            let dz = ssrt_linear_z(probe.z) - ssrt_linear_z(sample_depth);
            if dz > min_bias && dz < slab_thickness {
                let edge = min(probe.xy, 1.0 - probe.xy);
                let edge_fade = smoothstep(0.0, 0.03, min(edge.x, edge.y));
                let slab_fade = smoothstep(min_bias, min_bias * 2.0, dz)
                    * (1.0 - smoothstep(0.55 * slab_thickness, slab_thickness, dz));
                let ray_fade = 1.0 - smoothstep(0.65, 1.0, t);
                blocked = max(blocked, edge_fade * slab_fade * ray_fade);
                if blocked >= 0.95 {
                    break;
                }
            }
        }
    }
    return blocked * dist_fade * 0.72;
}

// Unified cascade handoff weight for Tier 2 (Hi-Z SSRT) and Tier 3 (Decima Heightfield).
// Keeps 0..96m 100% driven by the artifact-free 2048x2048 3D shadow map, then ramps in
// smoothly across the 114..140m shadow-map fade zone so distant shadows take over seamlessly.
fn distant_shadow_weight(position: vec3<f32>, distance: f32) -> f32 {
    if !pack_shadow_active() {
        return 1.0;
    }
    if distance < SHADOW_ASSIST_START {
        return 0.0;
    }
    let assist = smoothstep(SHADOW_ASSIST_START, SHADOW_FADE_START, distance) * 0.25;
    let p = pack_shadow_position(position);
    var edge_fade = 1.0;
    if p.z > 0.0 && p.z < 1.0 {
        edge_fade = pack_shadow_edge_fade(position, p);
    }
    return max(assist, smoothstep(0.02, 0.35, edge_fade));
}

// Tier 2: Hi-Z accelerated screen-space raytraced distant shadows (96..512m).
// Uses local thickness bounds (6..22m) and strict foreground rejection (sample_z > ray_z * 0.82)
// so foreground trees/cliffs across a ravine NEVER cast fake screen-space occlusion halos.
fn half_distant_shadow(position: vec3<f32>, normal: vec3<f32>, distance: f32,
    uv: vec2<f32>, center_depth: f32, world_to_clip: mat4x4<f32>, size: vec2<f32>, jitter: f32, fade: f32) -> f32 {
    let light = frame.light_direction;
    if fade <= 0.001 || dot(normal, light) <= 0.0 || light.y <= 0.025 {
        return 0.0;
    }

    // Small temporal cone jitter creates soft distant shadow penumbras under TAA.
    let j2 = fract(jitter * 7.2389 + 0.37);
    let angle = jitter * TAU;
    let cone_r = 0.012 * sqrt(j2);
    let light_t = normalize(cross(select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), abs(light.y) > 0.95), light));
    let light_b = cross(light, light_t);
    let ray_dir = normalize(light + (light_t * cos(angle) + light_b * sin(angle)) * cone_r);

    // Strictly outward normal-biased world origin so vertical cliff walls never self-intersect.
    let origin_world = position + normal * clamp(distance * 0.0018, 0.14, 0.55);
    let clip0 = world_to_clip * vec4(origin_world, 1.0);
    if clip0.w <= 0.01 {
        return 0.0;
    }
    var clip1 = world_to_clip * vec4(origin_world + ray_dir * max(1.0, distance * 0.025), 1.0);
    if clip1.w < 0.05 {
        let clip_t = (clip0.w - 0.05) / max(clip0.w - clip1.w, 1e-5);
        clip1 = mix(clip0, clip1, clamp(clip_t, 0.0, 1.0));
    }
    let ndc0 = clip0.xyz / clip0.w;
    let ndc1 = clip1.xyz / clip1.w;
    let ray_origin_screen = vec3(0.5 + ndc0.xy * vec2(0.5, -0.5), ndc0.z);
    let p1_screen = vec3(0.5 + ndc1.xy * vec2(0.5, -0.5), ndc1.z);
    let delta_screen = p1_screen - ray_origin_screen;
    let delta_len = length(delta_screen);
    if delta_len <= 1e-6 {
        return 0.0;
    }
    let ray_dir_screen = delta_screen / delta_len;

    let inv_dir = 1.0 / max(abs(ray_dir_screen), vec3(1e-6));
    let bound = select(vec3(0.0), vec3(1.0), ray_dir_screen >= vec3(0.0));
    let t_box = abs(bound - ray_origin_screen) * inv_dir;
    var ray_length = min(t_box.x, min(t_box.y, t_box.z));
    ray_length = min(ray_length, max(0.32, exp(-max(distance * 0.012 - 1.0, 0.0))));

    const STEP_COUNT: u32 = 16u;
    const STEP_RATIO: f32 = 1.28;
    let initial_scale = (STEP_RATIO - 1.0) / (pow(STEP_RATIO, f32(STEP_COUNT)) - 1.0);
    var step_len = ray_length * initial_scale;
    let pixel_step = length(1.0 / size) * 0.85;
    var ray_pos = ray_origin_screen + ray_dir_screen * pixel_step;

    let min_bias = clamp(distance * 0.0016, 0.16, 0.55);
    // Keep screen-space thickness local (6..24m for trees and cliff steps on the same hill);
    // broad cross-valley mountain shadows are handled in 3D world space by cached_heightfield_shadow.
    let z_tolerance = clamp(distance * 0.09, 6.0, 24.0);
    let hiz_size = vec2<f32>(textureDimensions(depth_hiz_4));
    var blocked = 0.0;

    for (var i = 0u; i < STEP_COUNT; i += 1u) {
        let ray_step = ray_dir_screen * step_len;
        let dithered = ray_pos + ray_step * jitter;
        ray_pos += ray_step;
        step_len *= STEP_RATIO;
        if any(dithered.xy <= vec2(0.0)) || any(dithered.xy >= vec2(1.0)) || dithered.z <= 0.0 || dithered.z >= 1.0 {
            break;
        }
        var sample_z = 0.0;
        if i < 4u {
            let cell = clamp(vec2<i32>(dithered.xy * size), vec2(0), vec2<i32>(size) - 1);
            let sd = textureLoad(scene_depth, cell, 0);
            if sd > 0.0 {
                sample_z = ssrt_linear_z(sd);
            }
        } else {
            let cell = clamp(vec2<i32>(dithered.xy * hiz_size), vec2(0), vec2<i32>(hiz_size) - 1);
            let hiz = textureLoad(depth_hiz_4, cell, 0);
            sample_z = select(hiz.g, hiz.r, i < 10u);
        }
        if sample_z > 0.0 {
            let ray_z = ssrt_linear_z(dithered.z);
            let dz = ray_z - sample_z;
            // Strict foreground rejection (sample_z > ray_z * 0.82) prevents nearby trees
            // or foreground ravine edges from casting fake occlusion halos onto background hills.
            if dz > min_bias && dz < z_tolerance && sample_z > ray_z * 0.82 {
                let edge = min(dithered.xy, 1.0 - dithered.xy);
                let edge_fade = smoothstep(0.0, 0.03, min(edge.x, edge.y));
                let bias_fade = smoothstep(min_bias, min_bias * 1.40, dz);
                let thickness_fade = 1.0 - smoothstep(0.72 * z_tolerance, z_tolerance, dz);
                blocked = max(blocked, edge_fade * bias_fade * thickness_fade);
                if blocked >= 0.98 {
                    break;
                }
            }
        }
    }
    return smoothstep(0.0, 0.62, blocked) * fade;
}

// Tier 3: Decima-inspired cached world-space terrain heightfield shadow march (96..512m):
// marches 16 jittered steps along frame.light_direction in world space against the
// scrolled 512x512 terrain_heightfield cache, casting true 360-degree off-screen
// mountain and hill shadows across valleys with zero screen-space halo artifacts.
fn cached_heightfield_shadow(position: vec3<f32>, normal: vec3<f32>, distance: f32, jitter: f32, fade: f32) -> f32 {
    if !cached_distant_shadows || fade <= 0.001 {
        return 0.0;
    }
    let light = frame.light_direction;
    if dot(normal, light) <= 0.0 || light.y <= 0.03 {
        return 0.0;
    }
    let anchor = hf_anchor(frame.camera_position.xz);
    let world_origin = frame.camera_position + position + normal * 0.55;
    let max_march = min(290.0, max(55.0, (320.0 - world_origin.y) / max(light.y, 0.08)));
    const HF_STEPS: u32 = 16u;
    let inv_steps = 1.0 / f32(HF_STEPS);
    var blocked = 0.0;
    for (var i = 0u; i < HF_STEPS; i += 1u) {
        let t = (f32(i) + 0.25 + 0.75 * jitter) * inv_steps;
        let d = 2.2 + max_march * (0.14 * t + 0.86 * t * t * t);
        let sample_w = world_origin + light * d;
        let uv = hf_world_to_uv(sample_w.xz, anchor);
        if any(uv <= vec2(0.01)) || any(uv >= vec2(0.99)) {
            break;
        }
        let hf = textureSampleLevel(terrain_heightfield, linear_sampler, uv, 0.0);
        if hf.g > 0.35 {
            let bias = 0.65 + 0.005 * d;
            let excess = hf.r - sample_w.y;
            if excess > bias {
                let penumbra = smoothstep(bias, bias + 0.75 + 0.007 * d, excess);
                let conf = smoothstep(0.35, 0.70, hf.g);
                blocked = max(blocked, penumbra * conf);
                if blocked >= 0.98 {
                    break;
                }
            }
        }
    }
    return smoothstep(0.0, 0.62, blocked) * fade;
}

// 4-tap local valley/crevice ambient occlusion over linear-Z depth_hiz_4 (48..320m):
// strictly rejects depth discontinuities (abs(sz - distance) < 10m) so foreground trees
// and ravine edges NEVER cast floating dark occlusion halos onto background cliffs.
fn macro_valley_ao(center: vec3<f32>, normal: vec3<f32>, distance: f32, uv: vec2<f32>) -> f32 {
    let hiz_size = vec2<f32>(textureDimensions(depth_hiz_4));
    let step_uv = clamp(vec2(5.5) / max(distance, 36.0), vec2(0.007), vec2(0.028));
    let max_dz = clamp(distance * 0.075, 3.5, 11.0);
    var occ = 0.0;
    for (var i = 0u; i < 4u; i += 1u) {
        let off = select(
            vec2(0.0, select(-step_uv.y, step_uv.y, i == 2u)),
            vec2(select(-step_uv.x, step_uv.x, i == 0u), 0.0),
            i < 2u,
        );
        let su = uv + off;
        if any(su <= vec2(0.01)) || any(su >= vec2(0.99)) {
            continue;
        }
        let sc = clamp(vec2<i32>(su * hiz_size), vec2(0), vec2<i32>(hiz_size) - 1);
        let sz = textureLoad(depth_hiz_4, sc, 0).b;
        if sz <= 0.0 || abs(sz - distance) > max_dz {
            continue;
        }
        // Reconstruct world-space neighbor position from linear view-space Z.
        let ray_far = view_position(su, 0.001);
        let ray_z = max(ssrt_linear_z(0.001), 1e-4);
        let nbr = ray_far * (sz / ray_z);
        let diff = nbr - center;
        let len_sq = dot(diff, diff);
        if len_sq > 1.0 && len_sq < 196.0 {
            let cos_h = dot(diff * inverseSqrt(len_sq), normal);
            if cos_h > 0.12 {
                let falloff = 1.0 - smoothstep(81.0, 196.0, len_sq);
                occ += smoothstep(0.12, 0.68, cos_h) * falloff;
            }
        }
    }
    let fade = smoothstep(44.0, 68.0, distance) * (1.0 - smoothstep(260.0, 360.0, distance));
    return 1.0 - clamp(occ * 0.12, 0.0, 0.24) * fade;
}

fn post(input: PostInput) -> vec4<f32> {
    let empty = vec4(1.0, 1.0, 0.0, 1.0);
    if frame.dimension != 0 || input.depth <= 0.0 || (contact_ao <= 0.0 && !shadows) {
        return empty;
    }
    let size = vec2<f32>(textureDimensions(scene_depth));
    let cell = clamp(vec2<i32>(input.uv * size), vec2(0), vec2<i32>(size) - 1);
    let surface = surface_at(cell);
    let depth = textureLoad(scene_depth, cell, 0);
    if !surface.present || surface_is_lava(u32(surface.data)) || depth <= 0.0 {
        return empty;
    }

    let uv = (vec2<f32>(cell) + 0.5) / size;
    let center = view_position(uv, depth);
    let distance = length(center);
    if distance < 0.05 {
        return empty;
    }

    let world_to_clip = frame.projection * frame.view;
    let spatial_jitter = fract(dot(floor(input.pixel), vec2(0.06711056, 0.00583715)) * 52.9829189);
    let jitter = fract(spatial_jitter + f32(frame.frame_index & 15u) * 0.61803399);

    var bounced = 1.0;
    var directional = 1.0;
    if contact_ao > 0.0 && distance < 56.0 {
        let toward_eye = -center / distance;
        let camera_right = normalize(frame.view_inverse[0].xyz);
        let camera_up = normalize(frame.view_inverse[1].xyz);
        let right = normalize(cross(camera_up, toward_eye));
        let up = cross(toward_eye, right);
        let close_detail = distance < 14.0;
        let slices = GTAO_SLICES;
        let steps = select(GTAO_STEPS, 3u, close_detail);
        let radius = clamp(distance * 0.22, 0.55, GTAO_RADIUS);
        let center_clip = world_to_clip * vec4(center, 1.0);
        var sum = 0.0;
        var open_sum = 0.0;
        var bent = vec3(0.0);
        for (var slice = 0u; slice < slices; slice += 1u) {
            let angle = (f32(slice) + jitter) * PHOTON_PI / f32(slices);
            let tangent = right * cos(angle) + up * sin(angle);
            let axis = cross(toward_eye, tangent);
            let projected_normal = surface.normal - axis * dot(surface.normal, axis);
            let normal_length = length(projected_normal);
            if normal_length <= 0.001 {
                continue;
            }
            let inv_norm = 1.0 / normal_length;
            let raw_cos = dot(projected_normal, toward_eye) * inv_norm;
            let raw_sin = dot(projected_normal, tangent) * inv_norm;
            let cos_gamma = max(raw_cos, 0.0);
            let sin_gamma = select(select(-1.0, 1.0, raw_sin >= 0.0), raw_sin, raw_cos >= 0.0);
            let offset = world_to_clip * vec4(tangent * (radius / f32(steps)), 0.0);
            let probe = center_clip + offset;
            let step_uv = (probe.xy / max(probe.w, 0.02) - center_clip.xy / center_clip.w) * vec2(0.5, -0.5);
            let start = 0.25 + 0.65 * fract(jitter + f32(slice) * 0.618034);
            let thickness = mix(0.38, 0.68, clamp(distance / 24.0, 0.0, 1.0));
            let bitmask = gtao_bitmask_dir(center, toward_eye, tangent, surface.normal, cos_gamma, sin_gamma, uv, -step_uv, size, radius, thickness, start, steps)
                | gtao_bitmask_dir(center, toward_eye, tangent, surface.normal, cos_gamma, sin_gamma, uv, step_uv, size, radius, thickness, start, steps);
            let left_open = 16.0 - f32(countOneBits(bitmask & 0x0000ffffu));
            let right_open = 16.0 - f32(countOneBits(bitmask >> 16u));
            let slice_vis = (left_open + right_open) * (1.0 / 32.0);
            let weighted = slice_vis * normal_length;
            sum += weighted;
            open_sum += normal_length;
            let delta = ((right_open - left_open) * (1.0 / 32.0)) * (0.5 * PHOTON_PI);
            let cd = cos(delta);
            let sd = sin(delta);
            let cos_med = cos_gamma * cd - sin_gamma * sd;
            let sin_med = sin_gamma * cd + cos_gamma * sd;
            bent += weighted * (toward_eye * cos_med + tangent * sin_med);
        }
        let visibility = clamp(sum / max(open_sum, 1e-4), 0.0, 1.0);
        let dist_fade = 1.0 - smoothstep(40.0, 56.0, distance);
        bounced = mix(1.0, pow(visibility, 1.35), dist_fade);
        if directional_skylight && bounced < 0.96 {
            let direction = normalize(bent + 0.25 * surface.normal + 0.001 * camera_right);
            let bent_normal = normalize(mix(surface.normal, direction, 1.0 - bounced));
            let unoccluded = gtao_sky_luma(surface.normal);
            let occluded = gtao_sky_luma(bent_normal);
            directional = clamp((occluded + 0.02) / (unoccluded + 0.02), 0.65, 1.0);
        }
    }
    if contact_ao > 0.0 && distance >= 38.0 && distance < 320.0 {
        bounced *= macro_valley_ao(center, surface.normal, distance, uv);
    }

    var blocked = 0.0;
    let packed = u32(surface.data);
    let direct_fraction = surface_direct(packed);
    if shadows && direct_fraction > 0.0 {
        if pack_shadow_active() && distance < 24.0 {
            blocked = near_contact_shadow(center, surface.normal, distance, world_to_clip, size, jitter);
        } else {
            let fade = distant_shadow_weight(center, distance);
            if fade > 0.001 {
                let ssrt_b = half_distant_shadow(center, surface.normal, distance, uv, depth, world_to_clip, size, jitter, fade);
                let hf_b = cached_heightfield_shadow(center, surface.normal, distance, jitter, fade);
                blocked = max(ssrt_b, hf_b);
            }
        }
    }

    return vec4(bounced, directional, blocked, 1.0);
}
