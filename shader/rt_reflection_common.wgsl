// Shared 4-tier hybrid reflection helpers:
//   Tier 2: Horizon API 0.2 hardware VK_KHR_ray_query (trace_layers on RAY_SOLID | RAY_ENTITY)
//   Tier 3: Decima 512x512 world-space terrain heightfield far-horizon fallback (96..450m)
// Included by both [material] (water.wgsl) and half-res Reflection TAA (fog_compute.wgsl).

fn shade_rt_reflection_hit(hit: RayHit, ray: vec3<f32>, sky_access: f32) -> vec3<f32> {
    var base = hit.albedo.rgb * (hit.tint.rgb * hit.shade * hit.ao);
    if (hit.tags & 1u) != 0u {
        // Match canopy_albedo() foliage richness on reflected trees.
        let lum = max(dot(base, frame.luminance), 1e-4);
        base = max(mix(vec3(lum), base, 1.12), vec3(0.0));
    }

    let sky = hit.light.y;
    let sky2 = sky * sky;
    let no_l = max(dot(hit.normal, frame.light_direction), 0.0);

    // Fast 1-tap shadow-map visibility inside the shadow frustum, skylight fallback outside.
    var sun_vis = smoothstep(14.1 / 15.0, 14.9 / 15.0, sky);
    if pack_shadow_active() && no_l > 0.0 {
        let sp = pack_shadow_position(hit.position + hit.normal * 0.14);
        if all(sp.xy >= vec2(0.0)) && all(sp.xy <= vec2(1.0)) && sp.z > 0.0 && sp.z < 1.0 {
            let fade = pack_shadow_edge_fade(hit.position, sp);
            if fade < 1.0 {
                let cmp = pack_shadow_compare(sp.xy, sp.z - 0.00025, false);
                sun_vis = mix(cmp * smoothstep(0.0, 2.0 / 15.0, sky), sun_vis, fade);
            }
        }
    }

    let sun_lit = direct_atmosphere_light() * (cloud_shadow(hit.position) * sun_vis * no_l);
    let hemi = 0.72 + 0.28 * max(hit.normal.y, 0.0);
    let sky_lit = photon_ambient_irradiance() * (sky2 * hemi);
    let b = hit.light.x;
    let b2 = b * b;
    let block_lit = vec3(1.0, 0.54, 0.34) * (4.2 * b2 * b2 + 0.55 * b2);
    let cave = 0.14 * (1.0 - sky2) * (1.0 - 0.7 * frame.darkness);
    let diffuse = (base / PHOTON_PI) * (sun_lit + sky_lit + block_lit + vec3(cave + 1.4 * frame.night_vision));
    let emissive = base * (2.6 * hit.emission);
    let lit = diffuse + emissive;
    let aerial = reflected_aerial_color(lit, ray, hit.distance, sky_access);
    // Soft firefly knee suppresses extreme emissive/HDR spikes before Fresnel blending.
    let peak = max(aerial.r, max(aerial.g, aerial.b));
    return aerial * (1.0 / (1.0 + 0.06 * max(peak - 1.6, 0.0)));
}

// Trace 1 hardware ray query against Horizon's static terrain TLAS (RAY_SOLID only).
// Avoiding RAY_ENTITY eliminates Horizon's per-frame dynamic entity BLAS/TLAS rebuild
// and removes ~250 lines of entity-atlas decoding from the full-res [material] shader.
fn trace_rt_reflection(origin: vec3<f32>, ray: vec3<f32>, sky_access: f32, max_dist: f32) -> vec4<f32> {
    if !rt_reflections || frame.has_ray_tracing == 0u || max_dist <= 0.5 {
        return vec4(0.0, 0.0, 0.0, -1.0);
    }
    let hit = trace(origin, ray, max_dist);
    if hit.distance < 0.0 {
        return vec4(0.0, 0.0, 0.0, -1.0);
    }
    return vec4(shade_rt_reflection_hit(hit, ray, sky_access), hit.distance);
}

// Tier 3 Decima World-Space Heightfield Far-Horizon Reflection Fallback (4..460m):
// Marches 6 fast steps against terrain_heightfield_previous only on near-horizon rays
// (-0.03 < ray.y < 0.34) so open-sky water pixels skip heightfield traversal completely.
fn heightfield_reflection_fallback(origin: vec3<f32>, ray: vec3<f32>, start_dist: f32, sky_access: f32) -> vec4<f32> {
    if ray.y < -0.03 || ray.y > 0.34 {
        return vec4(0.0);
    }
    let anchor = hf_anchor(frame.camera_position.xz);
    let world_origin = frame.camera_position + origin;
    let max_d = min(460.0, max(start_dist + 88.0, (290.0 - world_origin.y) / max(ray.y, 0.05)));
    let span = max(max_d - start_dist, 1.0);
    const STEPS: u32 = 6u;
    let hf_size = vec2<i32>(textureDimensions(terrain_heightfield_previous));
    for (var i = 0u; i < STEPS; i += 1u) {
        let t = (f32(i) + 0.5) / f32(STEPS);
        let d = start_dist + span * (0.22 * t + 0.78 * t * t);
        let p_w = world_origin + ray * d;
        let uv = hf_world_to_uv(p_w.xz, anchor);
        if any(uv <= vec2(0.01)) || any(uv >= vec2(0.99)) {
            break;
        }
        let hf_cell = clamp(vec2<i32>(uv * vec2<f32>(hf_size)), vec2(0), hf_size - 1);
        let hf = textureLoad(terrain_heightfield_previous, hf_cell, 0);
        if hf.g > 0.35 && hf.r > p_w.y + 1.2 {
            // Physically shaded distant mountain silhouette (snow-capped above Y=120, forested below).
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
