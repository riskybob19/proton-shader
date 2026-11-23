override colored_shafts: bool = false;

fn fog_shadow(position: vec3<f32>) -> vec3<f32> {
    // RDR2 unified atmosphere: cast live volumetric cloud shadows directly through
    // the air column across the entire view distance (creating sweeping cloud god rays
    // and shadowed mountain valleys even beyond the 128-block terrain shadow map).
    let cloud_vis = select(1.0, cloud_shadow_fast(position), volumetric_clouds && cloud_cover > 0.0);
    if !pack_shadow_active() {
        return vec3(cloud_vis);
    }
    let p = pack_shadow_position(position);
    if any(p.xy < vec2(0.0)) || any(p.xy > vec2(1.0)) || p.z <= 0.0 || p.z >= 1.0 {
        return vec3(cloud_vis);
    }
    let fade = pack_shadow_edge_fade(position, p);
    if fade >= 1.0 {
        return vec3(cloud_vis);
    }
    var light: vec3<f32>;
    if colored_shafts && frame.has_shadow == 0u {
        light = ray_shadow_filtered(p.xy, ray_shadow_volume(p.xy, p.z), false);
    } else {
        let solid = pack_shadow_compare(p.xy, p.z, false);
        light = vec3(solid);
        if colored_shafts {
            light = shadow_transmission(p.xy, p.z, solid);
        }
    }
    return mix(light, vec3(1.0), fade) * cloud_vis;
}

fn fog_position(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    return view_position(uv, select(depth, 1.0, depth <= 0.0));
}

fn fog_distance(ray_length: f32, depth: f32) -> f32 {
    return select(ray_length, frame.far, depth <= 0.0);
}

// Sub-cloud volumetric crepuscular god rays rendered in front of terrain inside the
// half-resolution fog pass. Uses 8 cached 1-fetch bilinear lookups into cloud_aux so
// sunlit cloud breaks cut visibly across forests and valleys at negligible GPU cost.
fn cloud_terrain_sunbeams(ray: vec3<f32>, distance: f32, jitter: f32, sky_access: f32) -> vec3<f32> {
    if !volumetric_clouds || cloud_cover <= 0.04 || frame.light_direction.y < 0.055 || distance <= 16.0 {
        return vec3(0.0);
    }
    if frame.camera_position.y >= CLOUD_BASE - 8.0 {
        return vec3(0.0);
    }
    var extent = min(distance, 370.0);
    if ray.y > 0.002 {
        extent = min(extent, (CLOUD_BASE - frame.camera_position.y) / ray.y);
    }
    if extent <= 16.0 {
        return vec3(0.0);
    }
    var illumination = 0.0;
    var shadow_sum = 0.0;
    const STEPS: u32 = 8u;
    let inv_steps = 1.0 / f32(STEPS);
    for (var i = 0u; i < STEPS; i += 1u) {
        let t0 = f32(i) * inv_steps;
        let t1 = f32(i + 1u) * inv_steps;
        let tm = mix(t0, t1, jitter);
        let d = extent * mix(tm, tm * tm, 0.35);
        let w0 = mix(t0, t0 * t0, 0.35);
        let w1 = mix(t1, t1 * t1, 0.35);
        let dt = w1 - w0;
        let pos = ray * d;
        let vis = cloud_shadow_fast(pos);
        let shaft_vis = smoothstep(0.32, 0.90, vis);
        let shaft = shaft_vis * shaft_vis;
        let world_y = frame.camera_position.y + pos.y;
        let alt_fade = exp(-max(world_y - 63.0, 0.0) * 0.0045) * smoothstep(0.0, 20.0, CLOUD_BASE - world_y);
        let dist_fade = (1.0 - exp(-d * 0.016)) * exp(-d * 0.0014);
        illumination += shaft * alt_fade * dist_fade * dt;
        shadow_sum += (1.0 - vis) * dt;
    }
    let cloud_contrast = smoothstep(0.03, 0.28, shadow_sum);
    if cloud_contrast <= 0.001 {
        return vec3(0.0);
    }
    let cos_vl = clamp(dot(ray, frame.light_direction), -1.0, 1.0);
    let phase = 0.55 * photon_phase(cos_vl, 0.72) + 0.35 * photon_phase(cos_vl, 0.22) + 0.012;
    let stratus = cumulus_stratus();
    let humidity = cloud_weather().y;
    let mist = mix(0.60, 1.45, clamp(humidity * 0.65 + 0.50 * stratus + 0.35 * frame.rain, 0.0, 1.0))
        * (1.0 - 0.50 * frame.thunder);
    let cloud_mask = smoothstep(0.08, 0.45, cloud_cover);
    let path_scale = clamp(extent * (1.0 / 280.0), 0.0, 1.0);
    let beam_color = mix(direct_atmosphere_light(), vec3(0.88, 0.93, 1.0) * dot(direct_atmosphere_light(), frame.luminance), 0.25 * frame.rain);
    let vis_mask = clamp(1.0 - frame.blindness - frame.darkness, 0.0, 1.0);
    return beam_color * (illumination * 0.24 * phase * mist * cloud_mask * cloud_contrast * path_scale * sky_access * vis_mask * clamp(fog_strength, 0.0, 1.5));
}

fn fog_at(ray: vec3<f32>, depth: f32, distance: f32, pixel: vec2<f32>) -> AirFog {
    // Animate IGN with the R1 golden-ratio sequence so TAA dissolves volumetric
    // fog shadow dither cleanly instead of locking a static grid onto the screen.
    let ign = fract(52.9829189 * fract(dot(floor(pixel), vec2(0.06711056, 0.00583715))));
    let jitter = fract(ign + f32(frame.frame_index & 63u) * 0.61803399);
    if use_underwater_fog() {
        return raymarch_underwater(ray, distance, jitter);
    }
    if depth <= 0.0 || fog_strength <= 0.0 {
        return AirFog(vec3(0.0), vec3(1.0));
    }
    let sky_access = air_fog_sky_access();
    if sky_access <= 0.0 {
        return AirFog(vec3(0.0), vec3(1.0));
    }
    var fog = raymarch_air_fog(ray, distance, jitter, sky_access, false);
    fog.scattering += fog.transmittance * cloud_terrain_sunbeams(ray, distance, jitter, sky_access);
    return fog;
}

fn fog_transmittance_at(ray: vec3<f32>, depth: f32, distance: f32) -> vec3<f32> {
    if use_underwater_fog() {
        let travel = min(distance, 64.0);
        if travel <= 0.0 {
            return vec3(1.0);
        }
        return exp(-jerlov_underwater_optics().extinction * travel);
    }
    if depth <= 0.0 || fog_strength <= 0.0 {
        return vec3(1.0);
    }
    let sky_access = air_fog_sky_access();
    if sky_access <= 0.0 {
        return vec3(1.0);
    }
    return air_fog_transmittance(ray, distance, sky_access);
}

fn use_volume_composite() -> bool {
    return frame.dimension == 0 && (use_underwater_fog()
        || (frame.eye_in == 0u && (fog_strength > 0.0 || (volumetric_clouds && cloud_cover > 0.0)
            || (rainbows && frame.wetness > 0.0))));
}

fn clouds_at(ray: vec3<f32>, depth: f32, distance: f32) -> vec4<f32> {
    // Clouds against open sky were drawn in the unbobbed sky stage. Only
    // geometry needs a ray clipped against the bobbed scene depth.
    if depth <= 0.0 || frame.eye_in != 0u || !volumetric_clouds || cloud_cover <= 0.0 {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    let eye = frame.camera_position.y;
    if eye + distance < CLOUD_BASE {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    if distance < 3000.0 && max(eye, cloud_radial_height(eye + ray.y * distance, ray.xz * distance)) < CLOUD_BASE {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    // Terrain clouds have no temporal accumulation. A common midpoint avoids
    // changing the march phase when a pixel switches between volume samples.
    return cloud_volume(ray, distance, 10u, 0.5);
}
