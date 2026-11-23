override fog_strength: f32 = 0.85;
const FOG_SEA_LEVEL: f32 = 63.0;
struct AirFog {
    scattering: vec3<f32>,
    transmittance: vec3<f32>,
}

fn air_fog_sky_access() -> f32 {
    return smoothstep(0.12, 0.85, frame.eye_light_smooth.y);
}

fn exp_layer_path(y0: f32, y1: f32, travel: f32, scale_height: f32) -> f32 {
    let dz = y1 - y0;
    if abs(dz) <= 0.001 * scale_height {
        return travel * exp(-max(0.5 * (y0 + y1), 0.0) / scale_height);
    }
    let p0 = exp(-max(y0, 0.0) / scale_height);
    let p1 = exp(-max(y1, 0.0) / scale_height);
    let mean_profile = clamp((min(y1, 0.0) - min(y0, 0.0) + scale_height * (p0 - p1)) / dz, 0.0, 1.0);
    return travel * mean_profile;
}

fn raymarch_air_fog(ray: vec3<f32>, distance: f32, jitter: f32, sky_access: f32, sky: bool) -> AirFog {
    let travel = distance;
    if travel <= 0.0 {
        return AirFog(vec3(0.0), vec3(1.0));
    }

    // Upstream Photon Overworld aerial perspective driven by smooth regional
    // weather (cloud_weather) rather than raw per-block biome temperature/downfall,
    // so crossing a biome border never causes a sudden fog pop.
    let regional = cloud_weather();
    let temp = clamp(regional.x, 0.0, 1.0);
    let humidity = clamp(regional.y, 0.0, 1.0);
    let day = photon_daylight();
    let cool_air = 1.0 - smoothstep(0.15, 0.55, temp);
    let dry_air = smoothstep(0.75, 1.0, temp) * (1.0 - smoothstep(0.15, 0.45, humidity));
    let rayleigh_tint = mix(
        mix(vec3(0.076, 0.415, 1.0), vec3(0.076, 0.70, 1.0), cool_air),
        vec3(0.39, 0.61, 1.0),
        dry_air,
    );
    let base_rayleigh = mix(mix(0.00038, 0.00050, cool_air), 0.00026, dry_air);
    let rayleigh = rayleigh_tint * mix(base_rayleigh * (1.0 + 0.40 * humidity * temp), 0.00050, frame.rain);

    let blue_hour = clamp((exp(-190.0 * (frame.sun_direction.y + 0.07283) * (frame.sun_direction.y + 0.07283)) - 0.05) / 0.95, 0.0, 1.0);
    var mie = 0.0022 * day.sunrise + 0.00010 * day.noon + 0.0024 * day.sunset
        + 0.0032 * day.midnight + 0.0014 * blue_hour;
    let hum_norm = clamp(humidity / 0.8, 0.0, 1.0);
    // Keep midday air crisp like upstream Photon (AIR_FOG_MIE_DENSITY_NOON = 0.0001) while
    // retaining dramatic valley mist under overcast clouds, dawn/dusk, and rain.
    let noon_clear = 1.0 - 0.65 * day.noon * (1.0 - frame.rain);
    let cloud_mist = 0.00024 * smoothstep(0.55, 0.92, cloud_cover) * (0.35 + 0.65 * hum_norm);
    mie = mix(mie + 0.00048 * hum_norm * hum_norm * noon_clear + cloud_mist, select(0.018, 0.012, frame.precipitation == 2u), frame.rain);
    let mie_extinction = mie;
    let mie_scattering = mie * mix(0.90, 0.58, frame.rain);

    let layer = textureLoad(sky_lighting, CLOUD_WEATHER_CELL, 0).zw;
    let forest = textureLoad(sky_lighting, FOREST_FOG_CELL, 0).xyz;
    let forest_h = max(forest.y, 10.0);
    // Subtle species tint: warm ivory-gold in oak/jungle/swamp vs cool blue-silver terpene haze in taiga.
    let forest_tint = mix(vec3(1.03, 1.00, 0.95), vec3(0.86, 0.95, 1.12), clamp(forest.z, 0.0, 1.0));
    let light = direct_atmosphere_light();
    let light_is_visible = max(light.r, max(light.g, light.b)) > 0.001;
    let cast_shadow = !sky && light_is_visible && (pack_shadow_active() || (volumetric_clouds && cloud_cover > 0.0));
    // Hot/dry desert wind-blown dust veil (warm ochre low-altitude Mie dust layer).
    let desert_dust = dry_air * smoothstep(0.48, 0.95, weather_wind()) * (1.0 - frame.rain) * 0.0022;
    let dust_tint = vec3(1.18, 0.96, 0.72);
    // Dynamic dawn river & valley mist banks (drifting low-lying golden haze at sunrise/blue-hour
    // anchored right above sea level Y=62..76, evaporating cleanly to zero by midday).
    let dawn_mist = (day.sunrise + 0.55 * blue_hour) * (1.0 - day.noon) * (1.0 - 0.65 * frame.rain)
        * smoothstep(0.18, 0.65, humidity) * 0.0052;
    let mist_tint = vec3(1.06, 1.01, 0.95);
    let drift = frame.time * vec2(0.0003, 0.00021);
    let h = max(layer.y, 1.0);
    let y_start = frame.camera_position.y - FOG_SEA_LEVEL;
    var transmittance = vec3(1.0);
    var ambient_rayleigh = vec3(0.0);
    var ambient_mie = vec3(0.0);
    var direct_rayleigh = vec3(0.0);
    var direct_mie = vec3(0.0);

    if !cast_shadow {
        // Upstream Photon analytic height-fog: Rayleigh half-life ~43 blocks above sea+30,
        // Mie valley mist half-life ~10 blocks above sea+7, plus ground & forest inversion layers.
        let y_end = y_start + ray.y * travel;
        let density_scale = 0.5 * fog_strength * sky_access;
        let rayleigh_path = exp_layer_path(y_start - 30.0, y_end - 30.0, travel, 43.3) * density_scale;
        let mid_xz = frame.camera_position.xz + ray.xz * (0.5 * travel);
        let noise = textureSampleLevel(wave_noise, wave_noise_sampler, mid_xz * 0.001 + drift, 0.0).a;
        let bank = 0.72 + 0.28 * sin(mid_xz.x * 0.011 + cos(mid_xz.y * 0.009));
        let mie_path = exp_layer_path(y_start - 7.0, y_end - 7.0, travel, 12.0) * mix(0.80, 1.20, noise) * density_scale;
        let ground_path = layer.x * exp_layer_path(y_start, y_end, travel, h) * fog_strength * sky_access;
        // World-anchored valley & canopy-hugging forest inversion layer with drifting wisps.
        let forest_path = forest.x * exp_layer_path(y_start - 8.0, y_end - 8.0, travel, forest_h)
            * mix(0.68, 1.32, noise) * bank * fog_strength * sky_access;
        let dust_path = desert_dust * exp_layer_path(y_start - 4.0, y_end - 4.0, travel, 18.0)
            * mix(0.75, 1.25, noise) * fog_strength * sky_access;
        let dawn_path = dawn_mist * exp_layer_path(y_start - 1.5, y_end - 1.5, travel, 8.5)
            * smoothstep(0.25, 0.80, noise) * bank * fog_strength * sky_access;
        let droplets = ground_path + forest_path + dust_path + dawn_path;
        let optical_depth = rayleigh * rayleigh_path + mie_extinction * mie_path + droplets;
        transmittance = exp(-optical_depth);
        let integrated = select((1.0 - transmittance) / max(optical_depth, vec3(1e-4)),
            1.0 - 0.5 * optical_depth, optical_depth < vec3(1e-4));
        ambient_rayleigh = integrated * rayleigh_path;
        ambient_mie = integrated * (mie_scattering * mie_path + ground_path + forest_path * forest_tint
            + dust_path * dust_tint + dawn_path * mist_tint);
        direct_rayleigh = ambient_rayleigh;
        direct_mie = ambient_mie;
    } else {
        let steps = u32(clamp(4.0 + 0.008 * travel, 4.0, 5.0));
        let step_length = travel / f32(steps);
        let step_scale = step_length * fog_strength * sky_access;
        let half_step_scale = 0.5 * step_scale;
        let inv_h = 1.0 / h;
        let inv_forest_h = 1.0 / forest_h;
        let has_forest = forest.x > 1e-5;
        let has_dust = desert_dust > 1e-5;
        let has_dawn = dawn_mist > 1e-5;
        var along = step_length * jitter;
        let dz = ray.y * step_length;
        var y0 = y_start;
        var profile0 = exp(-max(y0, 0.0) * inv_h);
        for (var step = 0u; step < steps; step += 1u) {
            let position = ray * along;
            let world_y = frame.camera_position.y + position.y - FOG_SEA_LEVEL;
            let world_xz = frame.camera_position.xz + position.xz;
            let noise = textureSampleLevel(wave_noise, wave_noise_sampler, world_xz * 0.001 + drift, 0.0).a;
            let d_rayleigh = exp(-max(world_y - 30.0, 0.0) * (1.0 / 43.3)) * half_step_scale;
            let d_mie = exp(-max(world_y - 7.0, 0.0) * (1.0 / 12.0)) * mix(0.80, 1.20, noise) * half_step_scale;
            let y1 = y0 + dz;
            let profile1 = exp(-max(y1, 0.0) * inv_h);
            var mean_profile = 0.5 * (profile0 + profile1);
            if abs(dz) > 0.001 * h {
                mean_profile = clamp((min(y1, 0.0) - min(y0, 0.0) + h * (profile0 - profile1)) / dz, 0.0, 1.0);
            }
            y0 = y1;
            profile0 = profile1;
            let ground_step = layer.x * mean_profile * step_scale;
            var forest_step = 0.0;
            if has_forest {
                let bank = 0.72 + 0.28 * sin(world_xz.x * 0.011 + cos(world_xz.y * 0.009));
                forest_step = forest.x * exp(-max(world_y - 8.0, 0.0) * inv_forest_h)
                    * mix(0.68, 1.32, noise) * bank * step_scale;
            }
            var dust_step = 0.0;
            if has_dust {
                dust_step = desert_dust * exp(-max(world_y - 4.0, 0.0) * (1.0 / 18.0))
                    * mix(0.75, 1.25, noise) * step_scale;
            }
            var dawn_step = 0.0;
            if has_dawn {
                let bank = 0.65 + 0.35 * sin(world_xz.x * 0.017 + cos(world_xz.y * 0.013) + frame.time * 0.025);
                dawn_step = dawn_mist * exp(-max(world_y - 1.5, 0.0) * (1.0 / 8.5))
                    * smoothstep(0.25, 0.80, noise) * bank * step_scale;
            }
            let droplets = ground_step + forest_step + dust_step + dawn_step;
            let optical_depth = rayleigh * d_rayleigh + mie_extinction * d_mie + droplets;
            let step_transmittance = exp(-optical_depth);
            let integrated = select((1.0 - step_transmittance) / max(optical_depth, vec3(1e-4)),
                1.0 - 0.5 * optical_depth, optical_depth < vec3(1e-4));
            let visible = transmittance * integrated;
            let rayleigh_step = visible * d_rayleigh;
            let mie_step = visible * (mie_scattering * d_mie + ground_step + forest_step * forest_tint
                + dust_step * dust_tint + dawn_step * mist_tint);
            let sun = fog_shadow(position);
            ambient_rayleigh += rayleigh_step;
            ambient_mie += mie_step;
            direct_rayleigh += rayleigh_step * sun;
            direct_mie += mie_step * sun;
            transmittance *= step_transmittance;
            along += step_length;
        }
    }

    // Painterly RDR2 aerial perspective: cool slate-blue ambient haze in cloud/terrain
    // shadows contrasted against warm forward-scattered sunlight in sunbeams.
    let ambient = 2.0 * PHOTON_ISOTROPIC * photon_ambient_irradiance()
        * (ambient_rayleigh * rayleigh * vec3(0.95, 0.99, 1.06) + ambient_mie);
    let cosine = clamp(dot(ray, frame.light_direction), -1.0, 1.0);
    let beam_warmth = mix(vec3(0.94, 0.98, 1.05), vec3(1.10, 1.01, 0.90), smoothstep(-0.25, 0.78, cosine));
    // 3-order multiple scattering for direct fog light, with Jendersie & d'Eon
    // (SIGGRAPH 2023) Draine + HG Mie phase on the primary scattering order for
    // crisp backlit valley/forest mist halos.
    var direct = vec3(0.0);
    var scatter_amt = 1.0;
    var aniso = 1.0;
    for (var o = 0u; o < 3u; o += 1u) {
        var mie_p = 0.7 * photon_phase(cosine, 0.5 * aniso) + 0.3 * photon_phase(cosine, -0.2 * aniso);
        if o == 0u {
            let g_d = 0.60;
            let a_d = 12.0;
            let d_denom = max(1.0 + g_d * g_d - 2.0 * g_d * cosine, 1e-4);
            let draine_p = ((1.0 - g_d * g_d) / (4.0 * PHOTON_PI * d_denom * sqrt(d_denom)))
                * (1.0 + a_d * cosine * cosine) / (1.0 + a_d * (1.0 + 2.0 * g_d * g_d) * (1.0 / 3.0));
            mie_p = 0.55 * photon_phase(cosine, 0.74) + 0.30 * draine_p + 0.15 * photon_phase(cosine, -0.20);
        }
        direct += scatter_amt * (direct_rayleigh * rayleigh * PHOTON_ISOTROPIC + direct_mie * (mie_p * beam_warmth));
        scatter_amt *= 0.5;
        aniso *= 0.7;
    }
    direct *= light * (1.0 - 0.52 * frame.rain);
    let evening_glow = 0.28 * clamp((exp(-300.0 * (frame.sun_direction.y + 0.02) * (frame.sun_direction.y + 0.02)) - 0.05) / 0.95, 0.0, 1.0);
    let total_scatter = ambient + direct * (1.0 + evening_glow);
    let visibility = clamp(1.0 - frame.blindness - frame.darkness, 0.0, 1.0);
    return AirFog(total_scatter * visibility, transmittance);
}

// Exact closed-form analytic air-fog transmittance (independent of shadows, 0 raymarch steps).
fn air_fog_transmittance(ray: vec3<f32>, distance: f32, sky_access: f32) -> vec3<f32> {
    if distance <= 0.0 {
        return vec3(1.0);
    }
    let regional = cloud_weather();
    let temp = clamp(regional.x, 0.0, 1.0);
    let humidity = clamp(regional.y, 0.0, 1.0);
    let day = photon_daylight();
    let cool_air = 1.0 - smoothstep(0.15, 0.55, temp);
    let dry_air = smoothstep(0.75, 1.0, temp) * (1.0 - smoothstep(0.15, 0.45, humidity));
    let rayleigh_tint = mix(
        mix(vec3(0.076, 0.415, 1.0), vec3(0.076, 0.70, 1.0), cool_air),
        vec3(0.39, 0.61, 1.0),
        dry_air,
    );
    let base_rayleigh = mix(mix(0.00038, 0.00050, cool_air), 0.00026, dry_air);
    let rayleigh = rayleigh_tint * mix(base_rayleigh * (1.0 + 0.40 * humidity * temp), 0.00050, frame.rain);

    let blue_hour = clamp((exp(-190.0 * (frame.sun_direction.y + 0.07283) * (frame.sun_direction.y + 0.07283)) - 0.05) / 0.95, 0.0, 1.0);
    var mie = 0.0022 * day.sunrise + 0.00010 * day.noon + 0.0024 * day.sunset
        + 0.0032 * day.midnight + 0.0014 * blue_hour;
    let hum_norm = clamp(humidity / 0.8, 0.0, 1.0);
    let noon_clear = 1.0 - 0.65 * day.noon * (1.0 - frame.rain);
    let cloud_mist = 0.00024 * smoothstep(0.55, 0.92, cloud_cover) * (0.35 + 0.65 * hum_norm);
    mie = mix(mie + 0.00048 * hum_norm * hum_norm * noon_clear + cloud_mist, select(0.018, 0.012, frame.precipitation == 2u), frame.rain);

    let layer = textureLoad(sky_lighting, CLOUD_WEATHER_CELL, 0).zw;
    let forest = textureLoad(sky_lighting, FOREST_FOG_CELL, 0).xyz;
    let forest_h = max(forest.y, 10.0);
    let desert_dust = dry_air * smoothstep(0.48, 0.95, weather_wind()) * (1.0 - frame.rain) * 0.0022;
    let dawn_mist = (day.sunrise + 0.55 * blue_hour) * (1.0 - day.noon) * (1.0 - 0.65 * frame.rain)
        * smoothstep(0.18, 0.65, humidity) * 0.0052;
    let drift = frame.time * vec2(0.0003, 0.00021);
    let h = max(layer.y, 1.0);
    let y_start = frame.camera_position.y - FOG_SEA_LEVEL;
    let y_end = y_start + ray.y * distance;
    let density_scale = 0.5 * fog_strength * sky_access;
    let rayleigh_path = exp_layer_path(y_start - 30.0, y_end - 30.0, distance, 43.3) * density_scale;
    let mid_xz = frame.camera_position.xz + ray.xz * (0.5 * distance);
    let noise = textureSampleLevel(wave_noise, wave_noise_sampler, mid_xz * 0.001 + drift, 0.0).a;
    let bank = 0.72 + 0.28 * sin(mid_xz.x * 0.011 + cos(mid_xz.y * 0.009));
    let mie_path = exp_layer_path(y_start - 7.0, y_end - 7.0, distance, 12.0) * mix(0.80, 1.20, noise) * density_scale;
    let ground_path = layer.x * exp_layer_path(y_start, y_end, distance, h) * fog_strength * sky_access;
    let forest_path = forest.x * exp_layer_path(y_start - 8.0, y_end - 8.0, distance, forest_h)
        * mix(0.68, 1.32, noise) * bank * fog_strength * sky_access;
    let dust_path = desert_dust * exp_layer_path(y_start - 4.0, y_end - 4.0, distance, 18.0)
        * mix(0.75, 1.25, noise) * fog_strength * sky_access;
    let dawn_path = dawn_mist * exp_layer_path(y_start - 1.5, y_end - 1.5, distance, 8.5)
        * smoothstep(0.25, 0.80, noise) * bank * fog_strength * sky_access;
    let optical_depth = rayleigh * rayleigh_path + mie * mie_path + (ground_path + forest_path + dust_path + dawn_path);
    return exp(-optical_depth);
}

