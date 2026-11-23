override cloud_cover: f32 = 0.55;
override volumetric_clouds: bool = true;
override blocky_clouds: bool = false;
override second_blocky_layer: bool = false;

const CLOUD_ROTATION = mat2x2<f32>(vec2(0.8, 0.6), vec2(-0.6, 0.8));
const CLOUD_LAYER_HEIGHT: f32 = 202.0;
const BLOCKY_SECOND_HEIGHT: f32 = 286.0;

fn cloud_hash(cell: vec2<i32>) -> f32 {
    // Preserve the lattice coordinates on both sides of the world origin.
    let bits = bitcast<vec2<u32>>(cell);
    var hash = (bits.x * 0x8da6b343u) ^ (bits.y * 0xd8163841u) ^ 0x9e3779b9u;
    hash = (hash ^ (hash >> 16u)) * 0x7feb352du;
    return f32((hash ^ (hash >> 15u)) >> 8u) * (1.0 / 16777216.0);
}

fn cloud_noise(position: vec2<f32>) -> vec3<f32> {
    let cell = vec2<i32>(floor(position));
    let f = fract(position);
    let w = f * f * (3.0 - 2.0 * f);
    let dw = 6.0 * f * (1.0 - f);
    let lower_left = cloud_hash(cell);
    let upper_left = cloud_hash(cell + vec2(0, 1));
    let lower_delta = cloud_hash(cell + vec2(1, 0)) - lower_left;
    let upper_delta = cloud_hash(cell + vec2(1, 1)) - upper_left;
    let lower = lower_left + w.x * lower_delta;
    let upper = upper_left + w.x * upper_delta;
    return vec3(
        mix(lower, upper, w.y),
        mix(lower_delta, upper_delta, w.y) * dw.x,
        (upper - lower) * dw.y,
    );
}

fn cloud_wind() -> vec2<f32> {
    return cloud_wind_at(frame.time);
}

fn cloud_wind_at(time: f32) -> vec2<f32> {
    let sway = sin(time * 0.038) * (0.035 + 0.065 * weather_wind());
    return vec2(0.013 * time + sway, -0.005 * time + 0.4 * sway);
}

struct CloudLayer {
    coverage: f32,
    gradient: vec2<f32>,
}

fn cloud_layer(coord: vec2<f32>, elevation: f32) -> CloudLayer {
    let lower = CLOUD_ROTATION * (coord + cloud_wind());
    let middle = CLOUD_ROTATION * lower * 2.03 + vec2(7.1, 13.4);
    let broad = cloud_noise(lower);
    let detail = cloud_noise(middle);
    let fine = cloud_noise(CLOUD_ROTATION * middle * 2.01 + vec2(19.0, 3.0)).x;
    let middle_fade = smoothstep(0.04, 0.20, elevation);
    let detail_fade = smoothstep(0.08, 0.34, elevation);
    let base = 0.5 + 0.56 * (broad.x - 0.5)
        + 0.28 * (detail.x - 0.5) * middle_fade + 0.16 * (fine - 0.5) * detail_fade;
    let humidity = cloud_weather().y;
    let threshold = mix(0.81, 0.35, cloud_cover) - 0.12 * frame.rain - 0.045 * (humidity - 0.5);
    let gradient = 0.56 * broad.yz + (0.28 * 2.03) * middle_fade * (detail.yz * CLOUD_ROTATION);
    return CloudLayer(smoothstep(threshold, threshold + 0.19, base), gradient);
}

fn cloud_linear(low: f32, high: f32, x: f32) -> f32 {
    return clamp((x - low) / (high - low), 0.0, 1.0);
}

fn cloud_square(x: f32) -> f32 {
    return x * x;
}

fn cloud_lift(x: f32, amount: f32) -> f32 {
    return x * (1.0 + amount) / (1.0 + x * amount);
}

fn photon_cloud_noise(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(photon_noise, photon_noise_sampler, uv, 0.0);
}

fn cumulus_stratus() -> f32 {
    return photon_dampen(cloud_linear(0.5, 1.0, 1.0 - cloud_weather().x))
        * photon_dampen(1.0 - weather_wind());
}

fn cumulus_coverage_range(stratus: f32) -> vec2<f32> {
    let temperature = 1.0 - 0.15 * cloud_linear(0.6, 1.0, cloud_weather().x);
    let humidity = cloud_weather().y;
    let humidity_weight = 0.4 * humidity + 0.5 * humidity * humidity;
    let variation = vec2(-0.1, 1.0) * (0.1 + 0.1 * weather_wind());
    let cover = clamp(temperature * humidity_weight + variation + 0.3 * stratus * stratus,
        vec2(0.0), vec2(1.0));
    // Preserve the prototype's coverage slider while using Photon's coverage model.
    return min(vec2(1.0), cover * cloud_cover / 0.55);
}

fn cumulus_coverage(world: vec2<f32>) -> f32 {
    let coord = world + cloud_drift(frame.time, 0u);
    let broad = photon_cloud_noise(0.00002 * coord).r;
    let shape = photon_cloud_noise(0.00027 * coord).a;
    let stratus = cumulus_stratus();
    let cover = cumulus_coverage_range(stratus);
    let cumulus = cloud_linear(1.0 - min(mix(cover.x, cover.y, broad), 0.999), 1.0, shape);
    var sheet = cloud_linear(0.9 - cover.x, 1.0, 2.0 * broad * cover.y);
    sheet = sheet * sheet * (3.0 - 2.0 * sheet) * (0.5 + cloud_linear(0.3, 0.6, shape));
    // Preserve natural localized cloud breaks ("sucker holes") in heavy overcast/stratus
    // so dramatic crepuscular god rays can punch through dark cloud undersides.
    let break_hole = smoothstep(0.76, 0.95, 1.0 - shape) * smoothstep(0.28, 0.82, stratus + 0.45 * frame.rain);
    let sheet_broken = (sheet / (sheet + 1.0)) * (1.0 - 0.82 * break_hole);
    return mix(cumulus, sheet_broken, stratus);
}

fn cloud_weather() -> vec2<f32> {
    return textureLoad(sky_lighting, CLOUD_WEATHER_CELL, 0).xy;
}

fn cloud_drift(time: f32, layer: u32) -> vec2<f32> {
    if blocky_clouds {
        return cloud_wind_at(time) * 200.0;
    }
    // Photon uses separate world-space wind vectors for each altitude.
    let velocity = select(vec2(2.1650635, 1.25), vec2(1.25, 2.1650635), layer == 2u);
    return time * select(velocity, vec2(0.0, 4.0), layer >= 3u);
}

fn cloud_height_fraction(height: f32, cover: f32, stratus: f32) -> f32 {
    let stratus_profile = photon_dampen(clamp(2.0 * height, 0.0, 1.0)
        * cloud_linear(0.0, 0.1, height) * cloud_linear(0.0, 0.6, 1.0 - height));
    let shaped = mix(cover, clamp(cover * stratus_profile, 0.0, 1.0), stratus);
    let egg = smoothstep(0.2, 1.0, height) * (0.6 - 0.3 * stratus);
    return (shaped - egg) * smoothstep(0.0, 0.2, height);
}

fn cumulus_height_scale(stratus: f32) -> f32 {
    let cover = cumulus_coverage_range(stratus);
    let dynamic = mix(0.5, 1.0, smoothstep(0.4, 0.6, dot(cover, vec2(0.25, 0.75))));
    return 0.8 / dynamic;
}

fn projected_cloud_shadow(position: vec3<f32>) -> f32 {
    let light = frame.light_direction;
    if cloud_cover <= 0.0 || light.y <= 0.05 {
        return 1.0;
    }
    let world = frame.camera_position + position;
    if volumetric_clouds {
        if blocky_clouds {
            var depth = 0.0;
            for (var layer = 0u; layer < select(1u, 2u, second_blocky_layer); layer += 1u) {
                let altitude = select(CLOUD_LAYER_HEIGHT, BLOCKY_SECOND_HEIGHT, layer == 1u);
                let start = max(0.0, (altitude - 8.0 - world.y) / light.y);
                let end = min(1600.0, (altitude + 8.0 - world.y) / light.y);
                if end > start {
                    depth += (end - start) * blocky_density(world + light * (0.5 * (start + end)), layer);
                }
            }
            return mix(1.0, exp(-0.17 * depth), 0.8 * smoothstep(0.05, 0.15, light.y));
        }
        let start = max(0.0, (CLOUD_BASE - world.y) / light.y);
        let end = min(2400.0, (CLOUD_TOP - world.y) / light.y);
        let travel = end - start;
        if travel <= 0.0 {
            return 1.0;
        }
        let stratus = cumulus_stratus();
        let scale = cumulus_height_scale(stratus) / (CLOUD_TOP - CLOUD_BASE);
        let step_size = travel * 0.5;
        var optical_depth = 0.0;
        for (var i = 0u; i < 2u; i += 1u) {
            let point = world + light * (start + (f32(i) + 0.5) * step_size);
            let height = (point.y - CLOUD_BASE) * scale;
            let density = cloud_height_fraction(height, cumulus_coverage(point.xz), stratus);
            optical_depth += max(0.0, density - 0.085) * step_size;
        }
        return mix(1.0, exp(-0.065 * optical_depth), 0.78 * smoothstep(0.05, 0.15, light.y));
    }
    let height = CLOUD_LAYER_HEIGHT - world.y;
    if height <= 0.0 {
        return 1.0;
    }
    let coord = (world.xz + light.xz * (height / light.y)) * 0.005;
    let layer = cloud_layer(coord, light.y);
    let fade = 0.8 * smoothstep(0.05, 0.15, light.y) * smoothstep(0.0, 20.0, height);
    return mix(1.0, exp(-layer.coverage / light.y), fade);
}

// Shared with the visible sky, skylight bake, and the water fallback.
fn cloud_environment(direction: vec3<f32>, air: vec3<f32>) -> vec3<f32> {
    if volumetric_clouds || direction.y <= 0.025 || cloud_cover <= 0.0 {
        return air;
    }

    let wind = cloud_wind();
    let world_offset = frame.camera_position.xz * 0.005;
    let plane = direction.xz / direction.y;
    let detail_fade = smoothstep(0.08, 0.34, direction.y);
    let upper = (CLOUD_ROTATION * (plane * 1.36 + world_offset + wind * 1.27))
        * vec2(0.44, 1.36);
    let upper_cloud = 0.74 * cloud_noise(upper).x
        + 0.26 * cloud_noise(CLOUD_ROTATION * upper * 1.97 + vec2(11.6, 5.2)).x;
    let wisps = smoothstep(0.64, 0.84, upper_cloud) * cloud_cover * detail_fade;

    let horizon_fade = smoothstep(0.025, 0.24, direction.y);
    let ambient = ambient_atmosphere_light();
    let light = direct_atmosphere_light();
    let cloud_tint = ambient * vec3(0.63, 0.68, 0.79);
    let rain_fade = 1.0 - 0.33 * frame.rain;
    let cirrus = mix(air, (cloud_tint + 0.04 * light) * rain_fade,
        wisps * 0.36 * horizon_fade);
    let layer_height = max(CLOUD_LAYER_HEIGHT - frame.camera_position.y, 0.0);
    let clouds = cloud_layer(world_offset + plane * (layer_height * 0.005), direction.y);
    let projected_light = CLOUD_ROTATION * frame.light_direction.xz;
    let edge_light = clamp(0.5 - 0.48 * dot(clouds.gradient, projected_light), 0.0, 1.0);
    let cloud_color = (cloud_tint + light * mix(0.01, 0.07, edge_light)) * rain_fade;
    let layer_fade = smoothstep(0.0, 20.0, layer_height);
    return mix(cirrus, cloud_color, clouds.coverage * horizon_fade * layer_fade * 0.91);
}

// Photon defines cloud altitude in meters relative to sea level at 10 m per block.
const CLOUD_BASE: f32 = 183.0;
const CLOUD_TOP: f32 = 303.0;
const MID_CLOUD_BASE: f32 = 383.0;
const MID_CLOUD_TOP: f32 = 431.0;
const CONGESTUS_BASE: f32 = 103.0;
const CONGESTUS_TOP: f32 = 1383.0;
const CLOUD_PLANET_RADIUS: f32 = 637100.0;

fn cloud_worley_slice(xy: vec2<f32>, z: f32) -> vec2<f32> {
    let index = floor(z) % 64.0;
    let tile = vec2(index % 8.0, floor(index / 8.0));
    // One wrapped texel surrounds each 64×64 slice for periodic bilinear filtering.
    let uv = (tile * 66.0 + 1.0 + fract(xy) * 64.0) / 528.0;
    return textureSampleLevel(cloud_noise_atlas, cloud_noise_atlas_sampler, uv, 0.0).rg;
}

fn cloud_worley(point: vec3<f32>) -> vec2<f32> {
    let z = fract(point.z) * 64.0 - 0.5;
    let slice = floor(z);
    return mix(cloud_worley_slice(point.xy, slice + 64.0),
        cloud_worley_slice(point.xy, slice + 65.0), fract(z));
}

fn cloud_radial_height(y: f32, xz: vec2<f32>) -> f32 {
    // Rationalize the difference of radii so local height retains precision.
    let r = CLOUD_PLANET_RADIUS + y - 63.0;
    return y + dot(xz, xz) / (sqrt(r * r + dot(xz, xz)) + r);
}

fn blocky_density(world: vec3<f32>, layer: u32) -> f32 {
    let altitude = select(CLOUD_LAYER_HEIGHT, BLOCKY_SECOND_HEIGHT, layer == 1u);
    let pixel = floor((world.xz + cloud_drift(frame.time, layer) + vec2(97.0 * f32(layer))) / 9.0);
    let density = cloud_noise(pixel * 0.11).x;
    return step(0.73 - 0.34 * cloud_cover, density)
        * (1.0 - smoothstep(6.0, 8.0, abs(world.y - altitude)));
}

fn cumulus_density(world: vec3<f32>, altitude: f32, coverage: f32, stratus: f32, altitude_scale: f32) -> f32 {
    let height = clamp((altitude - CLOUD_BASE) * altitude_scale / (CLOUD_TOP - CLOUD_BASE), 0.0, 1.0);
    var density = cloud_height_fraction(height, coverage, stratus);
    if density <= 0.0 {
        return 0.0;
    }

    let drift = cloud_drift(frame.time, 0u);
    let horiz_dist = length(world.xz - frame.camera_position.xz);
    // Fade sub-Nyquist high-frequency Worley erosion on distant clouds so
    // horizon silhouettes stay smooth instead of aliasing into fuzzy jagged teeth.
    let fine_lod = 1.0 - smoothstep(850.0, 2800.0, horiz_dist);
    let bubbly_lod = 1.0 - 0.50 * smoothstep(1600.0, 5200.0, horiz_dist);

    // Subtle Nubis3 updraft curl & altitude wind shear: keeps flat LCL condensation
    // bases smooth while adding gentle organic billow displacement toward cloud tops.
    let wind_shear = normalize(drift + vec2(0.001, 0.0)) * (10.0 * height * height * (1.0 - 0.65 * stratus));
    let curl_grad = cloud_noise((world.xz + drift) * 0.0018).yz;
    let curl_vec = vec2(curl_grad.y, -curl_grad.x);
    let curl_amp = (2.5 + 6.5 * height * height) * (1.0 - 0.72 * stratus) * bubbly_lod;
    let warped_xz = world.xz + wind_shear + curl_vec * curl_amp;

    let p = vec3(warped_xz.x, CLOUD_PLANET_RADIUS + world.y - 63.0, warped_xz.y);
    let bubbly = cloud_worley((p + vec3(drift.x, 0.0, drift.y) * 1.2) * 0.009).r;
    let fine_position = vec3(warped_xz.x, world.y, warped_xz.y) + vec3(drift.x, 0.0, drift.y) * 1.4;
    let fine_xz = CLOUD_ROTATION * fine_position.xz;
    let swirley = select(0.0, cloud_worley(vec3(fine_xz.x, fine_position.y, fine_xz.y) * 0.05).g, fine_lod > 0.01);
    let torn = cloud_linear(0.66, 0.9, weather_wind()) * (1.0 - stratus);
    let detail = mix(vec2(0.33, 0.40) * (1.0 + 0.5 * torn), vec2(0.07, 0.10),
        vec2(stratus * stratus, stratus)) * vec2(bubbly_lod, fine_lod);
    // Keep LCL base slightly softer/flatter and accentuate billowing tops.
    let billow_profile = mix(0.65, 1.12, smoothstep(0.12, 0.68, height));
    let detail_fade = 0.20 * smoothstep(0.85, 1.0, 1.0 - height)
        - 0.35 * smoothstep(0.05, 0.5, height) + 0.6;
    density -= dot(detail, vec2(bubbly * bubbly * billow_profile, swirley * swirley * detail_fade))
        * photon_dampen(clamp(1.0 - density, 0.0, 1.0));
    let edge = mix(vec2(3.0, 12.0), vec2(2.0, 7.0), stratus * stratus);
    let sharpness = mix(edge.x, edge.y, height) * mix(0.45, 1.0, fine_lod);
    return cloud_lift(max(0.0, density), sharpness)
        * (0.1 + 0.9 * smoothstep(0.18, 0.68, height));
}

fn mid_stratus() -> f32 {
    return smoothstep(0.5, 0.66, 1.0 - cloud_weather().x);
}

fn mid_coverage(stratus: f32) -> vec2<f32> {
    let humidity = cloud_weather().y;
    let wind = weather_wind();
    let patches = (0.5 + 0.5 * cloud_linear(0.4, 1.0, humidity)) * cloud_linear(0.4, 0.66, wind)
        * cloud_linear(1.0, 0.8, cloud_weather().x);
    let sheet = cloud_linear(0.25, 0.45, wind * humidity);
    return clamp(mix(vec2(0.5, 1.5) * patches, vec2(sheet), stratus) * cloud_cover / 0.55,
        vec2(0.0), vec2(1.0));
}

fn mid_cloud_density(world: vec3<f32>, altitude: f32, stratus: f32, cover: vec2<f32>) -> f32 {
    let drift = cloud_drift(frame.time, 2u);
    let p = world.xz + drift;
    let thickness = mix(0.5, 1.0, smoothstep(0.4, 0.6, cover.y));
    let height = clamp(0.8 * (altitude - MID_CLOUD_BASE) / ((MID_CLOUD_TOP - MID_CLOUD_BASE) * thickness), 0.0, 1.0);
    let broad = photon_cloud_noise(p * 0.00005 + 0.01).r;
    let shape = photon_cloud_noise(p * 0.00047 + 0.3).ar;
    let fine = photon_cloud_noise(p * 0.001 + 0.6).a;
    let threshold = min(0.999, cloud_square(1.0 - mix(cover.x, cover.y, photon_dampen(broad))));
    var patches = cloud_linear(threshold, 1.0, shape.x);
    patches -= 1.9 * cloud_square(1.0 - fine) * photon_dampen(1.0 - patches);

    var sheet = smoothstep(0.9 - cover.x, 1.0, 2.0 * broad * cover.y);
    sheet *= (0.5 + cloud_linear(0.3, 0.6, shape.x)) * (0.5 + cloud_linear(0.3, 0.7, shape.y));
    sheet /= 1.0 + sheet;
    var density = mix(cloud_height_fraction(height, patches, 0.0),
        clamp(sheet * photon_dampen(cloud_linear(0.0, 0.5, height)
            * cloud_linear(0.0, 0.1, height) * cloud_linear(0.0, 0.6, 1.0 - height)), 0.0, 1.0), stratus);
    if density <= 0.0 {
        return 0.0;
    }
    let coord = vec3(p.x + 0.2 * drift.x, CLOUD_PLANET_RADIUS + world.y - 63.0, p.y + 0.2 * drift.y);
    let worley = cloud_worley(coord * 0.03).g;
    density -= 0.2 * worley * worley * photon_dampen(clamp(1.0 - density, 0.0, 1.0)) * (1.0 - 0.85 * stratus);
    let edge = mix(vec2(3.0, 14.0), vec2(2.0, 7.0), stratus);
    return cloud_lift(max(0.0, density), mix(edge.x, edge.y, height))
        * (0.1 + 0.9 * smoothstep(0.2, 0.7, height));
}

fn congestus_amount() -> f32 {
    let t = cloud_linear(0.5, 1.0, cloud_weather().x);
    let humidity = cloud_linear(0.3, 0.9, cloud_weather().y);
    let wind = cloud_square(weather_wind());
    let cover = cumulus_coverage_range(cumulus_stratus());
    return clamp(1.5 * photon_dampen(photon_dampen(t * humidity * wind))
        * (1.0 - cloud_linear(0.45, 0.50, dot(cover, vec2(0.66, 0.33)))), 0.0, 1.0);
}

fn cumulus_density_fast(altitude: f32, coverage: f32, stratus: f32, altitude_scale: f32) -> f32 {
    let height = clamp((altitude - CLOUD_BASE) * altitude_scale / (CLOUD_TOP - CLOUD_BASE), 0.0, 1.0);
    var density = cloud_height_fraction(height, coverage, stratus);
    if density <= 0.0 {
        return 0.0;
    }
    // Match upstream Photon's non-Worley optical depth fallback (worley = 0.5 -> worley^2 = 0.25)
    // so thin cloud edges never drop to zero optical depth and blow out in forward scattering.
    let detail = mix(vec2(0.33, 0.40), vec2(0.07, 0.10), vec2(stratus * stratus, stratus));
    let detail_fade = 0.20 * smoothstep(0.85, 1.0, 1.0 - height)
        - 0.35 * smoothstep(0.05, 0.5, height) + 0.6;
    density -= dot(detail, vec2(0.20, 0.20 * detail_fade))
        * photon_dampen(clamp(1.0 - density, 0.0, 1.0));
    let edge = mix(vec2(3.0, 12.0), vec2(2.0, 7.0), stratus * stratus);
    return cloud_lift(max(0.0, density), mix(edge.x, edge.y, height))
        * (0.1 + 0.9 * smoothstep(0.2, 0.7, height));
}

fn mid_cloud_density_fast(world: vec3<f32>, altitude: f32, stratus: f32, cover: vec2<f32>) -> f32 {
    let p = world.xz + cloud_drift(frame.time, 2u);
    let thickness = mix(0.5, 1.0, smoothstep(0.4, 0.6, cover.y));
    let height = clamp(0.8 * (altitude - MID_CLOUD_BASE) / ((MID_CLOUD_TOP - MID_CLOUD_BASE) * thickness), 0.0, 1.0);
    let broad = photon_cloud_noise(p * 0.00005 + 0.01).r;
    let shape = photon_cloud_noise(p * 0.00047 + 0.3).ar;
    let threshold = min(0.999, cloud_square(1.0 - mix(cover.x, cover.y, photon_dampen(broad))));
    let patches = cloud_linear(threshold, 1.0, shape.x);
    var sheet = smoothstep(0.9 - cover.x, 1.0, 2.0 * broad * cover.y);
    sheet *= (0.5 + cloud_linear(0.3, 0.6, shape.x)) * (0.5 + cloud_linear(0.3, 0.7, shape.y));
    sheet /= 1.0 + sheet;
    let density = mix(cloud_height_fraction(height, patches, 0.0),
        clamp(sheet * photon_dampen(cloud_linear(0.0, 0.5, height)
            * cloud_linear(0.0, 0.1, height) * cloud_linear(0.0, 0.6, 1.0 - height)), 0.0, 1.0), stratus);
    if density <= 0.0 {
        return 0.0;
    }
    let edge = mix(vec2(3.0, 14.0), vec2(2.0, 7.0), stratus);
    return cloud_lift(max(0.0, density * 0.85), mix(edge.x, edge.y, height))
        * (0.1 + 0.9 * smoothstep(0.2, 0.7, height));
}

fn congestus_density_fast(world: vec3<f32>, relative: vec3<f32>, altitude: f32, amount: f32) -> f32 {
    let height = clamp((altitude - CONGESTUS_BASE) / (CONGESTUS_TOP - CONGESTUS_BASE), 0.0, 1.0);
    let distance = cloud_linear(3000.0, 5000.0, length(relative.xz));
    let drift = cloud_drift(frame.time, 1u);
    let shear = cloud_drift(50.0 * height * height, 1u);
    let shape = photon_cloud_noise((world.xz + drift + shear) * 0.00003).a;
    // RDR2 cumulonimbus anvil flare: tall storm towers widen into flat-topped anvils at height 0.74..0.96.
    let anvil_flare = smoothstep(0.72, 0.88, height) * (1.0 - smoothstep(0.92, 1.0, height))
        * (0.06 + 0.10 * max(frame.rain, frame.thunder));
    var density = 1.5 * cloud_square(cloud_linear(0.75 - 0.15 * amount - anvil_flare, 1.0, sqrt(shape)));
    density -= cloud_square(cloud_linear(0.3, 1.0, height)) * (1.0 - 0.2 * amount) * (1.0 - 0.65 * smoothstep(0.72, 0.90, height));
    density *= smoothstep(0.0, 0.1, height) * smoothstep(0.0, 0.08, 1.0 - height)
        * (4.0 * distance * (1.0 - distance)) * cloud_linear(0.0, 0.3, amount);
    if density <= 0.0 {
        return 0.0;
    }
    return cloud_lift(clamp(density * 0.75, 0.0, 1.0), mix(2.0, 10.0, height))
        * cloud_square(cloud_linear(0.0, 0.5, height));
}

fn congestus_density(world: vec3<f32>, relative: vec3<f32>, altitude: f32, amount: f32) -> f32 {
    let height = clamp((altitude - CONGESTUS_BASE) / (CONGESTUS_TOP - CONGESTUS_BASE), 0.0, 1.0);
    let distance = cloud_linear(3000.0, 5000.0, length(relative.xz));
    let drift = cloud_drift(frame.time, 1u);
    let shear = cloud_drift(50.0 * height * height, 1u);
    let shape = photon_cloud_noise((world.xz + drift + shear) * 0.00003).a;
    // RDR2 cumulonimbus anvil flare: widens tall storm towers into flat-topped anvils.
    let anvil_flare = smoothstep(0.72, 0.88, height) * (1.0 - smoothstep(0.92, 1.0, height))
        * (0.06 + 0.10 * max(frame.rain, frame.thunder));
    var density = 1.5 * cloud_square(cloud_linear(0.75 - 0.15 * amount - anvil_flare, 1.0, sqrt(shape)));
    density -= cloud_square(cloud_linear(0.3, 1.0, height)) * (1.0 - 0.2 * amount) * (1.0 - 0.65 * smoothstep(0.72, 0.90, height));
    density *= smoothstep(0.0, 0.1, height) * smoothstep(0.0, 0.08, 1.0 - height)
        * (4.0 * distance * (1.0 - distance)) * cloud_linear(0.0, 0.3, amount);
    if density <= 0.0 {
        return 0.0;
    }
    let p = vec3(world.x, CLOUD_PLANET_RADIUS + world.y - 63.0, world.z);
    let a = cloud_worley((p + vec3(drift.x, 0.0, drift.y) * 0.2) * 0.0005).g;
    let b = cloud_worley((p + vec3(drift.x, 0.0, drift.y) * 0.4) * 0.0023).g;
    let edge = clamp(1.0 - density, 0.0, 1.0);
    let detail = 0.20 * smoothstep(0.85, 1.0, 1.0 - height) - 0.35 * smoothstep(0.05, 0.5, height) + 0.8;
    density -= 7.0 * edge * edge * a * a * a * height * height;
    density -= 0.2 * b * b * photon_dampen(edge)
        * photon_dampen(photon_dampen(clamp(detail + 0.5 * height * height, 0.0, 1.0)));
    return cloud_lift(clamp(density, 0.0, 1.0), mix(2.0, 10.0, height))
        * cloud_square(cloud_linear(0.0, 0.5, height));
}
