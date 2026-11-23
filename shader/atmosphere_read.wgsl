fn atmosphere(direction: vec3<f32>) -> vec3<f32> {
    let azimuth = atan2(direction.x, direction.z + 1e-8) / TAU + 0.5;
    let elevation = 0.5 + 0.5 * sign(direction.y) * sqrt(min(abs(direction.y), 1.0));
    let texel = vec2(
        1.0 + azimuth * f32(SKY_AZIMUTH_TEXELS),
        0.5 + elevation * f32(SKY_ELEVATION_TEXELS - 1u),
    );
    let uv = texel / vec2<f32>(textureDimensions(sky_lighting));
    return textureSampleLevel(sky_lighting, scattering_atlas_sampler, uv, 0.0).rgb;
}

fn direct_atmosphere_light() -> vec3<f32> {
    return textureLoad(sky_lighting, vec2(0, SKY_LIGHT_ROW), 0).rgb;
}

fn ambient_atmosphere_light() -> vec3<f32> {
    return textureLoad(sky_lighting, vec2(1, SKY_LIGHT_ROW), 0).rgb;
}

// The sky map and its SH bake use a 2.25 multiplier. Photon uses 1.13 * tau.
const PHOTON_SKY_IRRADIANCE_SCALE: f32 = TAU * 1.13 / 2.25;

fn photon_ambient_irradiance() -> vec3<f32> {
    return ambient_atmosphere_light() * PHOTON_SKY_IRRADIANCE_SCALE;
}

fn solar_atmosphere_light() -> vec3<f32> {
    return textureLoad(sky_lighting, vec2(2, SKY_LIGHT_ROW), 0).rgb;
}
