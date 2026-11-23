override contact_ao: f32 = 0.65;
override terrain_reflections: bool = true;

// Keep the payload within 24 exact integer bits:
//   bits  0..11: reflectance RGB (4 bits each, sqrt-encoded for dielectric precision)
//   bits 12..16: direct_fraction (5 bits = 32 levels, +/-1.5% distant shadow accuracy)
//   bits 17..20: ambient_fraction (4 bits = 16 levels)
//   bits 21..23: skylight level (3 bits = 8 levels)
fn surface_payload(fresnel: vec3<f32>, ambient: f32, skylight: f32, direct_fraction: f32) -> f32 {
    let rgb = vec3<u32>(round(sqrt(clamp(fresnel, vec3(0.0), vec3(1.0))) * 15.0));
    let direct = u32(round(clamp(direct_fraction, 0.0, 1.0) * 31.0));
    let a = u32(round(clamp(ambient, 0.0, 1.0) * 15.0));
    let sky = u32(round(clamp(skylight, 0.0, 1.0) * 7.0));
    return f32(rgb.x | (rgb.y << 4u) | (rgb.z << 8u) | (direct << 12u) | (a << 17u) | (sky << 21u));
}

fn surface_reflectance(data: u32) -> vec3<f32> {
    let rgb = vec3<f32>(f32(data & 15u), f32((data >> 4u) & 15u), f32((data >> 8u) & 15u)) / 15.0;
    return rgb * rgb;
}

fn surface_direct(data: u32) -> f32 {
    return f32((data >> 12u) & 31u) * (1.0 / 31.0);
}

fn surface_ambient(data: u32) -> f32 {
    return f32((data >> 17u) & 15u) * (1.0 / 15.0);
}

fn surface_sky(data: u32) -> f32 {
    return f32((data >> 21u) & 7u) * (1.0 / 7.0);
}
