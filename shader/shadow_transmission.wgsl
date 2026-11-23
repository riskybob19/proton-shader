// Horizon stores linear color and coverage in the native translucent shadow map.
fn shadow_transmission(uv: vec2<f32>, depth: f32, solid: f32) -> vec3<f32> {
    if solid <= 0.005 {
        return vec3(solid);
    }
    let through = pack_shadow_compare(uv, depth, true);
    let coverage = clamp(solid - through, 0.0, 1.0);
    if coverage <= 0.0 {
        return vec3(solid);
    }

    let size = pack_shadow_size();
    let pixel = clamp(vec2<i32>(uv * vec2<f32>(size)), vec2(0), size - 1);
    let caster = pack_shadow_color(pixel);
    let transmission = mix(vec3(1.0), clamp(caster.rgb, vec3(0.0), vec3(1.0)), caster.a);
    return vec3(solid) - (1.0 - transmission) * coverage;
}
