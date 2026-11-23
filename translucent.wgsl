// Horizon supplies the opaque background directly to its forward pass. Rebuild
// a thin dielectric layer only where the selected sample is behind the surface.
fn refract_translucent(input: MaterialInput, normal: vec3<f32>, dielectric: vec3<f32>, ice: bool,
    material_color: vec4<f32>) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(opaque_depth));
    let texel = water_pixel(input.screen, size);
    let depth = textureLoad(opaque_depth, texel, 0);
    if !translucent_refraction || depth <= 0.0 {
        return material_color;
    }

    let position = (frame.view * vec4(input.position, 1.0)).xyz;
    let behind = water_view_position(water_ndc(texel, size), depth);
    let gap = position.z - behind.z;
    if gap <= 0.045 { return material_color; }

    let view = normalize(input.position + vec3(0.000001, 0.0, 0.0));
    let eta = select(1.0 / 1.50, 1.0 / 1.31, ice);
    let refracted = refract(view, normal, eta);
    let bend = (frame.view * vec4(refracted - view, 0.0)).xyz;
    let projected = frame.projection * vec4(behind + bend * min(gap, 2.5) * 0.17, 1.0);
    if projected.w <= 0.01 { return material_color; }

    let offset = water_limit_offset(projected.xy / projected.w / WATER_UV_TO_NDC + 0.5 - input.screen, 0.012);
    let uv = input.screen + offset * water_screen_edge(input.screen, 0.065);
    let refracted_texel = water_pixel(uv, size);
    let shifted_depth = textureLoad(opaque_depth, refracted_texel, 0);
    let shifted = water_view_position(water_ndc(refracted_texel, size), max(shifted_depth, 1e-6));
    let same_layer = abs(shifted.z - behind.z) < max(0.35, 0.045 * gap);
    var transmitted = textureLoad(opaque_color, texel, 0).rgb;
    if shifted_depth > 0.0 && shifted.z < position.z - 0.035 && same_layer {
        transmitted = textureLoad(opaque_color, refracted_texel, 0).rgb;
    }
    transmitted = reflected_aerial_color(transmitted, view, gap, input.light.y);
    let tint = mix(vec3(1.0), clamp(dielectric, vec3(0.08), vec3(1.0)), select(0.28, 0.44, ice));
    let reflection_dir = reflect(view, normal);
    let fresnel = water_fresnel(clamp(dot(-view, normal), 0.0, 1.0), eta);
    let reflection = environment_radiance(reflection_dir) * smoothstep(0.45, 0.95, input.light.y);
    let opacity = clamp(material_color.a, 0.0, 1.0);
    let base = mix(transmitted * tint, material_color.rgb, opacity);
    return vec4(mix(base, reflection + material_color.rgb * 0.38, fresnel), 1.0);
}
