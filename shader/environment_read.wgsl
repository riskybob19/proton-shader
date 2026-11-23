fn sample_environment(direction: vec3<f32>) -> vec4<f32> {
    let uv = cloud_map_texel(direction) / vec2<f32>(textureDimensions(cloud_environment_map));
    return textureSampleLevel(cloud_environment_map, scattering_atlas_sampler, uv, 0.0);
}

fn environment_radiance(direction: vec3<f32>) -> vec3<f32> {
    return sample_environment(direction).rgb;
}
