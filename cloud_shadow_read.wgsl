fn cloud_ground_map(uv: vec2<f32>) -> vec2<f32> {
    let cell = vec2(0.5) + uv * vec2(CLOUD_MAP_WIDTH - 1.0, CLOUD_MAP_HEIGHT - 1.0)
        + vec2(0.0, ENV_HEIGHT);
    return textureSampleLevel(cloud_aux, scattering_atlas_sampler,
        cell / vec2<f32>(textureDimensions(cloud_aux)), 0.0).rg;
}

fn cloud_coverage(world: vec2<f32>) -> f32 {
    let uv = 0.5 + (world - frame.camera_position.xz) / CLOUD_SHADOW_EXTENT;
    if all(uv >= vec2(0.0)) && all(uv <= vec2(1.0)) {
        return cloud_ground_map(uv).g;
    }
    return cumulus_coverage(world);
}

fn cloud_shadow(position: vec3<f32>) -> f32 {
    if cloud_cover <= 0.0 || frame.light_direction.y <= 0.05 {
        return 1.0;
    }
    if frame.camera_position.y + position.y > CLOUD_BASE - 8.0 {
        return projected_cloud_shadow(position);
    }
    // Project receiver onto the camera-height cloud shadow map along the active light;
    // covers +/-380 blocks at 1 hardware bilinear fetch regardless of vertical elevation.
    let projected = position.xz - frame.light_direction.xz * (position.y / frame.light_direction.y);
    let uv = 0.5 + projected * (1.0 / CLOUD_SHADOW_EXTENT);
    if any(uv <= vec2(0.01)) || any(uv >= vec2(0.99)) {
        return projected_cloud_shadow(position);
    }
    return cloud_ground_map(uv).r;
}

// Fast 1-fetch cloud shadow lookup for volumetric fog raymarching: projects onto the
// precomputed cloud_aux shadow map without falling back to expensive procedural
// projected_cloud_shadow() at high player elevations.
fn cloud_shadow_fast(position: vec3<f32>) -> f32 {
    if cloud_cover <= 0.0 || frame.light_direction.y <= 0.05 {
        return 1.0;
    }
    if frame.camera_position.y + position.y > CLOUD_BASE - 8.0 {
        return 1.0;
    }
    let projected = position.xz - frame.light_direction.xz * (position.y / frame.light_direction.y);
    let uv = 0.5 + projected * (1.0 / CLOUD_SHADOW_EXTENT);
    if any(uv <= vec2(0.01)) || any(uv >= vec2(0.99)) {
        return 1.0;
    }
    return cloud_ground_map(uv).r;
}
