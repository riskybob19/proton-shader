fn post(input: PostInput) -> vec4<f32> {
    if !use_volume_composite() {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    let depth = input.depth;
    let position = fog_position(input.uv, depth);
    let ray_length = length(position);
    let distance = fog_distance(ray_length, depth);
    let ray = position / max(ray_length, 1e-6);
    let pixel = floor(input.uv * frame.resolution * 0.5);
    var fog = fog_at(ray, depth, distance, pixel);
    // Volumetric 3D LPV colored light halos in mist/rain/night air around torches, lava, and lanterns.
    if colored_lpv && !use_underwater_fog() && depth > 0.0 && distance > 1.5 && fog_strength > 0.0 {
        let extent = min(distance, 38.0);
        let anchor = lpv_anchor(frame.camera_position);
        let ign = fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
        let j = fract(ign + f32(frame.frame_index & 15u) * 0.61803399);
        var lpv_scatter = vec3(0.0);
        let step_d = extent * (1.0 / 3.0);
        for (var s = 0u; s < 3u; s += 1u) {
            let d = (f32(s) + j) * step_d;
            let wp = frame.camera_position + ray * d;
            lpv_scatter += lpv_sample_trilinear(world_volumes_previous, wp, anchor);
        }
        let night_or_mist = clamp(0.32 + 0.68 * (1.0 - smoothstep(-0.05, 0.25, frame.sun_direction.y)) + 0.55 * frame.rain, 0.0, 1.4);
        fog.scattering += lpv_scatter * (step_d * 0.0028 * night_or_mist * clamp(fog_strength, 0.0, 1.5));
    }
    var clouds = vec4(0.0, 0.0, 0.0, 1.0);
    if !(use_underwater_fog() && depth > 0.0) {
        clouds = clouds_at(ray, depth, ray_length, pixel);
    }
    // When looking down from above the cloud deck, clouds are in front of ground valley fog.
    let cloud_in_front = smoothstep(CLOUD_BASE - 8.0, CLOUD_BASE + 24.0, frame.camera_position.y)
        * smoothstep(0.02, -0.08, ray.y);
    if depth > 0.0 && cloud_in_front > 0.001 {
        let end_dist = min(max(frame.fog_render.y, 16.0), max(frame.render_distance, 32.0));
        let cyl_ratio = clamp(length(position.xz) / end_dist, 0.0, 1.0);
        let rim_t = smoothstep(0.84, 1.0, cyl_ratio);
        let rim_fade = rim_t * rim_t * (3.0 - 2.0 * rim_t) * cloud_in_front;
        fog.scattering *= (1.0 - rim_fade);
    }
    let combined = mix(fog.scattering + fog.transmittance * clouds.rgb,
        clouds.rgb + clouds.a * fog.scattering, cloud_in_front);
    return vec4(combined, clouds.a);
}
