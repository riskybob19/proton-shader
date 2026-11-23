override entity_lighting: bool = true;

fn entity(input: EntityInput) -> Surface {
    let base = input.albedo.rgb * input.tint.rgb * input.shade;
    let gamma = vec3(1.0 / 2.2);
    let alpha = input.albedo.a * input.tint.a;

    // Keep the host's coverage and hurt-overlay semantics on every path.
    let flashed = pow(mix(pow(base, gamma), pow(input.overlay.rgb, gamma), input.overlay.a), vec3(2.2));
    if !entity_lighting || frame.dimension != 0 || input.full_light {
        return surface_fog(vec4(flashed * (input.lightmap + input.colored_light), alpha), input.position);
    }

    let sky = input.light.y;
    let normal = input.normal;
    let cloud = cloud_shadow(input.position);
    let no_l = max(0.0, dot(normal, frame.light_direction));
    var visibility = vec3(1.0);
    if no_l > 0.0 {
        let entity_cell = floor((input.position.xz + frame.camera_position.xz) * 16.0);
        visibility = terrain_shadow(input.position, normal, sky, 0.0, cloud, entity_cell).visibility;
    }
    let albedo = mix(input.albedo.rgb * input.tint.rgb, input.overlay.rgb, input.overlay.a);
    let direct = direct_atmosphere_light() * visibility * (no_l * cloud);
    let ambient = terrain_skylight(normal, sky, 1.0, 0.0, false);
    let local = surface_blocklight(input.position, input.light, 1.0, 1.0);
    let cave = 0.15 * (1.0 - sky * sky) * (1.0 - 0.7 * frame.darkness);
    let indirect = vec3(cave + 1.5 * frame.night_vision);
    return surface_fog(vec4(albedo * (direct + ambient + local + indirect) / PHOTON_PI, alpha), input.position);
}
