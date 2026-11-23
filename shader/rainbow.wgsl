override rainbows: bool = true;

fn rainbow_spectrum(x: f32) -> vec3<f32> {
    return clamp(vec3(1.45) - abs(vec3(4.0 * x) - vec3(0.7, 1.95, 3.30)),
        vec3(0.0), vec3(1.0));
}

fn rainbow_color(ray: vec3<f32>, distance: f32) -> vec3<f32> {
    if !rainbows || frame.sun_direction.y <= 0.015 || ray.y <= 0.0 {
        return vec3(0.0);
    }
    let moisture = frame.wetness * mix(0.65, 1.0, weather_humidity()) * (1.0 - frame.rain);
    let angle = acos(clamp(dot(ray, -frame.sun_direction), -1.0, 1.0)) * (180.0 / PHOTON_PI);
    let primary = (44.0 - angle) * 0.25;
    let secondary = (angle - 45.5) / 3.0;
    let a = rainbow_spectrum(primary) * smoothstep(0.0, 0.2, primary) * (1.0 - smoothstep(0.73, 1.0, primary));
    let b = rainbow_spectrum(secondary) * smoothstep(0.0, 0.2, secondary) * (1.0 - smoothstep(0.73, 1.0, secondary));
    let fade = moisture * smoothstep(250.0, 430.0, distance) * smoothstep(0.0, 0.06, ray.y);
    return (a + 0.21 * b) * direct_atmosphere_light() * (0.045 * fade);
}
