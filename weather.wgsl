override rain_opacity: f32 = 0.25;
override snow_opacity: f32 = 0.75;

fn weather(input: WeatherInput) -> vec4<f32> {
    if frame.dimension != 0 {
        return input.color;
    }

    let daylight = smoothstep(-0.10, 0.50, frame.sun_direction.y);
    let exposure = 0.24 + 0.76 * input.light.y;
    let local = pow(input.light.x, 3.0) * vec3(0.36, 0.19, 0.07);
    let rain_color = vec3(0.49, 0.65, 1.0) * mix(0.033, 0.66, daylight);
    let snow_color = vec3(0.76, 0.84, 1.0) * mix(0.09, 1.6, daylight);
    let sand = (1.0 - smoothstep(0.0, 0.18, frame.downfall)) * smoothstep(0.85, 1.35, frame.temperature);
    let color = mix(select(rain_color, snow_color, input.snow), vec3(0.82, 0.61, 0.34),
        sand * weather_wind() * 0.55);
    return vec4(input.color.rgb * (color * exposure + local + storm_flash() * 0.16),
        input.color.a * select(rain_opacity, snow_opacity, input.snow));
}
