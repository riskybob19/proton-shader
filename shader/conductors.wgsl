// Photon LabPBR conductor fits, credited upstream to Jessie LC.
// The table is in Photon's Rec.2020 working space.
fn conductor_fresnel(code: u32, cosine: f32) -> vec3<f32> {
    var f0: vec3<f32>;
    var f82: vec3<f32>;
    switch code {
        case 230u: { f0 = vec3(0.78, 0.77, 0.74); f82 = vec3(0.74, 0.76, 0.76); }
        case 231u: { f0 = vec3(1.00, 0.90, 0.61); f82 = vec3(1.00, 0.93, 0.73); }
        case 232u: { f0 = vec3(1.00, 0.98, 1.00); f82 = vec3(0.96, 0.97, 0.98); }
        case 233u: { f0 = vec3(0.77, 0.80, 0.79); f82 = vec3(0.74, 0.79, 0.78); }
        case 234u: { f0 = vec3(1.00, 0.89, 0.73); f82 = vec3(1.00, 0.90, 0.80); }
        case 235u: { f0 = vec3(0.79, 0.87, 0.85); f82 = vec3(0.83, 0.80, 0.83); }
        case 236u: { f0 = vec3(0.92, 0.90, 0.83); f82 = vec3(0.89, 0.91, 0.84); }
        default: { f0 = vec3(1.00, 1.00, 0.91); f82 = vec3(1.00, 1.00, 0.95); }
    }

    let m = pow(1.0 - cosine, 5.0);
    let coefficient = 17.6513846 * (f0 - f82) + 8.16666667 * (1.0 - f0);
    let color = clamp(f0 + (1.0 - f0) * m - coefficient * cosine * m * (1.0 - cosine),
        vec3(0.0), vec3(1.0));
    return clamp(REC2020_TO_REC709 * color, vec3(0.0), vec3(1.0));
}
