override color_saturation: f32 = 1.0;
override exposure: f32 = 1.0;
override color_grading: u32 = 1u;

// OpenColorIO ACES 2 output transforms baked from scene-linear Rec.709.
fn aces_slice(grid: vec3<f32>, blue: f32, row: f32) -> vec3<f32> {
    let tile_row = floor(blue / 7.0);
    let tile = vec2(blue - 7.0 * tile_row, row * 7.0 + tile_row);
    let pixel = tile * 49.0 + grid.xy + 0.5;
    let encoded = textureSampleLevel(aces_output, aces_output_sampler, pixel / vec2(343.0, 1029.0), 0.0).rgb;
    return encoded * encoded;
}

fn aces_peak(grid: vec3<f32>, row: f32, peak: f32) -> vec3<f32> {
    let blue = floor(grid.z);
    return peak * mix(aces_slice(grid, blue, row), aces_slice(grid, min(blue + 1.0, 48.0), row), fract(grid.z));
}

fn photon_gain(x: vec3<f32>, k: f32) -> vec3<f32> {
    let hi = step(vec3(0.5), x);
    let a = 0.5 * pow(max(2.0 * mix(x, 1.0 - x, hi), vec3(0.0)), vec3(k));
    return mix(a, 1.0 - a, hi);
}

// Upstream Photon default tonemapper: Timothy Lottes (GDC 2016) filmic curve
// (a = 1.5 toe contrast, d = 0.91 shoulder, hdr_max = 8.0, mid_in = 0.26, mid_out = 0.32).
fn tonemap_lottes(rgb: vec3<f32>) -> vec3<f32> {
    let c_in = max(rgb, vec3(0.0));
    let p_a = pow(c_in, vec3(1.5));
    let p_ad = pow(c_in, vec3(1.365));
    return p_a / (p_ad * 1.312060 + 0.205606);
}

fn photon_grade_output(rgb: vec3<f32>) -> vec3<f32> {
    let p = sqrt(clamp(rgb, vec3(0.0), vec3(1.0)));
    // Subtle teal saturation lift (upstream Photon GRADE_TEAL_SAT_BOOST = 0.10) + S-curve gain(1.06).
    let luma = dot(p, frame.luminance);
    let teal_weight = clamp((min(p.g, p.b) - p.r) * 3.0, 0.0, 1.0);
    let boosted = clamp(mix(vec3(luma), p, 1.0 + 0.12 * teal_weight), vec3(0.0), vec3(1.0));
    let curved = photon_gain(boosted, 1.06);
    return curved * curved;
}

// White-preserving painterly RDR2 film-stock matrix (column-major; each row sums to 1.0):
// shifts neon Minecraft greens toward grounded olive/pine, enriches warm earth/ochre,
// and deepens midday skies toward natural Hudson-River-School cerulean.
const RDR2_FILM_MATRIX = mat3x3<f32>(
    vec3( 0.975,  0.032, -0.007),
    vec3( 0.048,  0.946,  0.006),
    vec3(-0.023,  0.022,  1.001),
);

fn rdr2_pre_grade(rgb: vec3<f32>) -> vec3<f32> {
    var c = max(RDR2_FILM_MATRIX * rgb, vec3(0.0));
    // Creamy filmic highlight desaturation (AgX / Kodak 2383 style) so bright sun,
    // clouds, and sandstone roll off toward warm ivory instead of clipping harsh yellow.
    let luma = dot(c, frame.luminance);
    let hi_desat = smoothstep(0.95, 4.5, luma) * 0.32;
    let warm_ivory = vec3(luma) * vec3(1.03, 1.00, 0.94);
    c = mix(c, warm_ivory, hi_desat);
    return max(c, vec3(0.0));
}

// RDR2 filmic curve in Rec.2020: rich velvety toe with creamy rolled-off highlights.
fn tonemap_rdr2_filmic(rgb: vec3<f32>) -> vec3<f32> {
    let c_in = max(rgb, vec3(0.0));
    let p_a = pow(c_in, vec3(1.42));
    let p_ad = pow(c_in, vec3(1.315));
    return p_a / (p_ad * 1.245 + 0.215);
}

// Kodak 2383 print-film split-tone & painterly green restraint (RDR2 palette):
// cool slate-blue shadows, warm cream-amber highlights (boosted during golden hour),
// velvety lifted black toe, and restrained neon-green foliage saturation.
fn rdr2_grade_output(rgb: vec3<f32>) -> vec3<f32> {
    var p = sqrt(clamp(rgb, vec3(0.0), vec3(1.0)));
    let luma = dot(p, frame.luminance);

    // Restrain overly electric Minecraft lime-greens toward natural painterly foliage.
    let green_excess = clamp((p.g - max(p.r, p.b)) * 2.4, 0.0, 1.0);
    p = mix(p, vec3(luma), green_excess * 0.13);

    // Split-tone: cool slate shadows + warm cream-gold sunlit highlights (extra warm at golden hour).
    let golden = select(0.0,
        smoothstep(-0.04, 0.16, frame.sun_direction.y) * (1.0 - smoothstep(0.18, 0.44, frame.sun_direction.y))
        * (1.0 - 0.65 * frame.rain),
        frame.dimension == 0);
    let shadow_w = 1.0 - smoothstep(0.0, 0.48, luma);
    let hi_w = smoothstep(0.42, 0.95, luma);
    let shadow_tint = vec3(0.955, 0.985, 1.038);
    let hi_tint = mix(vec3(1.028, 1.005, 0.962), vec3(1.068, 1.012, 0.918), golden);
    p *= mix(vec3(1.0), shadow_tint, shadow_w * 0.72);
    p *= mix(vec3(1.0), hi_tint, hi_w * 0.85);

    // Subtle filmic S-curve + velvety lifted black toe so forest shadows stay rich, never crushed.
    p = clamp(p, vec3(0.0), vec3(1.0));
    let curved = photon_gain(p, 1.08);
    let lifted = mix(vec3(0.004), vec3(1.0), curved);
    return lifted * lifted;
}

fn tonemap(input: TonemapInput) -> vec3<f32> {
    let window = input.uv * (1.0 - input.uv);
    let darkness_pulse = select(0.0, 1.0 - photon_dampen(abs(cos(2.0 * frame.time))), frame.darkness > 0.0);
    let cave = 1.0 - smoothstep(0.0, 0.28, frame.eye_light_smooth.y);
    let vignette_exponent = 0.08 + 0.10 * cave + 0.30 * frame.blindness
        + 0.20 * darkness_pulse * frame.darkness;
    let vignette = pow(clamp(16.0 * window.x * window.y, 0.0, 1.0), vignette_exponent);
    var scene = max(input.color * (exposure * vignette), vec3(0.0));

    // Upstream Photon Purkinje shift in moonlight / dark outdoor scenes.
    let night = 1.0 - smoothstep(-0.12, -0.06, frame.sun_direction.y);
    let purkinje_i = 0.045 * night * (0.3 + 0.7 * frame.eye_light_smooth.y * frame.eye_light_smooth.y);
    if purkinje_i > 0.0005 && frame.dimension == 0 {
        let rod = dot(scene, vec3(0.05, 0.58, 0.37));
        let shift = exp2(-rod / purkinje_i);
        scene = mix(scene, rod * vec3(0.55, 0.74, 1.0), shift * 0.85);
    }

    let y = dot(scene, frame.luminance);
    var graded = max(vec3(0.0), mix(vec3(y), scene, color_saturation));
    let rdr2 = color_grading == 1u;
    if rdr2 {
        graded = rdr2_pre_grade(graded);
    }
    let peak = clamp(frame.headroom, 1.0, 10.0);
    if peak <= 1.0 {
        // Triangular-PDF IGN dithering eliminates 8-bit banding in dark caves and twilight skies.
        let px = floor(input.uv * frame.resolution);
        let n1 = fract(52.9829189 * fract(dot(px, vec2(0.06711056, 0.00583715)))
            + f32(frame.frame_index & 15u) * 0.61803399);
        let n2 = fract(n1 * 1.61803399 + 0.3719);
        let dither = (n1 + n2 - 1.0) * (0.75 / 255.0);
        if rdr2 {
            let rec2020 = max(REC709_TO_REC2020 * (graded * 0.86), vec3(0.0));
            let mapped = clamp(REC2020_TO_REC709 * tonemap_rdr2_filmic(rec2020), vec3(0.0), vec3(1.0));
            return clamp(rdr2_grade_output(mapped) + vec3(dither), vec3(0.0), vec3(1.0));
        }
        // Match upstream Photon c19_color_grading: Rec.2020 grade_input(0.83) -> tonemap_lottes -> Rec.709 -> grade_output.
        let rec2020 = max(REC709_TO_REC2020 * (graded * 0.83), vec3(0.0));
        let mapped = clamp(REC2020_TO_REC709 * tonemap_lottes(rec2020), vec3(0.0), vec3(1.0));
        return clamp(photon_grade_output(mapped) + vec3(dither), vec3(0.0), vec3(1.0));
    }
    // HDR display path: OpenColorIO ACES 2 output transforms.
    let highlight_gain = 1.0 + 0.189207 * smoothstep(0.55, 2.0, y);
    let grid = min(log2(1.0 + 32.0 * graded * highlight_gain) * (48.0 / 14.0), vec3(48.0));
    let medium = aces_peak(grid, 1.0, 3.0);
    if peak <= 3.0 {
        return mix(aces_peak(grid, 0.0, 1.0), medium, (peak - 1.0) / 2.0);
    }
    return mix(medium, aces_peak(grid, 2.0, 10.0), (peak - 3.0) / 7.0);
}
