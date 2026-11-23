override tree_scattering: f32 = 1.25;
override neural_sss: bool = true;
override bump_strength: f32 = 0.50;
override labpbr_pom: bool = true;

// Path-traced neural canopy & snow multi-bounce SSS MLP (8 -> 12 -> 12 -> 3, ReLU).
// Trained on 24,000 Monte Carlo volumetric states (384 paths x 16 bounces = 9.2M rays).
// Validation R2=0.9637, energy ratio vs path tracer=0.9935.
// Zero-spill compile-time mat4x4 unrolled MLP (eliminates Naga/SPIR-V local array spill).
fn neural_subsurface_scatter(tau_rgb: vec3<f32>, forward_cos: f32, signed_no_l: f32,
    g_aniso: f32, mesh_ao: f32) -> vec3<f32> {
    let tau_mean = dot(tau_rgb, vec3(0.333333));
    let e_tau = exp(-0.5 * tau_rgb);
    let x0 = vec4(e_tau.r, e_tau.g, e_tau.b, 1.0 / (1.0 + 0.35 * tau_mean));
    let x1 = vec4(forward_cos, signed_no_l, g_aniso, clamp(mesh_ao, 0.25, 1.0));
    let h1_0 = max(vec4(-0.00649, 1.00516, -0.02101, -0.04420) + mat4x4<f32>(
        vec4(0.53448, -0.32946, -0.52977, -0.09054),
        vec4(0.13515, -0.30876, 0.94403, -0.38421),
        vec4(0.34970, -0.24532, 0.25915, 0.27893),
        vec4(0.26165, 0.20655, 0.56204, 0.28444),
    ) * x0 + mat4x4<f32>(
        vec4(0.01576, -0.22384, 0.08548, -0.48636),
        vec4(-0.00176, 0.00058, -0.00953, -0.00582),
        vec4(-0.55198, -0.61586, 0.12576, 0.55509),
        vec4(0.05999, 0.00580, 0.17681, 0.15977),
    ) * x1, vec4(0.0));
    let h1_1 = max(vec4(-0.19398, -0.12308, 0.46083, -0.14674) + mat4x4<f32>(
        vec4(-0.44328, -0.05349, -0.24518, 1.13307),
        vec4(-0.08311, -1.60314, -0.09775, 0.23988),
        vec4(0.88467, 0.74098, -0.51408, -0.32740),
        vec4(0.23568, 1.09326, 0.68950, 0.43851),
    ) * x0 + mat4x4<f32>(
        vec4(0.10054, -0.00355, -0.78811, 0.61604),
        vec4(0.00315, 0.00106, 0.00454, -0.01272),
        vec4(0.16123, -0.03563, 0.16363, 0.45202),
        vec4(-0.06673, -0.00537, 0.03723, 0.18505),
    ) * x1, vec4(0.0));
    let h1_2 = max(vec4(-0.07657, 0.44071, -1.66894, -0.15806) + mat4x4<f32>(
        vec4(0.41335, 0.46134, 0.02572, -1.23756),
        vec4(0.37458, -1.06218, 0.00561, -0.02862),
        vec4(-0.97902, 0.14048, 0.03872, 1.36478),
        vec4(-0.21961, 0.43825, 0.09552, 0.37738),
    ) * x0 + mat4x4<f32>(
        vec4(0.02877, -0.62442, 1.38319, 0.47195),
        vec4(-0.05304, 0.00081, 0.00348, -0.00892),
        vec4(-0.37289, -0.01691, 0.34005, 0.69761),
        vec4(0.03480, 0.02692, -0.00312, 0.11668),
    ) * x1, vec4(0.0));
    let h2_0 = max(vec4(0.29277, -0.12997, 0.09200, -0.33817) + mat4x4<f32>(
        vec4(0.66770, 0.18550, 0.28301, -0.29458),
        vec4(0.60958, -0.05202, -0.01024, -0.23610),
        vec4(-0.18531, -0.48037, 0.43091, 0.04291),
        vec4(0.71893, -0.20777, 0.01472, -0.59021),
    ) * h1_0 + mat4x4<f32>(
        vec4(-0.26426, -0.51185, -0.22774, -0.31564),
        vec4(0.83435, -0.12714, 0.76188, 0.65202),
        vec4(-0.53210, -0.31604, 0.22220, 0.50776),
        vec4(-0.08239, -0.13843, 0.18423, 0.47465),
    ) * h1_1 + mat4x4<f32>(
        vec4(0.73548, -0.45068, 0.27213, -0.70296),
        vec4(0.01616, 0.51241, 0.13251, 0.12959),
        vec4(-3.45795, -0.19651, 1.48150, 2.42919),
        vec4(0.22118, -0.09216, 0.20650, -0.16098),
    ) * h1_2, vec4(0.0));
    let h2_1 = max(vec4(-0.01251, -0.19897, -0.15841, -0.23309) + mat4x4<f32>(
        vec4(-0.04423, -0.10487, -0.29967, 0.47598),
        vec4(0.26468, -0.41107, -0.88935, 0.26843),
        vec4(-0.29417, -0.32277, 0.37628, -0.04166),
        vec4(0.29236, -0.37353, -0.52297, -0.62011),
    ) * h1_0 + mat4x4<f32>(
        vec4(-0.05064, -0.40167, 0.32621, -1.21582),
        vec4(0.37812, 0.27740, 0.56162, -0.32146),
        vec4(-0.17848, 0.35721, 0.30615, -0.13813),
        vec4(0.47392, 0.53088, -0.10279, 0.11406),
    ) * h1_1 + mat4x4<f32>(
        vec4(-0.14123, -0.08658, 0.32030, -0.26391),
        vec4(-0.05270, 0.33599, 0.57429, -0.66484),
        vec4(-0.52612, 3.58642, 4.09685, -0.31359),
        vec4(-0.53100, -0.06232, 0.45751, -0.70780),
    ) * h1_2, vec4(0.0));
    let h2_2 = max(vec4(-0.12644, -0.82843, 0.23980, 0.04363) + mat4x4<f32>(
        vec4(0.04341, 0.93157, -0.35424, 0.24218),
        vec4(-1.00500, -0.44210, -0.91261, -0.02890),
        vec4(-0.16966, -0.17014, 0.54493, 0.28352),
        vec4(0.01711, 0.41377, -0.43296, -0.11813),
    ) * h1_0 + mat4x4<f32>(
        vec4(-0.26093, 0.73063, -0.47058, -0.29812),
        vec4(-0.30217, 0.26061, -2.27597, 0.06795),
        vec4(-0.16378, -0.20312, 0.25839, 0.14429),
        vec4(-0.05476, -0.40564, -0.13193, -0.29265),
    ) * h1_1 + mat4x4<f32>(
        vec4(0.21068, 0.02138, 0.01981, -0.14346),
        vec4(-0.31674, 0.19359, 0.16202, 0.00852),
        vec4(-0.04768, 2.00730, 1.49343, 1.90722),
        vec4(0.31854, 0.45815, 0.00769, 0.43334),
    ) * h1_2, vec4(0.0));
    let out_log = vec3(0.08150, 0.17801, -0.04877) + mat4x3<f32>(
        vec3(-0.25912, -0.81854, -0.13022),
        vec3(0.55066, 0.19996, -0.00465),
        vec3(0.58538, 0.42405, 0.03980),
        vec3(-0.40786, 0.70801, 0.76958),
    ) * h2_0 + mat4x3<f32>(
        vec3(-0.70690, 0.25886, 0.20910),
        vec3(1.75497, -0.48320, -0.30404),
        vec3(0.22542, 0.22501, 0.90176),
        vec3(-0.07782, -0.07308, 0.09889),
    ) * h2_1 + mat4x3<f32>(
        vec3(0.22214, 0.70150, -0.61020),
        vec3(0.09293, -0.10676, 1.07922),
        vec3(-0.02395, 0.66718, -0.36062),
        vec3(-1.04758, 0.50197, 0.46388),
    ) * h2_2;
    return max(exp(max(out_log, vec3(0.0))) - vec3(1.0), vec3(0.0));
}

fn pbr_normal_at(uv: vec2<f32>) -> vec4<f32> {
    return block_normal_map(uv);
}

fn mapped_normal(input: MaterialInput, tangent_normal: vec2<f32>) -> vec3<f32> {
    let tangent = input.tangent.xyz;
    let bitangent = cross(input.normal, tangent) * input.tangent.w;
    let up = sqrt(max(0.0, 1.0 - dot(tangent_normal, tangent_normal)));
    return normalize(tangent * tangent_normal.x + bitangent * tangent_normal.y + input.normal * up);
}

// Fast rational Kubelka-Munk slab (no exp() calls) for diffuse reflection and transmission.
struct FoliageOptics { reflected: vec3<f32>, transmitted: vec3<f32> }

fn foliage_optics(albedo: vec3<f32>, thickness: f32) -> FoliageOptics {
    let r = clamp(albedo, vec3(0.001), vec3(0.999));
    let x = 0.5 * (1.0 / r - r) * thickness;
    let attenuation = 1.0 / (1.0 + x + 0.5 * x * x);
    let a2 = attenuation * attenuation;
    let r2 = r * r;
    let inv_denom = 1.0 / (1.0 - r2 * a2);
    return FoliageOptics(r * (1.0 - a2) * inv_denom, attenuation * (1.0 - r2) * inv_denom);
}

fn sss_hg_phase(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let d = max(1.0 + g2 - 2.0 * g * cos_theta, 1e-4);
    return (1.0 - g2) / (4.0 * PHOTON_PI * d * sqrt(d));
}

// Physical canopy thickness estimator for foliage outside the shadow map
// (blended smoothly across 104..128 blocks): combines vertex skylight, back-facing
// crown depth, and vertex AO so distant trees never drop to 0-thickness neon glow.
fn estimated_sss_thickness(sky: f32, mesh_ao: f32, signed_no_l: f32) -> f32 {
    let sky_depth = max(1.0 - sky, 0.0) * 8.0;
    let back_depth = max(-signed_no_l, 0.0) * 1.45 + (1.0 - max(signed_no_l, 0.0)) * 0.55;
    let cavity_depth = max(1.0 - mesh_ao, 0.0) * 2.4;
    return min(sky_depth + back_depth + cavity_depth, 4.2);
}

struct SssProfile {
    tint: vec3<f32>,
    absorption: vec3<f32>,
    density_scale: f32,
    amount_scale: f32,
}

// Per-pixel botanical pigment optics for Minecraft plants (separates green stems
// from red/pink/blue/white/golden flower petals within the same sprite).
fn plant_sss_profile(albedo: vec3<f32>) -> SssProfile {
    let c_max = max(albedo.r, max(albedo.g, albedo.b));
    let c_min = min(albedo.r, min(albedo.g, albedo.b));
    let sat = (c_max - c_min) / max(c_max, 1e-4);
    let luma = dot(albedo, frame.luminance);
    // White / cream petals (oxeye daisy, lily of the valley, white tulip, azure bluet):
    // air-filled unpigmented parenchyma transmits warm luminous pearl-white.
    if sat < 0.24 && luma > 0.28 {
        return SssProfile(vec3(1.12, 1.08, 0.98), vec3(0.75, 0.78, 0.88), 0.50, 1.0);
    }
    // Red / pink / magenta petals (poppy, rose bush, red/pink tulip, peony, allium):
    // thin anthocyanin petals absorb green and transmit vivid ruby/rose light.
    if albedo.r > albedo.g * 1.22 && albedo.r > albedo.b * 1.05 {
        return SssProfile(vec3(1.38, 0.72, 0.85), vec3(0.55, 1.35, 0.95), 0.55, 1.0);
    }
    // Blue / violet petals (cornflower, blue orchid, lilac):
    // delphinidin anthocyanin petals absorb red and transmit luminous sky-blue/violet.
    if albedo.b > albedo.r * 1.12 && albedo.b >= albedo.g * 0.92 {
        return SssProfile(vec3(0.78, 0.96, 1.38), vec3(1.35, 0.92, 0.55), 0.55, 1.0);
    }
    // Golden / yellow petals & ripe crops (dandelion, sunflower, ripe wheat):
    // carotenoids absorb blue and transmit brilliant golden-amber light.
    if albedo.r > albedo.b * 1.55 && albedo.g > albedo.b * 1.35 && albedo.r >= albedo.g * 0.88 {
        return SssProfile(vec3(1.38, 1.18, 0.20), vec3(0.65, 0.75, 1.95), 0.65, 1.0);
    }
    // Green grass blades, ferns, saplings, flower stems (chlorophyll a/b + carotenoids):
    // absorbs blue & deep red, peaks warmly at yellow-green (550-575 nm).
    return SssProfile(vec3(1.34, 1.12, 0.22), vec3(0.82, 0.70, 1.75), 0.78, 1.0);
}

// Botanical optical properties per Minecraft foliage class (PROSPECT leaf model):
// chlorophyll + carotenoids strongly absorb blue (430-470nm) and transmit warm
// yellow-green (550-575nm), while cherry & flower petals transmit anthocyanin/carotenoid light.
fn sss_profile(tags: u32, leaves: bool, plant: bool, albedo: vec3<f32>) -> SssProfile {
    if leaves {
        if (tags & 8192u) != 0u {
            // Cherry blossoms (Prunus x yedoensis, van der Kooi et al. 2016):
            // dense floral + bract cluster LAD u_L = 3.1 m^2/m^3, Ross-Nilson Omega*G = 0.44
            // -> sigma_geom = 1.36 m^-1; cyanidin-3-glucoside (C3G) absorbs green (510-535nm).
            return SssProfile(vec3(1.18, 0.74, 0.92), vec3(0.38, 1.18, 0.76), 1.36, 0.90);
        }
        if (tags & 16384u) != 0u {
            // Pale oak: dense sclerophyll/lichen crown LAD u_L = 3.8 m^2/m^3 -> sigma_geom = 1.65 m^-1.
            return SssProfile(vec3(1.05, 1.02, 0.90), vec3(0.85, 0.84, 0.92), 1.65, 0.75);
        }
        if (tags & 4096u) != 0u {
            // Spruce & dark oak (Smolander & Stenberg 2003, Ross 1981):
            // dense conifer needle shoot voxel LAD u_L = 4.6 m^2/m^3, shoot clumping Omega*G = 0.43
            // -> sigma_geom = 1.98 m^-1; high chlorophyll a+b (65 ug/cm^2) & leaf thickness N = 2.2.
            return SssProfile(vec3(1.12, 1.10, 0.32), vec3(0.89, 0.78, 1.48), 1.98, 0.85);
        }
        // Broadleaf deciduous (Quercus/Betula - oak, birch, jungle, acacia, mangrove, azalea):
        // PROSPECT (Jacquemoud & Baret 1990) C_ab = 45 ug/cm^2, N = 1.55, omega_0 = (0.18, 0.34, 0.07)
        // + dense crown voxel LAD u_L = 3.5 m^2/m^3, Ross G = 0.50, clumping Omega = 0.86 -> sigma_geom = 1.50 m^-1.
        return SssProfile(vec3(1.28, 1.08, 0.24), vec3(0.82, 0.66, 1.52), 1.50, 1.0);
    }
    if plant {
        return plant_sss_profile(albedo);
    }
    if (tags & 256u) != 0u {
        // Grass blocks (turf): dense mat of upright chlorophyll blades over soil;
        // warm golden-chartreuse grazing/backlit glow with crisp shadow falloff.
        return SssProfile(vec3(1.42, 1.15, 0.22), vec3(0.80, 0.68, 1.85), 0.95, 1.0);
    }
    if (tags & 32768u) != 0u {
        // Snow & powder snow: cool glacial blue-white subsurface scatter.
        return SssProfile(vec3(0.92, 0.97, 1.08), vec3(1.18, 1.0, 0.80), 1.6, 0.75);
    }
    // Ice, glass, LabPBR subsurface solids: neutral pigment-driven transmission.
    return SssProfile(vec3(1.0), vec3(1.0), 1.0, 1.0);
}

// Exact 3D volumetric path length (meters) through a 1x1x1m voxel block:
// For leaf blocks, treats the 1x1x1m cube as a dense 3D foliage volume and computes the
// exact 3D ray-box slab chord lengths along both the solar entry ray (+L) and viewer sightline (-V),
// so grazed block rims are thin (0.25-0.40m) while the 3D block core is dense and dark (1.2-1.7m).
fn voxel_micro_thickness(position: vec3<f32>, normal: vec3<f32>, tags: u32,
    leaves: bool, plant: bool, signed_no_l: f32, mesh_ao: f32) -> f32 {
    let world = position + frame.camera_position;
    let turf = (tags & 256u) != 0u;
    if turf {
        let local = fract(world + 0.0005);
        if abs(normal.y) < 0.5 {
            // Grass block side overhang: light enters from the sunlit top face and
            // attenuates rapidly down the hanging grass fringe over opaque dirt.
            let y_down = 1.0 - local.y;
            let back = clamp(-signed_no_l, 0.0, 1.0);
            return y_down * (2.6 + 1.4 * back) + max(1.0 - mesh_ao, 0.0) * 0.45 + 0.12 * back;
        }
        // Top grass mat: thinner at sunlit grazing tips, deeper in block crevices.
        return max(1.0 - mesh_ao, 0.0) * 0.65 + (1.0 - max(signed_no_l, 0.0)) * 0.35;
    }
    if plant {
        let local = fract(world + 0.0005);
        // Thin grass/flower blade: thicker near the dense root cluster, paper-thin at tips.
        return (1.0 - local.y) * 0.45 + max(1.0 - mesh_ao, 0.0) * 0.35;
    }
    if leaves {
        // Beyond 48m, individual 1m leaf blocks are sub-5px; use the analytical mean dense-block
        // chord instead of running two 3D ray-box slab intersections on distant forests.
        let pos_d2 = dot(position, position);
        if pos_d2 > 2304.0 {
            return 0.78 + max(1.0 - mesh_ao, 0.0) * 0.65;
        }
        // Step 2mm inside the 1x1x1m leaf cube and intersect the 3D unit box [0, 1]^3
        // along both the solar ray (+light_direction) and the viewing ray (normalize(position)).
        let u_box = clamp(fract(world - normal * 0.002), vec3(0.001), vec3(0.999));
        let l_dir = frame.light_direction;
        let l_inv = select(-1.0 / max(abs(l_dir), vec3(1e-4)), 1.0 / max(abs(l_dir), vec3(1e-4)), l_dir >= vec3(0.0));
        let t_sun = (select(vec3(0.0), vec3(1.0), l_dir >= vec3(0.0)) - u_box) * l_inv;
        let s_light = clamp(min(t_sun.x, min(t_sun.y, t_sun.z)), 0.0, 1.732);

        let v_in = position * inverseSqrt(max(pos_d2, 1e-8));
        let v_inv = select(-1.0 / max(abs(v_in), vec3(1e-4)), 1.0 / max(abs(v_in), vec3(1e-4)), v_in >= vec3(0.0));
        let t_eye = (select(vec3(0.0), vec3(1.0), v_in >= vec3(0.0)) - u_box) * v_inv;
        let s_view = clamp(min(t_eye.x, min(t_eye.y, t_eye.z)), 0.0, 1.732);

        // Total intra-voxel 3D path through the dense leaf cluster: solar entry chord +
        // mean sightline in-scatter depth (0.48 * s_view) + adjacent-block cluster density.
        return s_light + 0.48 * s_view + max(1.0 - mesh_ao, 0.0) * 0.70;
    }
    let local = fract(world + 0.0005);
    let dist_from_center = abs(local - 0.5) * 2.0;
    let an = abs(normal);
    let edge_axes = dist_from_center * (1.0 - step(vec3(0.5), an));
    let edge_prox = max(edge_axes.x, max(edge_axes.y, edge_axes.z));
    let center_weight = 1.0 - edge_prox * edge_prox;
    let back = max(-signed_no_l, 0.0);
    // Ice, snow, and translucent solids: crisp corner translucency, deep thick cores.
    return back * (0.15 + 0.60 * center_weight) + max(1.0 - mesh_ao, 0.0) * 0.40;
}

// Physically-based PROSPECT + Ross-Goudriaan (1977/1981) 3D volumetric leaf-block & subsurface
// radiative transfer combining 3D ray-box intra-block chord depth and inter-block shadow depth.
fn terrain_subsurface(albedo: vec3<f32>, amount: f32, sheen_amount: f32,
    profile: SssProfile, shadow: TerrainShadow, position: vec3<f32>, normal: vec3<f32>,
    tags: u32, leaves: bool, plant: bool, signed_no_l: f32, light_view: f32,
    ao: f32, sky: f32, mesh_ao: f32) -> vec3<f32> {
    let eff_amount = amount * profile.amount_scale;
    if eff_amount <= 1e-4 {
        return vec3(0.0);
    }
    let est_macro = estimated_sss_thickness(sky, mesh_ao, signed_no_l);
    let fade_w = clamp(shadow.fade, 0.0, 1.0);
    let raw_macro = mix(max(shadow.thickness, 0.0), est_macro, fade_w);
    let micro_depth = voxel_micro_thickness(position, normal, tags, leaves, plant, signed_no_l, mesh_ao);
    // For 3D dense leaf blocks, raw_macro (from the 3D shadow map) already includes the solar entry
    // chord s_light when shadowed; max(raw_macro, micro_depth) + 0.35 * min(raw_macro, micro_depth)
    // seamlessly combines the inter-block crown path with the 3D intra-block ray-box volume chord.
    let eff_thickness = select(
        raw_macro + micro_depth,
        max(raw_macro, micro_depth) + 0.35 * min(raw_macro, micro_depth),
        leaves,
    );

    let luma = dot(albedo, frame.luminance) + 1e-4;
    let tint = clamp(albedo * inverseSqrt(luma), vec3(0.0), vec3(1.0));
    // For leaves, profile.density_scale is the research Ross-Nilson geometric extinction
    // sigma_geom = Omega * G * u_L (1.36..1.98 m^-1 for LAD u_L = 3.1..4.6 m^2/m^3);
    // multiplier 3.45 converts PROSPECT leaf absorptance + LAD into base-2 optical depth per meter.
    let density_mult = select(8.5, 3.45, leaves);
    let coeff = (1.0 - select(0.75, 0.65, leaves) * tint) * profile.absorption * ((density_mult * profile.density_scale) / eff_amount);
    let forward_cos = -light_view;
    let trans_shallow = exp2(-coeff * eff_thickness);
    let vis_luma = clamp(dot(shadow.visibility, vec3(0.333333)), 0.0, 1.0);
    let is_sakura = leaves && (tags & 8192u) != 0u;
    let is_turf = (tags & 256u) != 0u;

    // Goudriaan (1977) / SAIL first-collision volumetric conversion factor p_col = 1 - exp(-sigma_geom * s)
    // and transmitted-hemisphere directional gate: at s -> 0 on sun-facing outer surfaces, light is
    // uncollided direct reflection (already in diffuse), while volumetric multi-scattered light builds
    // up across the 3D interior of the dense leaf block and exits on backlit/grazing edges.
    var volume_gate = 1.0;
    if leaves {
        let p_col = 1.0 - exp(-profile.density_scale * eff_thickness);
        let trans_dir = mix(0.08, 1.0, smoothstep(-0.35, 0.55, -signed_no_l))
            * mix(0.22, 1.0, smoothstep(-0.20, 0.85, forward_cos));
        volume_gate = p_col * trans_dir;
    }

    var sss: vec3<f32>;
    let pos_dist_sq = dot(position, position);
    let dist_damp = select(1.0, mix(1.0, 0.72, fade_w), leaves);
    let phase = mix(1.0 / (4.0 * PHOTON_PI), sss_hg_phase(forward_cos, select(0.62, 0.40, is_sakura)), 0.52);
    if neural_sss && !is_turf && shadow.fade < 0.99 && pos_dist_sq < 3136.0 {
        let tau_rgb = clamp(coeff * (eff_thickness * 0.6931472), vec3(0.01), vec3(22.0));
        let is_glacial = (tags & 32768u) != 0u || (tags & 8u) != 0u;
        let g_aniso = select(select(select(0.52, 0.58, leaves || plant), 0.34, is_sakura), 0.78, is_glacial);
        let neural = neural_subsurface_scatter(tau_rgb, forward_cos, signed_no_l, g_aniso, mesh_ao);
        // Normalize the 4*PI-integrated neural MLP exitance by the solid-angle phase factor
        // so neural_sss matches the physical PROSPECT+SAIL Beer-Lambert radiance 1:1 without over-brightening.
        let neural_norm = select(0.32, 0.24 * (0.35 + 2.50 * phase) * volume_gate, leaves);
        let neural_sss_val = neural * (neural_norm * dist_damp * photon_dampen(eff_amount) * profile.tint);
        if pos_dist_sq > 2304.0 {
            let trans_deep = exp2(-coeff * (0.48 * eff_thickness));
            let transmittance = mix(trans_deep * 0.30, trans_shallow, 0.72);
            let analytic_sss = (1.95 * PHOTON_PI * phase * volume_gate * dist_damp * photon_dampen(eff_amount)) * transmittance * profile.tint;
            let n_fade = 1.0 - smoothstep(2304.0, 3136.0, pos_dist_sq);
            sss = mix(analytic_sss, neural_sss_val, n_fade);
        } else {
            sss = neural_sss_val;
        }
    } else {
        // Energy-matched dual-lobe PROSPECT+SAIL Beer-Lambert diffusion path.
        let trans_deep = exp2(-coeff * (0.48 * eff_thickness));
        let transmittance = mix(trans_deep * 0.30, trans_shallow, 0.72);
        sss = (1.95 * PHOTON_PI * phase * volume_gate * dist_damp * photon_dampen(eff_amount)) * transmittance * profile.tint;
    }
    if sheen_amount > 0.0 {
        let sheen_ramp = clamp((forward_cos + 0.65) / 0.85, 0.0, 1.0);
        let sheen_gate = select(1.0, volume_gate, leaves);
        let sheen = (0.75 * profile.tint) * trans_shallow * sss_hg_phase(forward_cos, 0.55) * (sheen_ramp * sheen_gate);
        sss += sheen * sheen_amount;
    }
    let ao2 = ao * ao;
    let sss_ao = select(ao2, mix(0.25, 1.0, ao2), leaves || plant);
    var vis_scale = 1.0;
    if is_turf {
        let dist_fade = 1.0 - smoothstep(2304.0, 10404.0, dot(position, position));
        vis_scale = mix(0.25, 1.0, vis_luma) * mix(0.22, 1.0, dist_fade);
    } else if leaves || plant {
        vis_scale = mix(1.0, mix(0.18, 1.0, vis_luma), fade_w);
    }
    return sss * (sss_ao * vis_scale);
}

// Danny Chan (SIGGRAPH 2018, Call of Duty: WWII) analytical directional micro-shadowing:
// derives a visibility cone aperture from local AO so crevices and brick mortar fall into
// directional shadow at low sun angles while flat sunlit surfaces stay clean.
fn micro_shadow(no_l: f32, visibility: f32) -> f32 {
    let aperture = inverseSqrt(max(1.0 - clamp(visibility, 0.0, 0.999), 1e-4));
    let ms = clamp(no_l * aperture, 0.0, 1.0);
    return ms * ms;
}

// Portsmouth et al. (EGSR 2025) Energy-Preserving Oren-Nayar (EON) rough diffuse.
// Returns a dimensionless multiplier relative to Lambertian (albedo / PI) so it
// preserves exact energy conservation while adding tactile retro-reflection and
// multi-bounce chroma on rough stone, dirt, sand, wood, and terracotta.
fn eon_diffuse_factor(albedo: vec3<f32>, roughness: f32, no_l: f32, no_v: f32, lo_v: f32) -> vec3<f32> {
    let r = clamp(roughness, 0.0, 1.0);
    if r <= 0.02 {
        return vec3(1.0);
    }
    let mu_i = clamp(no_l, 0.01, 1.0);
    let mu_o = clamp(no_v, 0.01, 1.0);
    let s = lo_v - mu_i * mu_o;
    let t = select(1.0, max(mu_i, mu_o), s > 0.0);
    // 0.5 - 2 / (3 * PI) = 0.2877934
    let a_f = 1.0 / (1.0 + 0.2877934 * r);
    let g_i = mu_i * (0.0571085 + mu_i * (0.4918819 + mu_i * (-0.3321814 + 0.0714430 * mu_i)));
    let g_o = mu_o * (0.0571085 + mu_o * (0.4918819 + mu_o * (-0.3321814 + 0.0714430 * mu_o)));
    let e_i = a_f * (1.0 + r * g_i);
    let e_o = a_f * (1.0 + r * g_o);
    let e_avg = a_f * (1.0 + 0.072508 * r);
    let f_ss = a_f * (1.0 + r * (s / max(t, 0.01)));
    let rho_ms = (albedo * e_avg) / max(vec3(1.0) - albedo * (1.0 - e_avg), vec3(0.05));
    let f_ms = rho_ms * ((1.0 - e_i) * (1.0 - e_o) / max(1.0 - e_avg, 0.05));
    return clamp(vec3(f_ss) + f_ms, vec3(0.55), vec3(1.45));
}

// Guerrilla Games / Decima (Horizon Zero Dawn) GGX spherical area-light sun approximation:
// reflects the true finite angular radius of the sun/moon disk on wet stone, ice, metals, and water.
fn decima_noh_sq(no_l: f32, no_v: f32, lo_v: f32, light_radius: f32) -> f32 {
    let radius_cos = cos(light_radius);
    let radius_tan = tan(light_radius);
    let ro_l = 2.0 * no_l * no_v - lo_v;
    if ro_l >= radius_cos {
        return 1.0;
    }
    let r_over_len_t = radius_cos * radius_tan * inverseSqrt(max(1.0 - ro_l * ro_l, 1e-6));
    let not_r = r_over_len_t * (no_v - ro_l * no_l);
    let vot_r = r_over_len_t * (2.0 * no_v * no_v - 1.0 - ro_l * lo_v);
    let new_no_l = no_l * radius_cos + not_r;
    let new_lo_v = lo_v * radius_cos + vot_r;
    let no_h = no_v + new_no_l;
    let ho_h = max(2.0 * new_lo_v + 2.0, 1e-5);
    return clamp(no_h * no_h / ho_h, 0.0, 1.0);
}

fn terrain_specular(normal: vec3<f32>, view: vec3<f32>, no_l: f32, roughness: f32, f0: vec3<f32>, conductor: u32, albedo: vec3<f32>) -> vec3<f32> {
    let half_vector = frame.light_direction + view;
    let half_dir = half_vector * inverseSqrt(max(dot(half_vector, half_vector), 1e-8));
    let no_v = max(dot(normal, view), 0.01);
    let lo_v = clamp(dot(frame.light_direction, view), -1.0, 1.0);
    let lo_h = clamp(dot(frame.light_direction, half_dir), 0.0, 1.0);
    let om = 1.0 - lo_h;
    let om2 = om * om;
    var fresnel = mix(f0, vec3(1.0), om2 * om2 * om);
    if conductor >= 230u && conductor <= 237u {
        fresnel = conductor_fresnel(conductor, lo_h) * albedo;
    }

    let light_radius = select(0.018, 0.025, frame.sun_direction.y > -0.02);
    let n2 = decima_noh_sq(no_l, no_v, lo_v, light_radius);
    let alpha = max(0.014, roughness);
    let alpha2 = alpha * alpha;
    let denominator = 1.0 - n2 + n2 * alpha2;
    let distribution = alpha2 / (PHOTON_PI * denominator * denominator);
    let ggx_l = no_v * sqrt(no_l * no_l * (1.0 - alpha2) + alpha2);
    let ggx_v = no_l * sqrt(no_v * no_v * (1.0 - alpha2) + alpha2);
    let geometry = 0.5 / (ggx_l + ggx_v);
    // Turquin / Kulla-Conty analytical multi-scattering GGX energy compensation:
    // restores lost multi-bounce reflectance on rough metals and dielectrics.
    let e_ss = 1.0 - alpha * (0.28 + 0.22 * (1.0 - no_v)) * (1.0 - 0.35 * alpha);
    let f_avg = select(f0, albedo, conductor >= 230u && conductor <= 237u);
    let multi_scatter = 1.0 + f_avg * ((1.0 - e_ss) / max(e_ss, 0.25));
    return min(fresnel * multi_scatter * (no_l * distribution * geometry), vec3(4.0));
}

// Screen-space directional lightmap shading (upstream Photon directional_lightmaps.glsl):
// infers local torch/skylight direction from precomputed gradients (zero texture fetches).
fn directional_lightmap_mul(pos_dx: vec3<f32>, pos_dy: vec3<f32>,
    lm_dx: vec2<f32>, lm_dy: vec2<f32>, normal: vec3<f32>, light_levels: vec2<f32>) -> vec2<f32> {
    var mul = vec2(1.0);
    let b_grad = vec2(lm_dx.x, lm_dy.x);
    if dot(b_grad, b_grad) > 1e-10 && light_levels.x > 0.10 {
        let dir = pos_dx * b_grad.x + pos_dy * b_grad.y;
        let len_sq = dot(dir, dir);
        if len_sq > 1e-10 {
            let align = clamp(dot(dir * inverseSqrt(len_sq), normal) + 0.8, 0.0, 1.0);
            let raw_mul = clamp((align * 0.50 + 0.50) * inverseSqrt(sqrt(light_levels.x) + 0.08), 0.72, 1.18);
            mul.x = mix(1.0, raw_mul, smoothstep(0.10, 0.36, light_levels.x));
        }
    }
    let s_grad = vec2(lm_dx.y, lm_dy.y);
    if dot(s_grad, s_grad) > 1e-10 && light_levels.y > 0.02 {
        let dir = pos_dx * s_grad.x + pos_dy * s_grad.y;
        let len_sq = dot(dir, dir);
        if len_sq > 1e-10 {
            let align = clamp(dot(dir * inverseSqrt(len_sq), normal) + 0.8, 0.0, 1.0);
            mul.y = clamp((align * 0.35 + 0.65) * inverseSqrt(sqrt(light_levels.y) + 0.05), 0.78, 1.22);
        }
    }
    return mul;
}

// Subtle close-range bevel catchlight along voxel block edges (upstream Photon edge_highlight).
fn block_edge_highlight(position: vec3<f32>, normal: vec3<f32>, dist_sq: f32) -> f32 {
    if dist_sq > 256.0 {
        return 0.0;
    }
    let world = position + frame.camera_position;
    let local = abs(fract(world) - 0.5);
    // Ignore the axis aligned with the face normal so only the 4 perimeter edges highlight.
    let an = abs(normal);
    let edge_axes = local * (1.0 - step(vec3(0.5), an));
    let max_edge = max(edge_axes.x, max(edge_axes.y, edge_axes.z));
    let fade = 1.0 - smoothstep(100.0, 256.0, dist_sq);
    return smoothstep(0.458, 0.494, max_edge) * fade * 0.065;
}

// Flicker-free Nyquist-bandlimited procedural bump & micro-cavity for vanilla 16x16 textures:
// - Avoids dpdx/dpdy on nearest-neighbor albedo (which flickers across 2x2 quads under TAA jitter)
// - Uses exact world-space pixel footprint (fwidth) to bandlimit sub-texel bevels & per-texel tilts
// - Outputs both the perturbed shading normal AND a stable micro-cavity AO for Chan (2018) micro-shadowing
struct AutoBump {
    normal: vec3<f32>,
    cavity: f32,
}

fn auto_bump_surface(position: vec3<f32>, normal: vec3<f32>, pos_dx: vec3<f32>, pos_dy: vec3<f32>,
    value: f32, dist_sq: f32, strength: f32, organic: bool) -> AutoBump {
    let dist_fade = 1.0 - smoothstep(100.0, 576.0, dist_sq);
    let eff = strength * dist_fade;
    if eff <= 0.001 {
        return AutoBump(normal, 1.0);
    }
    var u_axis = vec3(1.0, 0.0, 0.0);
    var v_axis = vec3(0.0, 1.0, 0.0);
    if abs(normal.y) > 0.5 {
        u_axis = vec3(1.0, 0.0, 0.0);
        v_axis = vec3(0.0, 0.0, 1.0);
    } else if abs(normal.x) > 0.5 {
        u_axis = vec3(0.0, 0.0, 1.0);
        v_axis = vec3(0.0, 1.0, 0.0);
    }
    // Exact pixel footprint in 16x16 texel units (Nyquist filter width) to prevent
    // sub-pixel aliasing and TAA jitter flickering at medium distance or grazing angles.
    let fw_x = length(vec2(dot(pos_dx, u_axis), dot(pos_dx, v_axis))) * 16.0;
    let fw_y = length(vec2(dot(pos_dy, u_axis), dot(pos_dy, v_axis))) * 16.0;
    let fw = max(fw_x, fw_y);
    let grain_aa = 1.0 - smoothstep(0.32, 0.95, fw);
    if grain_aa <= 0.001 {
        return AutoBump(normal, 1.0);
    }
    let bevel_aa = 1.0 - smoothstep(0.10, 0.42, fw);

    let world = position + frame.camera_position;
    let tex_coord = vec2(dot(world, u_axis), dot(world, v_axis)) * 16.0;
    let cell = floor(tex_coord);
    let local = fract(tex_coord) - 0.5;
    // Analytic 16x16 sub-texel micro-bevel (only active when screen resolves >= ~2.5 px/texel).
    let rim = smoothstep(vec2(0.30), vec2(0.49), abs(local));
    let bevel = sign(local) * rim;
    let grad_bevel = (u_axis * bevel.x + v_axis * bevel.y) * (select(0.15, 0.05, organic) * bevel_aa);

    // Deterministic per-texel micro-tilt (constant across each 16x16 texel -> 100% TAA-stable).
    let block_seed = floor(world + 0.001);
    let h1 = fract(sin(dot(cell, vec2(127.1, 311.7)) + dot(block_seed, vec3(74.7, 19.3, 53.1))) * 43758.5453);
    let h2 = fract(sin(dot(cell, vec2(269.5, 183.3)) + dot(block_seed, vec3(37.2, 91.7, 13.9))) * 43758.5453);
    let tilt = (vec2(h1, h2) - 0.5) * (select(0.08, 0.22, organic) * grain_aa);
    let grad_grain = u_axis * tilt.x + v_axis * tilt.y;

    // Stable micro-cavity visibility for Chan (SIGGRAPH 2018) directional micro-shadowing:
    // darker recessed texels (mortar, cracks, grass blade bases) + micro-bevel seams get
    // crisp directional self-shadowing at oblique sun angles with zero temporal flicker.
    let height_vis = mix(0.68, 1.0, smoothstep(0.16, 0.58, value));
    let seam_occl = 1.0 - max(rim.x, rim.y) * (select(0.22, 0.08, organic) * bevel_aa);
    let cavity = mix(1.0, height_vis * seam_occl, clamp(eff * grain_aa, 0.0, 1.0));

    return AutoBump(normalize(normal + (grad_bevel + grad_grain) * eff), cavity);
}

fn linear_step(edge0: f32, edge1: f32, x: f32) -> f32 {
    return clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
}

// Single-site wave_noise helper for lava (keeps [material] static texture-read count minimal).
fn lava_noise(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(wave_noise, wave_noise_sampler, uv, 0.0).rgb;
}

// Physically-based Minecraft-native Molten Lava Shader:
// Preserves 100% of Minecraft's pixel-art lava texture while driving its thermal state
// with Stora et al. (GI'99) macro convection crusting, flow-shear heating, Planck blackbody
// chromaticity, Stefan-Boltzmann HDR radiance, and basaltic crust rim sheen.
fn shaded_lava(input: MaterialInput) -> vec4<f32> {
    let dist = length(input.position);
    let view = -input.position / max(dist, 1e-5);
    let world = input.position + frame.camera_position;
    let flat = input.normal;
    let side = input.face >= 2u;
    let flowing = side || abs(flat.y) < 0.985;

    // Flow-aware macro coordinates (breaks up 1x1 block uniformity across large lava oceans
    // without replacing Minecraft's crisp pixel-art lava texture).
    let axis_u = select(vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, -1.0), input.face >= 4u);
    let axis_v = select(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), side);
    var coord = vec2(dot(world, axis_u), dot(world, axis_v));
    var flow_vel = vec2(0.012, -0.009);
    if side {
        coord = vec2(coord.x * 1.2, coord.y * 0.45);
        flow_vel = vec2(0.0, 0.18);
    } else if flowing {
        let downhill = normalize(flat.xz + vec2(1e-5, 0.0));
        flow_vel = -downhill * 0.07;
    }

    // 1. Extract Minecraft's native pixel-art lava heat & color from input.albedo.
    //    Bright yellow pixels in the vanilla lava texture = hot molten magma;
    //    darker orange/brown pixels = cooler crusted basalt skin.
    let mc_albedo = max(input.albedo.rgb, vec3(0.001));
    let mc_luma = dot(mc_albedo, frame.luminance);
    // Normalize vanilla lava texel brightness into a crisp [0, 1] pixel-art heat map.
    let tex_heat = clamp((mc_luma - 0.14) / 0.62, 0.0, 1.0);

    // 2. Texel-snapped macro tectonic convection (~20-90m scale):
    //    Quantized to Minecraft's 16x16 texel grid so thermal variation aligns cleanly
    //    with the pixel art (zero smooth contour rings or non-pixelated ripples).
    let t = frame.time;
    let p_coord = floor(coord * 16.0) * (1.0 / 16.0);
    let macro_n = lava_noise(p_coord * 0.009 + flow_vel * (t * 0.30));
    let conv_n = lava_noise(p_coord * 0.026 - flow_vel.yx * (t * 0.22) + vec2(0.37, 0.71));

    let magma_river = smoothstep(0.22, 0.82, macro_n.r * 0.65 + conv_n.r * 0.35);
    let pulse = 0.035 * sin(t * 0.72 + macro_n.b * TAU + conv_n.b * 3.0);
    // Stora et al. (1999): flowing streams & vertical lavafalls fracture crust and stay hotter.
    let flow_heat = select(0.0, select(0.16, 0.26, side), flowing);

    // Keep the entire lava sea molten (zero black patches): cooler convection zones
    // glow rich deep volcanic crimson-orange while hotter currents & lavafalls surge
    // up the Planck blackbody curve into bright molten gold.
    let env_temp = clamp(0.54 + 0.34 * magma_river + flow_heat * 0.75 + pulse, 0.48, 1.0);
    let temp = clamp(mix(env_temp * 0.78, 0.58 + 0.42 * env_temp, tex_heat), 0.42, 1.0);

    // Crust mask on cooled, darker texels of the Minecraft lava texture.
    let crust = 1.0 - smoothstep(0.26, 0.66, temp);

    // 3. Physical Planck blackbody emission blended with Minecraft's texture chroma
    //    so pixel-art detail stays crisp while temperature shifts hue & HDR intensity naturally.
    let bb = lava_blackbody(temp);
    let mc_chroma = mc_albedo / max(mc_luma, 0.08);
    let emission = bb * mix(vec3(1.0), clamp(mc_chroma, vec3(0.55), vec3(1.45)), 0.35);

    // 4. Cooled basaltic crust shading on dark pixels (charred rock + subtle glassy Fresnel sheen).
    let env_tint = select(vec3(0.20, 0.08, 0.035),
        pow(max(frame.fog_color, vec3(0.12, 0.04, 0.015)), vec3(2.2)) * 1.35,
        frame.dimension == -1);
    let no_v = clamp(dot(flat, view), 0.0, 1.0);
    let fresnel = 0.04 + 0.96 * pow(1.0 - no_v, 4.0);
    let basalt_albedo = mix(vec3(0.038, 0.026, 0.022), mc_albedo * 0.18, 0.5);
    let crust_diffuse = basalt_albedo * (env_tint + vec3(0.18, 0.045, 0.01) * (1.0 - crust));
    let crust_sheen = (env_tint + vec3(0.22, 0.06, 0.012)) * (fresnel * 0.32 * crust);
    let edge_sss = vec3(0.42, 0.04, 0.003) * (crust * smoothstep(0.16, 0.46, temp) * 0.50);

    let final_rgb = crust_diffuse + crust_sheen + edge_sss + emission;
    set_surface(flat, 0.0, lava_surface_payload(tex_heat, temp));
    return vec4(final_rgb, input.albedo.a);
}

fn material(input: MaterialInput) -> Surface {
    // Compute derivatives in uniform control flow for directional lightmaps, Nyquist bump, and LabPBR POM.
    let pos_dx = dpdx(input.position);
    let pos_dy = dpdy(input.position);
    let lm_dx = dpdx(input.light);
    let lm_dy = dpdy(input.light);
    let uv_dx = dpdx(input.uv);
    let uv_dy = dpdy(input.uv);

    if input.fluid == 2u {
        return surface_fog(shaded_lava(input), input.position);
    }

    if frame.dimension == -1 {
        // Custom Nether shading: biome-tinted cavern fill + warm downward magma-sea bounce
        // + directional face contrast so unlit cliffs never crush to pitch black.
        let albedo = input.albedo.rgb * input.tint.rgb;
        let ao = clamp(input.ao, 0.0, 1.0);
        let directional = (0.86 + 0.14 * input.normal.x) * (0.76 + 0.24 * abs(input.normal.y));
        let dir_lm = directional_lightmap_mul(pos_dx, pos_dy, lm_dx, lm_dy, input.normal, input.light);
        let fog_tint = pow(max(frame.fog_color, vec3(0.14, 0.06, 0.04)), vec3(2.2));
        let fog_luma = max(dot(fog_tint, frame.luminance), 1e-4);
        let fog_norm = fog_tint / fog_luma;
        let ambient_tint = mix(fog_norm, vec3(1.0), 0.38);
        // Subtle warm bounce from the molten lava sea below onto downward-facing ledges/overhangs.
        let lava_bounce = vec3(1.0, 0.42, 0.16) * (0.09 * (0.5 - 0.5 * input.normal.y));
        let base_amb = (ambient_tint * clamp(0.24 + 0.35 * fog_luma, 0.28, 0.42) + lava_bounce) * mix(0.46, 1.0, ao);
        let ambient = (base_amb + vec3(1.5 * frame.night_vision) * ao) * directional;
        let block = surface_blocklight(input.position, input.light, ao, directional) * dir_lm.x;
        let glow = albedo * (0.65 * input.emission);
        let lit = (albedo / PHOTON_PI) * (ambient + block * 1.15) + glow;
        return surface_fog(vec4(lit, input.albedo.a), input.position);
    }
    if frame.dimension == 1 {
        // Custom End shading: directional End sun + deep violet-indigo ambient + glowing blocklight.
        let albedo = input.albedo.rgb * input.tint.rgb;
        let ao = clamp(input.ao, 0.0, 1.0);
        let directional = (0.9 + 0.1 * input.normal.x) * (0.8 + 0.2 * abs(input.normal.y));
        let dir_lm = directional_lightmap_mul(pos_dx, pos_dy, lm_dx, lm_dy, input.normal, input.light);
        let end_light_dir = normalize(vec3(-0.83, 0.31, 0.46));
        let end_no_l = clamp(dot(input.normal, end_light_dir), 0.0, 1.0);
        let end_sun_lit = vec3(1.15, 0.82, 0.96) * (terrain_lift(end_no_l, 0.35) * ao * ao * 0.95);
        let end_ambient = vec3(0.24, 0.18, 0.36) * (directional * ao);
        let block = surface_blocklight(input.position, input.light, ao, directional) * dir_lm.x;
        let glow = albedo * (0.65 * input.emission);
        let lit = (albedo / PHOTON_PI) * (end_sun_lit + end_ambient + block) + glow;
        return surface_fog(vec4(lit, input.albedo.a), input.position);
    }
    if frame.dimension != 0 {
        return surface_fog(input.albedo * vec4(input.shaded_tint.rgb * input.lightmap, 1.0), input.position);
    }
    if input.fluid == 1u && input.translucent {
        return surface_fog(refracted_water(input), input.position);
    }

    var albedo = input.albedo.rgb * input.tint.rgb;
    let leaves = (input.tags & 1u) != 0u;
    let cherry_leaves = leaves && (input.tags & 8192u) != 0u;
    if leaves {
        albedo = canopy_albedo(albedo);
        if cherry_leaves {
            // van der Kooi et al. (Proc. R. Soc. B 2016): real flower petals cap at ~45-50%
            // diffuse reflectance because thin mesophyll transmits more light than it reflects.
            albedo = albedo / (vec3(1.0) + 0.36 * albedo);
        }
    }
    let plant = (input.tags & 2u) != 0u;
    // Ordinary distant blocks outside the shadow map skip shadow-map, PBR, wet,
    // and specular work, while matching the near path's foliage optics, cloud
    // shadows, AO contrast, and skylight so there is no visual seam.
    let dist_sq = dot(input.position, input.position);
    let detailed = (input.tags & 124u) != 0u || input.translucent || block_pbr_maps() || input.emission > 0.0;
    let outside_shadow_map = !input.translucent && input.emission <= 0.0 && dist_sq > SHADOW_MAP_DIST_SQ;
    if outside_shadow_map || (!pack_shadow_active() && ((!detailed && (leaves || plant || dist_sq > 1024.0))
        || (!input.translucent && input.emission <= 0.0 && dist_sq > 9216.0))) {
        let turf = (input.tags & 256u) != 0u;
        let mesh_ao = clamp(input.ao, 0.0, 1.0);
        let ao = select(mesh_ao, mix(mesh_ao, 1.0, 0.35), leaves);
        let sky = input.light.y;
        let sky2 = sky * sky;
        let view = -input.position * inverseSqrt(max(dist_sq, 1e-8));
        let raw_normal = select(input.normal, -input.normal, plant && dot(input.normal, view) < 0.0);
        let normal = select(raw_normal, normalize(mix(raw_normal, vec3(0.0, 1.0, 0.0), 0.55)), plant);
        let directional = (0.9 + 0.1 * normal.x) * (0.8 + 0.2 * abs(input.normal.y));
        let signed_no_l = dot(normal, frame.light_direction);
        let no_l = clamp(signed_no_l, 0.0, 1.0);
        let subsurface = select(select(0.0, 0.5, plant), min(tree_scattering, 1.0), leaves);
        let sheen_amount = select(select(0.0, 1.0, plant), 0.5, leaves);
        let leaf_strength = clamp(tree_scattering / 1.25, 0.0, 1.0);

        if turf {
            let layer = foliage_optics(albedo, 0.35);
            let soil = vec3(0.12, 0.072, 0.038);
            let turf_albedo = layer.reflected + layer.transmitted * layer.transmitted * soil
                / (1.0 - layer.reflected * soil);
            albedo = mix(albedo, turf_albedo, 0.35);
        }

        let light = direct_atmosphere_light() * cloud_shadow(input.position);
        var vis = smoothstep(14.1 / 15.0, 14.9 / 15.0, sky);
        if leaves {
            // Match the near 3D shadow-map + GTAO crown self-occlusion so distant canopies
            // (>114m) have the exact same deep shaded underside and interior contrast.
            let crown_vis = clamp(0.10 + 0.90 * smoothstep(-0.05, 0.62, signed_no_l), 0.08, 1.0)
                * mix(0.22, 1.0, mesh_ao * mesh_ao * mesh_ao);
            vis *= crown_vis;
        }
        let diffuse_term = terrain_lift(no_l, 0.25) * (1.0 - 0.5 * subsurface) * ao * ao;
        let bounce_scale = select(0.033, 0.0, ssilvb_gi || gi_rsm_bounce);
        let bounce = bounce_scale * (1.0 - vis) * (1.0 - 0.1 * max(normal.y, 0.0))
            * (ao * sqrt(ao)) * sky2 * sky2;
        var sunlight = vec3(select(diffuse_term, terrain_lift(no_l, 0.22) * ao * ao, turf) * vis + bounce);
        if (leaves || plant) && subsurface > 0.0 && sky >= 0.1 && dist_sq < 48400.0 {
            let light_view = clamp(dot(frame.light_direction, view), -1.0, 1.0);
            let distant_shadow = TerrainShadow(vec3(vis), 1.2 * (1.0 - vis), 1.0);
            let profile = sss_profile(input.tags, leaves, plant, albedo);
            let scatter = terrain_subsurface(albedo, subsurface, sheen_amount, profile, distant_shadow,
                input.position, normal, input.tags, leaves, plant, signed_no_l, light_view, ao, sky, mesh_ao);
            sunlight += scatter * select(1.0, leaf_strength * mix(0.38, 1.0, vis), leaves);
        }
        sunlight *= light;

        let blocklight = surface_blocklight(input.position, input.light, ao, directional);
        let cave = 0.15 * (1.0 - sky2) * (1.0 - 0.7 * frame.darkness);
        let base = (cave + 1.5 * frame.night_vision) * ao * directional;
        let diffuse_scale = albedo / PHOTON_PI;
        var ambient = diffuse_scale * (blocklight + base);
        if leaves {
            // Match upstream Photon diffuse_lighting.glsl: blend directional skylight
            // with hemispherical sky fill so dark spruce/pine canopies stay cool and legible.
            let sky_lit = terrain_skylight(normal, sky, ao, subsurface, true);
            ambient += diffuse_scale * sky_lit;
        } else {
            ambient += diffuse_scale * terrain_skylight(normal, sky, ao, subsurface, false);
        }

        let direct_rgb = diffuse_scale * sunlight;
        let color = direct_rgb + ambient;
        if outside_shadow_map && (shadows || contact_ao > 0.0) {
            let inv_lum = 1.0 / max(dot(color, frame.luminance), 1e-5);
            let direct_frac = dot(direct_rgb, frame.luminance) * inv_lum;
            let ambient_frac = dot(ambient, frame.luminance) * inv_lum;
            set_surface(normal, 0.0, surface_payload(vec3(0.0), ambient_frac, sky, direct_frac));
        }
        return surface_fog(vec4(color, input.albedo.a), input.position);
    }
    let turf = (input.tags & 256u) != 0u;
    let grass_blade = (input.tags & 512u) != 0u;
    var blade_exposure = 1.0;
    if grass_blade || plant {
        var height = fract(fract(frame.camera_position.y) + input.position.y);
        if (input.tags & 3072u) != 0u {
            height = 0.5 * (height + select(0.0, 1.0, (input.tags & 2048u) != 0u));
        }
        blade_exposure = smoothstep(0.02, 0.85, height);
    }
    // Horizon's mesh AO is already dark in leaf clusters. Photon obtains
    // its canopy fill from a separate deferred ambient scattering term.
    let mesh_ao = clamp(input.ao, 0.0, 1.0);
    let ao = select(mesh_ao, mix(mesh_ao, 1.0, 0.35), leaves);
    var normal = input.normal;
    var material_ao = 1.0;
    var emission = 0.0;
    var subsurface = select(select(0.0, 0.5, plant), min(tree_scattering, 1.0), leaves);
    let sheen_amount = select(select(0.0, 1.0, plant), 0.5, leaves);
    let c_max = max(albedo.r, max(albedo.g, albedo.b));
    let masked_emitter = (input.tags & 64u) != 0u;
    let snow_and_ores = (input.tags & 32768u) != 0u;
    var value = 0.0;
    if !plant || snow_and_ores || masked_emitter {
        value = pow(c_max, 1.0 / 2.2);
    }
    // Upstream Photon IntegratedPBR (include/surface/material.glsl):
    // per-texel lightness drives smoothness so brighter grass/leaf/stone texels catch
    // crisp individual sun highlights while darker shadowed texels remain matte.
    var smoothness = select(select(0.33 * smoothstep(0.2, 0.6, value), 0.0, plant),
        select(0.5 * smoothstep(0.16, 0.5, value), 0.18 * smoothstep(0.22, 0.65, value), cherry_leaves), leaves);
    // Magnus-Tetens morning dew sheen on sky-exposed grass, turf, and leaves at dawn
    // in above-freezing non-desert biomes; evaporates cleanly as the morning sun climbs.
    if (leaves || plant || turf) && input.light.y > 0.72 && frame.temperature > 0.15 && frame.downfall > 0.25 {
        let night_age = fract(frame.day_time + 0.5);
        let dawn_dew = smoothstep(0.42, 0.50, night_age) * (1.0 - smoothstep(0.56, 0.70, night_age))
            * smoothstep(0.72, 0.95, input.light.y) * (1.0 - 0.75 * frame.rain) * weather_humidity();
        smoothness = mix(smoothness, select(0.42, 0.52, leaves), dawn_dew * 0.55);
    }
    let metal = (input.tags & 4u) != 0u;
    let ice = (input.tags & 8u) != 0u;
    let glass = (input.tags & 16u) != 0u;
    let polished_stone = (input.tags & 32u) != 0u;
    let transparent_solid = (input.tags & 128u) != 0u;
    let c_min = min(albedo.r, min(albedo.g, albedo.b));
    let sat = (c_max - c_min) / max(c_max, 1e-4);
    var metallic = metal;
    var reflectance = select(vec3(0.02), albedo, metal);
    var conductor = 0u;
    var porosity = select(select(0.45, 0.68, turf), 0.05, metal || ice || glass || polished_stone || transparent_solid);
    // Upstream Photon default LabPBR material assignments (include/surface/material.glsl).
    if metal {
        smoothness = min(0.82, sqrt(linear_step(0.1, 0.9, value)));
    } else if ice {
        let ice_s = linear_step(0.4, 0.8, value);
        let ice_s2 = ice_s * ice_s;
        smoothness = 0.6 * ice_s2 * ice_s2;
        subsurface = 0.75;
    } else if glass || transparent_solid {
        smoothness = 0.68;
        reflectance = vec3(0.04);
        subsurface = 0.5;
    } else if polished_stone {
        if sat > 0.48 && value > 0.28 {
            // Gems & vibrant glazed terracotta: high dielectric F0 (0.25) and crisp gloss.
            smoothness = min(0.82, sqrt(linear_step(0.1, 0.9, value)));
            reflectance = vec3(0.25);
        } else if value < 0.24 {
            // Obsidian, blackstone, dark nether/deepslate bricks: volcanic-glass gloss.
            smoothness = linear_step(0.02, 0.4, value);
            reflectance = vec3(0.02);
        } else {
            // Polished stone, quartz, and wood planks.
            smoothness = 0.5 * linear_step(0.4, 0.8, value);
        }
    } else if !turf && !leaves && !plant && !snow_and_ores && value > 0.78 && sat > 0.14 && sat < 0.42
        && albedo.r >= albedo.g && albedo.g > albedo.b {
        // Sand & sandstone: individual sparkling quartz grains instead of uniform sheen.
        smoothness = 0.8 * linear_step(0.81, 0.96, value);
    }
    if snow_and_ores {
        if value > 0.80 && sat < 0.16 {
            // Snow & powder snow: glacial subsurface scattering + close-range sunlit crystal sparkle.
            var sparkle = 0.0;
            if dist_sq < 576.0 && input.light.y > 0.75 && frame.sun_direction.y > 0.05 {
                let sw = floor((input.position + frame.camera_position) * 16.0);
                let view_q = floor((-input.position * inverseSqrt(max(dist_sq, 1e-6))) * 14.0);
                let h = fract(sin(dot(sw + view_q * 0.37, vec3(127.1, 311.7, 74.7))) * 43758.5453);
                let sp_fade = (1.0 - smoothstep(144.0, 576.0, dist_sq)) * (1.0 - frame.rain);
                sparkle = smoothstep(0.965, 0.996, h) * sp_fade;
            }
            smoothness = min(0.88, 0.30 * smoothstep(0.84, 0.98, value) + 0.52 * sparkle);
            subsurface = 0.60;
            porosity = 0.15;
        } else {
            // Glowing ores, crying obsidian, amethyst, sculk, magma: isolate vivid veins only.
            let ore_mask = smoothstep(0.42, 0.72, sat) * smoothstep(0.10, 0.38, value);
            emission = max(emission, 0.14 * ore_mask);
            smoothness = max(smoothness, 0.65 * ore_mask);
        }
    }

    var gbuffer_normal = input.normal;
    var pom_shadow = 1.0;
    if block_pbr_maps() {
        var sample_uv = input.uv;
        var map_normal = pbr_normal_at(sample_uv);
        // LabPBR Parallax Occlusion Mapping (6-step relief + 4-step tangent-space self-shadowing)
        // using block_normal_map().a heightmap, clamped safely within the local sprite tile scale.
        if labpbr_pom && dist_sq < 576.0 && map_normal.a > 0.004 && map_normal.a < 0.996 {
            let tangent = input.tangent.xyz;
            let bitangent = cross(input.normal, tangent) * input.tangent.w;
            let v_dir = -input.position * inverseSqrt(max(dist_sq, 1e-8));
            let tv_z = max(dot(v_dir, input.normal), 0.18);
            let tv_xy = vec2(dot(v_dir, tangent), dot(v_dir, bitangent));
            // Derive conservative UV-per-block scale from derivatives to avoid crossing atlas tile edges.
            let dp_len = max(length(pos_dx), length(pos_dy));
            let du_len = max(length(uv_dx), length(uv_dy));
            let uv_per_m = clamp(du_len / max(dp_len, 1e-5), 0.0005, 0.0625);
            let pom_fade = 1.0 - smoothstep(256.0, 576.0, dist_sq);
            let max_shift = min(0.055 * uv_per_m * pom_fade, 0.0032);
            let step_uv = clamp(-(tv_xy / tv_z) * (max_shift / 6.0), vec2(-0.0006), vec2(0.0006));
            var cur_h = 1.0;
            var cur_uv = input.uv;
            var hit_h = map_normal.a;
            for (var s = 0u; s < 6u; s += 1u) {
                cur_h -= 1.0 / 6.0;
                cur_uv += step_uv;
                let probe = pbr_normal_at(cur_uv);
                if probe.a >= cur_h {
                    sample_uv = cur_uv;
                    map_normal = probe;
                    hit_h = probe.a;
                    break;
                }
            }
            // 4-step parallax self-shadow march toward the sun in tangent space.
            let tl_z = dot(frame.light_direction, input.normal);
            if tl_z > 0.08 && hit_h < 0.96 {
                let tl_xy = vec2(dot(frame.light_direction, tangent), dot(frame.light_direction, bitangent));
                let l_step = clamp((tl_xy / max(tl_z, 0.20)) * (max_shift * 0.25), vec2(-0.0008), vec2(0.0008));
                var sh_h = hit_h;
                var sh_uv = sample_uv;
                var occ = 0.0;
                for (var s = 0u; s < 4u; s += 1u) {
                    sh_h += (1.0 - hit_h) * 0.25;
                    sh_uv += l_step;
                    let h_s = pbr_normal_at(sh_uv).a;
                    if h_s > sh_h + 0.02 {
                        occ = max(occ, smoothstep(0.02, 0.14, h_s - sh_h) * (1.0 - f32(s) * 0.18));
                    }
                }
                pom_shadow = 1.0 - occ * (0.72 * pom_fade);
            }
        }
        normal = mapped_normal(input, map_normal.rg * 2.0 - 1.0);
        gbuffer_normal = normal;
        material_ao = mix(0.7, 1.0, map_normal.b);
        let pbr = block_specular_map(sample_uv);
        // An absent specular sprite uses Horizon's all-zero default.
        if any(pbr != vec4(0.0)) {
            smoothness = pbr.r;
            emission = select(pbr.a, 0.0, pbr.a >= 0.999);
            conductor = u32(round(pbr.g * 255.0));
            metallic = conductor >= 230u;
            reflectance = select(vec3(max(0.02, pbr.g)), albedo, metallic);
            if !metallic {
                if pbr.b > 64.5 / 255.0 {
                    subsurface = max(subsurface, (pbr.b - 64.0 / 255.0) / (1.0 - 64.0 / 255.0));
                    porosity = 0.0;
                } else {
                    porosity = clamp(pbr.b * (255.0 / 64.0), 0.0, 1.0);
                }
            }
        }
    } else if bump_strength > 0.0 && dist_sq < 576.0 && !leaves && !plant && !glass && !ice
        && !transparent_solid && !input.translucent && input.fluid == 0u {
        let is_snow = snow_and_ores && value > 0.80 && sat < 0.16;
        let organic = turf || is_snow || porosity > 0.40;
        let mat_scale = select(select(select(1.0, 0.55, polished_stone || metal), 0.85, turf), 0.45, is_snow);
        let bump = auto_bump_surface(input.position, normal, pos_dx, pos_dy, value,
            dist_sq, bump_strength * mat_scale, organic);
        normal = bump.normal;
        material_ao *= bump.cavity;
    }

    if turf && !metallic {
        // A finite diffuse grass layer backed by opaque soil. Keep the blend gentle
        // so sunlit grass blocks preserve Photon's vibrant daytime sunlight intensity.
        let layer = foliage_optics(albedo, 0.35);
        let soil = vec3(0.12, 0.072, 0.038);
        let turf_albedo = layer.reflected + layer.transmitted * layer.transmitted * soil
            / (1.0 - layer.reflected * soil);
        albedo = mix(albedo, turf_albedo, 0.35);
    }

    var wet_film = 0.0;
    if wet_surfaces && !leaves && !plant && !input.translucent && input.fluid == 0u && input.emission <= 0.0 {
        let wy = input.position.y + frame.camera_position.y;
        if frame.wetness > 0.0 || (wy >= 61.9 && wy <= 63.5) {
            let wet = apply_wet_surface(input, normal, porosity);
            albedo *= wet.darkening;
            wet_film = wet.film;
            normal = wet.normal;
            gbuffer_normal = normalize(mix(gbuffer_normal, wet.normal, wet_film));
            smoothness = mix(smoothness, 0.94, wet_film);
        }
    }

    let sky = input.light.y;
    let sky2 = sky * sky;
    let view = -input.position * inverseSqrt(max(dot(input.position, input.position), 1e-8));
    if plant {
        if dot(input.normal, view) < 0.0 {
            normal = -normal;
        }
        // Soften horizontal crossed-quad normals upward so intersecting grass/flower
        // planes catch natural sky/sun lighting without a harsh X-crease down the center.
        normal = normalize(mix(normal, vec3(0.0, 1.0, 0.0), 0.55));
        gbuffer_normal = normal;
    }
    let directional = (0.9 + 0.1 * normal.x) * (0.8 + 0.2 * abs(input.normal.y));
    let signed_no_l = dot(normal, frame.light_direction);
    let no_l = clamp(signed_no_l, 0.0, 1.0);
    let light_view = clamp(dot(frame.light_direction, view), -1.0, 1.0);
    // Tokuyoshi (JCGT 2021) Geometric Specular Anti-Aliasing: analytically widen GGX
    // roughness from screen-space normal derivatives to kill specular crawling/sparkle
    // on wet surfaces, metals, polished stone, and steep block edges before AA runs.
    let dndx = dpdx(normal);
    let dndy = dpdy(normal);
    let geom_var = min(0.5 * (dot(dndx, dndx) + dot(dndy, dndy)), 0.18);
    let base_rough = (1.0 - smoothness) * (1.0 - smoothness);
    let roughness = sqrt(clamp(base_rough * base_rough + geom_var, 0.0004, 1.0));
    let shadowed = pack_shadow_active();
    var light = direct_atmosphere_light();
    var foliage = vec3(0.0);
    let leaf_strength = clamp(tree_scattering / 1.25, 0.0, 1.0);
    var leaf: FoliageOptics;
    if !shadowed && leaves {
        // Use the tinted sprite as the pigment estimate before canopy grading.
        leaf = foliage_optics(input.albedo.rgb * input.tint.rgb, 0.70 / max(1.0, tree_scattering / 1.25));
    }
    var sunlight = vec3(0.0);
    var sss_protected = vec3(0.0);
    var specular = vec3(0.0);
    if (!shadowed || no_l > 0.0 || subsurface > 0.0 || sky > 0.0) && max(light.r, max(light.g, light.b)) > 0.0 {
        let cloud = cloud_shadow(input.position);
        light *= cloud;
        // Submerged blocks always have skylight < 14.5/15 (water attenuates skylight by 1+ per block),
        // and leaves are never submerged, so gating on !leaves && sky < 0.97 skips ~98% of dry terrain.
        if shadowed && input.fluid == 0u && !input.translucent && !leaves && dist_sq < 2304.0 && sky < 0.97 {
            light *= water_receiver_light(input.position, normal);
        }
        let pixel_cell = floor(input.screen * frame.resolution);
        var visibility = vec3(1.0);
        if shadowed {
            var shadow = TerrainShadow(vec3(0.0), 0.0, 1.0);
            if leaves {
                shadow = leaf_shadow(input.position, input.normal, sky, subsurface, cloud, pixel_cell);
                // Distant canopies outside the 128-block shadow map lose branch self-shadowing;
                // approximate crown occlusion from sun-facing normal + vertex AO so distant
                // forests match nearby shaded tree contrast seamlessly across 104..128 blocks.
                if shadow.fade > 0.001 {
                    let ao2_m = mesh_ao * mesh_ao;
                    let crown_vis = clamp(0.06 + 0.94 * smoothstep(-0.02, 0.64, signed_no_l), 0.06, 1.0)
                        * mix(0.14, 1.0, ao2_m * ao2_m);
                    shadow.visibility *= mix(1.0, crown_vis, shadow.fade);
                }
            } else if no_l > 0.0 || (subsurface > 0.0 && !turf) {
                // Crossed plant quads are visible on both sides. Bias toward
                // the light so a backlit blade does not shadow itself.
                let receiver_normal = select(input.normal, -input.normal,
                    plant && dot(input.normal, frame.light_direction) < 0.0);
                // Opaque turf ground uses micro-blade SSS only (skip 4-tap shadow blocker thickness).
                let shadow_sss = select(subsurface, 0.0, turf);
                shadow = terrain_shadow(input.position, receiver_normal, sky, shadow_sss, cloud, pixel_cell);
            }
            visibility = shadow.visibility;
            var eon_mul = vec3(1.0);
            var dir_micro = 1.0;
            if !leaves && !plant && no_l > 0.0 {
                if !metallic && !input.translucent {
                    let no_v_eon = clamp(dot(normal, view), 0.0, 1.0);
                    eon_mul = eon_diffuse_factor(albedo, 1.0 - smoothness, no_l, no_v_eon, light_view);
                }
                let cavity_vis = clamp(material_ao * mix(0.62, 1.0, ao), 0.08, 0.999);
                dir_micro = mix(1.0, micro_shadow(no_l, cavity_vis), 0.65);
            }
            let diffuse = terrain_lift(no_l, 0.25) * (1.0 - 0.5 * subsurface) * ao * ao;
            let bounce_scale = select(0.033, 0.0, ssilvb_gi || gi_rsm_bounce);
            let bounce = bounce_scale * (1.0 - shadow.visibility) * (1.0 - 0.1 * max(normal.y, 0.0))
                * (ao * sqrt(ao)) * sky2 * sky2;
            sunlight = diffuse * eon_mul * (dir_micro * pom_shadow) * shadow.visibility + bounce;
            if subsurface > 0.0 && sky >= 0.1 {
                let profile = sss_profile(input.tags, leaves, plant, albedo);
                let scatter = terrain_subsurface(albedo, subsurface, sheen_amount, profile, shadow,
                    input.position, normal, input.tags, leaves, plant, signed_no_l, light_view, ao, sky, mesh_ao)
                    * select(1.0, leaf_strength, leaves);
                sunlight += scatter;
                if leaves || plant {
                    // Only protect the strictly backlit transmission component (signed_no_l < 0)
                    // from screen-space contact shadow re-occlusion; front-lit faces keep full contact shadows.
                    let backlit_only = clamp(-signed_no_l, 0.0, 1.0);
                    sss_protected = scatter * (backlit_only * (1.0 - clamp(shadow.fade, 0.0, 1.0))) * light;
                }
            }
            if grass_blade || plant {
                // Sun-elevation grass-field self-shadowing: roots darken deeper at low sun angles
                // while upper tips stay brightly sunlit and catch backlit SSS.
                let root_floor = mix(0.28, 0.60, smoothstep(0.05, 0.55, frame.light_direction.y));
                sunlight *= mix(root_floor, 1.0, blade_exposure);
            }

        } else {
            let diffuse = (0.6 * terrain_lift(no_l, 0.5) + 0.4) * ao * directional * sky2 * sky2;
            sunlight = vec3(diffuse * (0.5 + 0.5 * photon_dampen(frame.light_direction.y)));
            if turf { sunlight = vec3(no_l * sky2 * sky2); }
            if plant || leaves {
                sunlight = vec3(0.0);
                visibility = vec3(sky2 * sky2);
            }
        }
        if !shadowed && (plant || leaves) {
            let facing = select(-input.normal, input.normal, dot(input.normal, view) >= 0.0);
            let incidence = dot(facing, frame.light_direction);
            let strength = select(min(1.0, 2.0 * subsurface), clamp(tree_scattering / 1.25, 0.0, 1.0), leaves);
            var thickness = 0.42;
            var projected = vec2(max(incidence, 0.0), max(-incidence, 0.0));
            if plant || leaves {
                // A grass card or leaf block depicts multiple orientations. Average positive
                // and negative light cosines over viewer-facing blade orientations.
                let clump = 0.25 * vec2(1.0 + light_view, 1.0 - light_view);
                projected = mix(clump, projected, select(0.25, 0.45, leaves));
            }
            var pigment = albedo;
            if grass_blade {
                thickness = mix(0.65, 0.27, blade_exposure);
                // A small warm shift follows the biome tint along sunlit tips.
                pigment *= mix(vec3(1.0), vec3(1.14, 1.0, 0.88), blade_exposure);
                projected *= mix(0.48, 1.0, blade_exposure);
            }
            var optics = leaf;
            if !leaves { optics = foliage_optics(pigment, thickness); }
            foliage = (optics.reflected * projected.x + optics.transmitted * (strength * projected.y))
                * visibility * light / PHOTON_PI;
        }
        if no_l > 0.0 && max(visibility.r, max(visibility.g, visibility.b)) > 0.005 {
            specular = terrain_specular(normal, view, no_l, roughness, reflectance, conductor, albedo)
                * light * visibility * ao;
        }
        sunlight *= light;
    }

    let ambient_ao = ao * material_ao;
    var dir_lm = vec2(1.0);
    if !plant && (dot(lm_dx, lm_dx) + dot(lm_dy, lm_dy) > 1e-10) {
        dir_lm = directional_lightmap_mul(pos_dx, pos_dy, lm_dx, lm_dy, normal, input.light);
    }

    let blocklight = surface_blocklight(input.position, input.light, ambient_ao, directional) * dir_lm.x;
    let cave = 0.15 * (1.0 - sky2) * (1.0 - 0.7 * frame.darkness);
    let base = (cave + 1.5 * frame.night_vision) * ao * directional;
    let diffuse_scale = albedo * (select(1.0, 0.5, metallic) / PHOTON_PI);
    var ambient = diffuse_scale * (blocklight + base);
    if leaves {
        // Match upstream Photon diffuse_lighting.glsl skylight SSS on canopies.
        let local_sky = terrain_skylight(normal, sky, ambient_ao, subsurface, true);
        ambient += diffuse_scale * local_sky;
    } else {
        let local_sky = terrain_skylight(normal, sky, ambient_ao, subsurface, false) * dir_lm.y;
        ambient += diffuse_scale * local_sky;
    }
    if grass_blade || plant {
        ambient *= mix(0.58, 1.0, blade_exposure);
    }
    var diffuse = diffuse_scale * sunlight + foliage + ambient;
    if !leaves && !plant && !input.translucent && input.fluid == 0u {
        let bevel = block_edge_highlight(input.position, input.normal, dist_sq);
        diffuse *= 1.0 + bevel;
    }
    let glow_mask = select(1.0, smoothstep(0.34, 0.75, value), masked_emitter);
    let glow = albedo * max(40.0 * albedo * emission / PHOTON_PI,
        vec3(0.48 * input.emission * glow_mask));
    let color = diffuse + specular + glow;
    if !input.translucent && (contact_ao > 0.0 || terrain_reflections || shadows) {
        let inv_lum = 1.0 / max(dot(color, frame.luminance), 1e-5);
        let ambient_fraction = dot(ambient, frame.luminance) * inv_lum;
        let reflectivity = select(max(wet_film, smoothstep(0.37, 0.79, smoothness)),
            0.0, leaves || plant || input.fluid != 0u);
        var refl_payload = vec3(0.0);
        if reflectivity > 0.0 {
            let no_v = clamp(dot(normal, view), 0.0, 1.0);
            let omv = 1.0 - no_v;
            let omv2 = omv * omv;
            let fresnel5 = omv2 * omv2 * omv;
            var fresnel = mix(reflectance, vec3(1.0), fresnel5);
            if conductor >= 230u && conductor <= 237u {
                fresnel = conductor_fresnel(conductor, no_v) * albedo;
            }
            if wet_film > 0.0 {
                let water_fresnel = mix(vec3(0.02), vec3(1.0), fresnel5);
                fresnel = mix(fresnel, water_fresnel, wet_film);
            }
            refl_payload = fresnel * reflectivity;
        }
        let direct_rgb = max(diffuse_scale * sunlight - diffuse_scale * sss_protected, vec3(0.0)) + foliage + specular;
        let direct_fraction = dot(direct_rgb, frame.luminance) * inv_lum;
        // Write macro/geometric normal to G-buffer (not high-frequency auto-bump) so
        // screen-space contact shadows, GTAO, and SSILVB GI stay 100% flicker-free.
        set_surface(gbuffer_normal, smoothness, surface_payload(refl_payload, ambient_fraction, sky, direct_fraction));
    }
    var shaded = vec4(color, input.albedo.a);
    if input.translucent && (glass || ice || transparent_solid) {
        shaded = refract_translucent(input, normal, albedo, ice, shaded);
    }
    return surface_fog(shaded, input.position);
}
