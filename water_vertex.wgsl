override waving_foliage: bool = true;
override temporal_aa: bool = true;

fn wave_position(input: VertexInput) -> vec3<f32> {
    if frame.dimension != 0 || (input.fluid == 0u && input.material == 0u) {
        return input.position;
    }
    let world = input.position + frame.camera_position;
    let horiz_sq = dot(input.position.xz, input.position.xz);

    // Water surface waves
    if water_displacement && wave_strength > 0.0 && input.fluid == 1u {
        let horiz_dist = sqrt(horiz_sq);
        let phase = dot(world.xz, vec2(0.8660254, 0.5)) * 1.9 - 2.9 * frame.time;
        let crest = 0.5 + 0.5 * sin(phase);
        let distance_fade = 1.0 - smoothstep(0.45, 0.92,
            horiz_dist / max(frame.render_distance, 1.0));
        let height = (crest * crest - 0.38) * 0.045 * wave_strength * distance_fade
            * (0.1 + 0.9 * input.light.y);
        return input.position + vec3(0.0, height, 0.0);
    }

    // Wind-driven waving leaves (material 1) and rooted plants (materials 2, 3, 4)
    if waving_foliage && input.material >= 1u && input.material <= 4u && horiz_sq < 7056.0 {
        let horiz_dist = sqrt(horiz_sq);
        let fade = (1.0 - smoothstep(48.0, 84.0, horiz_dist)) * smoothstep(0.15, 0.65, input.light.y);
        if fade > 0.0 {
            let wind_base = weather_wind();
            let storm = frame.rain * 0.75 + frame.thunder * 0.55;
            let wind = 0.45 + 0.85 * wind_base + storm;
            // Slowly veering prevailing wind direction so gusts sweep naturally across fields.
            let wind_angle = 0.55 + 0.32 * sin(frame.time * 0.06);
            let wind_dir = vec2(cos(wind_angle), sin(wind_angle));
            let wind_perp = vec2(-wind_dir.y, wind_dir.x);

            if input.material == 1u {
                // Leaves: gentle 3D canopy rustle driven by the same travelling wind fronts.
                let front_a = sin(dot(world.xz, wind_dir) * 0.14 - frame.time * (1.05 + 0.5 * storm));
                let front_b = sin(dot(world.xz, vec2(-0.06, 0.13)) - frame.time * 0.74 + 1.3);
                let gust_wave = 0.5 + 0.5 * (0.65 * front_a + 0.35 * front_b);
                let gust = 0.55 + 0.75 * gust_wave * gust_wave;
                let t = frame.time * (1.85 + 0.85 * storm);
                let p1 = dot(world, vec3(0.75, 0.45, 0.55)) + t;
                let p2 = dot(world, vec3(-0.42, 0.85, 0.63)) + t * 1.47;
                let sway = vec3(sin(p1) * 0.026 + cos(p2) * 0.012,
                    sin(p2 * 0.85) * 0.014,
                    cos(p1 * 0.92) * 0.024 + sin(p2) * 0.010);
                return input.position + sway * (wind * gust * fade);
            } else {
                // Physically anchored grass, ferns, flowers & crops:
                // Horizon's sprite uv.y is 1.0 at the ground root and 0.0 at the top edge.
                // Using uv.y guarantees roots are 100% locked to the ground (even with
                // Minecraft's random Y block offset) while only upper tips bend.
                let local_top = 1.0 - clamp(input.uv.y, 0.0, 1.0);
                let is_tall_top = input.material == 3u;
                let is_tall_bot = input.material == 4u;
                let is_tall = is_tall_top || is_tall_bot;
                let h = select(local_top, local_top + 1.0, is_tall_top);
                if h <= 0.001 {
                    return input.position;
                }

                // Euler-Bernoulli cantilever stem deflection:
                // 1-block plants & flowers bend smoothly along the lower/mid stem (zero slope
                // at the ground root h=0, coherent rigid tilt across the flower head);
                // 2-block tall plants share an exact C1-continuous curve across the waist (h=1).
                var bend_weight = h * h * (1.5 - 0.5 * h);
                if is_tall {
                    let u = 0.5 * h;
                    bend_weight = 1.72 * (u * u * (6.0 - 4.0 * u + u * u) * (1.0 / 3.0));
                }

                // Evaluate wind on a smooth, continuous world-space field (NO floor() block hash!)
                // so crossed flower/grass sprites that straddle a block boundary due to Minecraft's
                // random XZ plant offset NEVER split or shear apart.
                let along_coord = dot(world.xz, wind_dir);
                let across_coord = dot(world.xz, wind_perp);
                let gust_speed = 1.15 + 0.75 * wind_base + 0.90 * storm;
                let front_a = sin(along_coord * 0.16 - frame.time * gust_speed + across_coord * 0.04);
                let front_b = sin(along_coord * 0.09 - frame.time * (gust_speed * 0.62) - across_coord * 0.11 + 1.7);
                let gust_raw = 0.5 + 0.5 * (0.68 * front_a + 0.32 * front_b);
                let gust = gust_raw * gust_raw;

                // Continuous spatial phase variation (wavelength ~4-5m) gives each clump its own
                // organic spring rhythm while keeping all vertices of a single flower 100% locked together.
                let freq = 2.25 * (1.0 + 0.40 * storm);
                let spatial_phase = sin(world.x * 0.65 + world.z * 0.47) * 1.65
                    + cos(world.x * 0.38 - world.z * 0.71) * 1.15;
                let phase = frame.time * freq + spatial_phase + along_coord * 0.42;
                let primary_osc = sin(phase) + 0.30 * sin(phase * 1.93 + 0.8);
                let cross_osc = cos(phase * 0.87 + 1.4) * 0.42 + 0.22 * sin(phase * 2.31);

                // Asymmetric downwind stem bend: steady lean + gust surge + spring rebound.
                let downwind = ((0.028 + 0.080 * gust) + 0.038 * primary_osc * (0.55 + 0.75 * gust)) * wind;
                let crosswind = (0.022 * cross_osc * (0.50 + 0.65 * gust)) * wind;
                var delta_xz = (wind_dir * downwind + wind_perp * crosswind) * (bend_weight * fade);

                // Player-proximity grass/flower parting: smoothly bend upper blades away from player.
                if horiz_dist < 1.35 && horiz_dist > 0.02 && input.position.y > -2.2 && input.position.y < 0.6 {
                    let push = 1.0 - smoothstep(0.18, 1.35, horiz_dist);
                    let push2 = push * push;
                    let dir = input.position.xz / horiz_dist;
                    delta_xz += dir * (0.44 * push2 * min(bend_weight, 1.25));
                }

                // Exact arc-length conservation: a real grass blade has fixed length, so
                // horizontal bending lowers the tip along a circular arc (no stretching!).
                let max_xz = 0.78 * h;
                let xz_len2 = dot(delta_xz, delta_xz);
                if xz_len2 > max_xz * max_xz {
                    delta_xz *= max_xz * inverseSqrt(xz_len2);
                }
                let final_xz2 = dot(delta_xz, delta_xz);
                let arc_h = max(h, 0.25);
                let drop_y = sqrt(max(arc_h * arc_h - final_xz2, 0.04 * arc_h * arc_h)) - arc_h;
                return input.position + vec3(delta_xz.x, drop_y, delta_xz.y);
            }
        }
    }

    // Viscous ocean-current aquatic flora:
    // material 5 = seagrass, kelp, kelp_plant, coral & coral fans
    // material 6 = tall_seagrass_top, material 7 = tall_seagrass_bottom
    if waving_foliage && input.material >= 5u && input.material <= 7u && horiz_sq < 5184.0 {
        let fade = 1.0 - smoothstep(1764.0, 5184.0, horiz_sq);
        if fade > 0.0 {
            let local_top = 1.0 - clamp(input.uv.y, 0.0, 1.0);
            let is_tall_top = input.material == 6u;
            let is_tall_bot = input.material == 7u;
            let is_tall = is_tall_top || is_tall_bot;

            // Slow, smoothly veering underwater ocean swell (unaffected by surface rain gusts).
            let curr_angle = 0.85 + 0.45 * sin(frame.time * 0.045);
            let curr_dir = vec2(cos(curr_angle), sin(curr_angle));
            let curr_perp = vec2(-curr_dir.y, curr_dir.x);

            // Continuous travelling vertical & horizontal wave phase (no floor() seams)
            // so stacked kelp_plant columns and crossed seagrass/coral sprites stay 100% locked.
            let spatial_phase = sin(world.x * 0.55 + world.z * 0.41) * 1.4;
            let swell_a = sin(dot(world.xz, curr_dir) * 0.22 + world.y * 0.72 - frame.time * 0.92 + spatial_phase);
            let swell_b = cos(dot(world.xz, curr_perp) * 0.19 + world.y * 0.54 - frame.time * 0.64 - spatial_phase * 0.8);
            let surge = 0.5 + 0.5 * sin(dot(world.xz, curr_dir) * 0.09 - frame.time * 0.42);

            var bend_weight: f32;
            var h = local_top;
            if is_tall {
                h = select(local_top, local_top + 1.0, is_tall_top);
                if h <= 0.001 {
                    return input.position;
                }
                let u = 0.5 * h;
                bend_weight = 1.65 * (u * u * (6.0 - 4.0 * u + u * u) * (1.0 / 3.0));
            } else {
                // Single-block seagrass / coral / kelp: keep root anchored for seagrass/coral
                // while giving continuous gentle ribbon wave along kelp columns.
                let base_y = fract(world.y);
                let ribbon = 0.30 + 0.70 * smoothstep(0.0, 0.85, local_top);
                bend_weight = select(local_top * local_top, ribbon, base_y > 0.02 || local_top > 0.05) * 0.85;
            }

            let along_disp = (0.045 + 0.075 * surge) * swell_a + 0.028 * swell_b;
            let cross_disp = 0.042 * swell_b;
            var delta_xz = (curr_dir * along_disp + curr_perp * cross_disp) * (bend_weight * fade);

            // Fluid wake parting when the player swims through kelp or seagrass beds.
            if horiz_sq < 2.4025 && horiz_sq > 0.0004 && input.position.y > -2.4 && input.position.y < 1.0 {
                let horiz_dist = sqrt(horiz_sq);
                let push = 1.0 - smoothstep(0.20, 1.55, horiz_dist);
                let dir = input.position.xz / horiz_dist;
                delta_xz += dir * (0.38 * push * push * min(bend_weight, 1.25));
            }

            let max_xz = 0.42 * max(h, 0.5);
            let xz_len2 = dot(delta_xz, delta_xz);
            if xz_len2 > max_xz * max_xz {
                delta_xz *= max_xz * inverseSqrt(xz_len2);
            }
            let drop_y = -0.18 * dot(delta_xz, delta_xz);
            return input.position + vec3(delta_xz.x, drop_y, delta_xz.y);
        }
    }
    return input.position;
}

// Pack rendering uses an unjittered camera. Offset the block raster in its
// image plane, at constant clip depth, using an 8-sample Halton(2, 3) sequence
// covering the full pixel footprint (Decima-style fast convergence + 8x SSAA).
fn vertex(input: VertexInput) -> vec3<f32> {
    let position = wave_position(input);
    // Never shift water vertices in world-space Y/X with TAA jitter: at mid/far distance
    // world-space vertical jitter lifts water off frame.water_plane and breaks planar mirror stability.
    if !temporal_aa || input.fluid == 1u { return position; }
    let phase = frame.frame_index & 7u;
    let pattern = array(
        vec2( 0.00000, -0.15000),
        vec2(-0.22500,  0.15000),
        vec2( 0.22500, -0.35000),
        vec2(-0.33750, -0.05000),
        vec2( 0.11250,  0.25000),
        vec2(-0.11250, -0.25000),
        vec2( 0.33750,  0.05000),
        vec2(-0.39375,  0.35000),
    );
    // Keep alpha-tested foliage jitter slightly tighter (0.55x) so leaf shadow edges don't strobe.
    let jitter_scale = select(1.0, 0.55, input.material >= 1u && input.material <= 7u);
    let view_z = frame.view[0].z * position.x + frame.view[1].z * position.y
        + frame.view[2].z * position.z + frame.view[3].z;
    let clip_w = frame.projection[2].w * view_z + frame.projection[3].w;
    let ndc = (pattern[phase] * jitter_scale) * vec2(2.0, -2.0) / max(frame.resolution, vec2(1.0));
    let ox = frame.projection_inverse[0].x * ndc.x * clip_w;
    let oy = frame.projection_inverse[1].y * ndc.y * clip_w;
    return position + frame.view_inverse[0].xyz * ox + frame.view_inverse[1].xyz * oy;
}
