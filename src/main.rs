use std::f32::consts::{FRAC_PI_2, PI};

use bevy::{
    asset::RenderAssetUsages,
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit},
    pbr::{CascadeShadowConfigBuilder, DistanceFog, FogFalloff, NotShadowCaster},
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology},
    window::PrimaryWindow,
};

// ───────────────────────────── World / terrain ─────────────────────────────

const HALF: f32 = 120.0; // terrain spans [-HALF, HALF] on both axes
const CELL: f32 = 0.4;
const EDGE_FADE: f32 = 25.0;
const CAMERA_LIMIT: f32 = 95.0;
const SKY: (f32, f32, f32) = (0.60, 0.72, 0.85);
const EDGE_GRASS: (f32, f32, f32) = (0.24, 0.38, 0.12);

fn hash(ix: i32, iz: i32, seed: u32) -> f32 {
    let mut h = (ix as u32).wrapping_mul(0x27d4_eb2d)
        ^ (iz as u32).wrapping_mul(0x1656_67b1)
        ^ seed.wrapping_mul(0x9e37_79b1);
    h ^= h >> 15;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h as f32 / u32::MAX as f32
}

fn value_noise(x: f32, z: f32, seed: u32) -> f32 {
    let (fx, fz) = (x.floor(), z.floor());
    let (ix, iz) = (fx as i32, fz as i32);
    let (tx, tz) = (x - fx, z - fz);
    let (ux, uz) = (tx * tx * (3.0 - 2.0 * tx), tz * tz * (3.0 - 2.0 * tz));
    let a = hash(ix, iz, seed);
    let b = hash(ix + 1, iz, seed);
    let c = hash(ix, iz + 1, seed);
    let d = hash(ix + 1, iz + 1, seed);
    (a + (b - a) * ux) + ((c + (d - c) * ux) - (a + (b - a) * ux)) * uz
}

/// Fractal noise, roughly in [0, 1].
fn fbm(mut x: f32, mut z: f32, seed: u32, octaves: u32) -> f32 {
    let (mut sum, mut amp, mut norm) = (0.0, 1.0, 0.0);
    for i in 0..octaves {
        sum += value_noise(x, z, seed + i * 31) * amp;
        norm += amp;
        amp *= 0.5;
        x = x * 2.03 + 17.1;
        z = z * 2.03 - 9.7;
    }
    sum / norm
}

fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn edge_factor(x: f32, z: f32) -> f32 {
    smooth(HALF, HALF - EDGE_FADE, x.abs().max(z.abs()))
}

/// 0 = grass, 1 = mud.
fn mud_mask(x: f32, z: f32) -> f32 {
    let n = fbm(x * 0.035 + 17.0, z * 0.035 - 9.0, 4, 4);
    let detail = fbm(x * 0.45, z * 0.45, 5, 2) - 0.5;
    smooth(0.52, 0.62, n + detail * 0.14)
}

fn height(x: f32, z: f32) -> f32 {
    let broad = (fbm(x * 0.02, z * 0.02, 1, 3) - 0.5) * 1.8;
    let mid = (fbm(x * 0.11, z * 0.11, 2, 3) - 0.5) * 0.55;
    let fine = (fbm(x * 0.7, z * 0.7, 3, 2) - 0.5) * 0.14;
    let mud_dip = -mud_mask(x, z) * 0.28; // mud pools sit in shallow dips
    (broad + mid + fine + mud_dip) * edge_factor(x, z)
}

fn terrain_color(x: f32, z: f32, h: f32) -> Vec3 {
    let mud = mud_mask(x, z);
    let patch = fbm(x * 0.9, z * 0.9, 6, 3);
    let blades = fbm(x * 3.5, z * 3.5, 7, 1);
    let dry = smooth(0.55, 0.75, fbm(x * 0.05, z * 0.05, 8, 2));

    let grass_dark = Vec3::new(0.14, 0.30, 0.08);
    let grass_light = Vec3::new(0.38, 0.52, 0.17);
    let grass_dry = Vec3::new(0.50, 0.50, 0.22);
    let mut grass = grass_dark.lerp(grass_light, (patch * 0.8 + blades * 0.35).clamp(0.0, 1.0));
    grass = grass.lerp(grass_dry, dry * 0.45);

    let mud_dark = Vec3::new(0.17, 0.11, 0.07);
    let mud_light = Vec3::new(0.38, 0.28, 0.17);
    let mut m = mud_dark.lerp(mud_light, (patch * 0.9 + blades * 0.3).clamp(0.0, 1.0));
    m *= 0.75 + 0.25 * smooth(-0.5, 0.2, h); // wetter in the low spots

    let c = grass.lerp(m, mud);
    let edge = Vec3::new(EDGE_GRASS.0, EDGE_GRASS.1, EDGE_GRASS.2);
    c.lerp(edge, 1.0 - edge_factor(x, z))
}

fn srgb_to_linear_arr(c: Vec3) -> [f32; 4] {
    let l = Color::srgb(c.x, c.y, c.z).to_linear();
    [l.red, l.green, l.blue, 1.0]
}

fn build_terrain_mesh() -> Mesh {
    let n = (2.0 * HALF / CELL) as usize;
    let stride = n + 1;
    let at = |i: usize| -HALF + i as f32 * CELL;

    let mut heights = vec![0.0f32; stride * stride];
    for iz in 0..stride {
        for ix in 0..stride {
            heights[iz * stride + ix] = height(at(ix), at(iz));
        }
    }

    let mut positions = Vec::with_capacity(stride * stride);
    let mut normals = Vec::with_capacity(stride * stride);
    let mut uvs = Vec::with_capacity(stride * stride);
    let mut colors = Vec::with_capacity(stride * stride);
    for iz in 0..stride {
        for ix in 0..stride {
            let (x, z) = (at(ix), at(iz));
            let h = heights[iz * stride + ix];
            let hl = heights[iz * stride + ix.saturating_sub(1)];
            let hr = heights[iz * stride + (ix + 1).min(n)];
            let hd = heights[iz.saturating_sub(1) * stride + ix];
            let hu = heights[(iz + 1).min(n) * stride + ix];
            let normal = Vec3::new(hl - hr, 2.0 * CELL, hd - hu).normalize();
            positions.push([x, h, z]);
            normals.push(normal.to_array());
            uvs.push([x * 0.1, z * 0.1]);
            colors.push(srgb_to_linear_arr(terrain_color(x, z, h)));
        }
    }

    let mut indices = Vec::with_capacity(n * n * 6);
    for iz in 0..n {
        for ix in 0..n {
            let i = (iz * stride + ix) as u32;
            let s = stride as u32;
            indices.extend_from_slice(&[i, i + s, i + 1, i + 1, i + s, i + s + 1]);
        }
    }

    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
        .with_inserted_indices(Indices::U32(indices))
}

/// Where a camera ray hits the terrain.
fn pick_ground(ray: Ray3d) -> Option<Vec3> {
    let d = *ray.direction;
    if d.y >= -1e-4 {
        return None;
    }
    let mut y = 0.0;
    let mut p = ray.origin;
    for _ in 0..5 {
        p = ray.origin + d * ((y - ray.origin.y) / d.y);
        y = height(p.x, p.z);
    }
    Some(Vec3::new(p.x, y, p.z))
}

// ───────────────────────────── Components ─────────────────────────────

#[derive(Component)]
struct Hero {
    target: Option<Vec3>,
    speed: f32,
    yaw: f32,
}

#[derive(Component, Clone, Copy, PartialEq)]
enum Limb {
    Body,
    Chest,
    Head,
    HipL,
    HipR,
    KneeL,
    KneeR,
    ShoulderL,
    ShoulderR,
    ElbowL,
    ElbowR,
    Cape,
}

#[derive(Resource, Default)]
struct Anim {
    phase: f32,
    blend: f32,
    time: f32,
}

#[derive(Component)]
struct RtsCamera {
    focus: Vec3,
    yaw: f32,
    zoom: f32,
    zoom_target: f32,
}

#[derive(Component)]
struct MoveMarker {
    t: f32,
}

const MARKER_TIME: f32 = 0.7;

// ───────────────────────────── App ─────────────────────────────

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "SurviveGame".into(),
                resolution: (1280.0_f32, 720.0_f32).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::srgb(SKY.0, SKY.1, SKY.2)))
        .insert_resource(AmbientLight {
            color: Color::srgb(0.75, 0.85, 1.0),
            brightness: 450.0,
            ..default()
        })
        .init_resource::<Anim>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                issue_orders,
                hero_move,
                animate_hero,
                camera_control,
                update_marker,
            )
                .chain(),
        )
        .run();
}

// ───────────────────────────── Setup ─────────────────────────────

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Terrain
    commands.spawn((
        Mesh3d(meshes.add(build_terrain_mesh())),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::WHITE,
            perceptual_roughness: 0.95,
            reflectance: 0.08,
            ..default()
        })),
    ));

    // Endless flat ground past the terrain edge, melts into the fog.
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(3000.0, 3000.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(EDGE_GRASS.0, EDGE_GRASS.1, EDGE_GRASS.2),
            perceptual_roughness: 1.0,
            reflectance: 0.05,
            ..default()
        })),
        Transform::from_xyz(0.0, -0.03, 0.0),
        NotShadowCaster,
    ));

    // Sun
    commands.spawn((
        DirectionalLight {
            illuminance: 11_000.0,
            color: Color::srgb(1.0, 0.95, 0.85),
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::YXZ, -0.9, -0.85, 0.0)),
        CascadeShadowConfigBuilder {
            num_cascades: 4,
            first_cascade_far_bound: 12.0,
            maximum_distance: 140.0,
            ..default()
        }
        .build(),
    ));

    // Camera
    commands.spawn((
        Camera3d::default(),
        Msaa::Sample4,
        Projection::Perspective(PerspectiveProjection {
            fov: 45f32.to_radians(),
            far: 1500.0,
            ..default()
        }),
        DistanceFog {
            color: Color::srgb(SKY.0, SKY.1, SKY.2),
            falloff: FogFalloff::Linear {
                start: 70.0,
                end: 260.0,
            },
            ..default()
        },
        Transform::default(),
        RtsCamera {
            focus: Vec3::ZERO,
            yaw: 0.0,
            zoom: 30.0,
            zoom_target: 30.0,
        },
    ));

    spawn_scatter(&mut commands, &mut meshes, &mut materials);
    spawn_hero(&mut commands, &mut meshes, &mut materials);

    // Move-command marker (reused every click)
    commands.spawn((
        Mesh3d(meshes.add(Annulus::new(0.42, 0.52))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgba(0.3, 1.0, 0.4, 1.0),
            unlit: true,
            alpha_mode: AlphaMode::Blend,
            cull_mode: None,
            ..default()
        })),
        Transform::from_rotation(Quat::from_rotation_x(-FRAC_PI_2)),
        Visibility::Hidden,
        NotShadowCaster,
        MoveMarker { t: 0.0 },
    ));
}

/// Grass tufts and rocks sprinkled over the plain.
fn spawn_scatter(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let blade = meshes.add(Cone::new(0.05, 0.5).mesh().resolution(4));
    let greens: Vec<_> = [
        (0.20, 0.38, 0.10),
        (0.30, 0.46, 0.14),
        (0.42, 0.55, 0.18),
        (0.46, 0.46, 0.20),
    ]
    .iter()
    .map(|&(r, g, b)| {
        materials.add(StandardMaterial {
            base_color: Color::srgb(r, g, b),
            perceptual_roughness: 1.0,
            ..default()
        })
    })
    .collect();

    let span = HALF - 8.0;
    for i in 0..9000 {
        let x = (hash(i, 0, 100) * 2.0 - 1.0) * span;
        let z = (hash(i, 1, 100) * 2.0 - 1.0) * span;
        if mud_mask(x, z) > 0.3 {
            continue;
        }
        let h = height(x, z);
        let mat = &greens[(hash(i, 2, 100) * greens.len() as f32) as usize % greens.len()];
        for k in 0..3 {
            let s = 0.6 + hash(i, 10 + k, 100) * 1.1;
            let yaw = hash(i, 20 + k, 100) * PI * 2.0;
            let tilt_x = (hash(i, 30 + k, 100) - 0.5) * 0.7;
            let tilt_z = (hash(i, 40 + k, 100) - 0.5) * 0.7;
            let ox = (hash(i, 50 + k, 100) - 0.5) * 0.18;
            let oz = (hash(i, 60 + k, 100) - 0.5) * 0.18;
            commands.spawn((
                Mesh3d(blade.clone()),
                MeshMaterial3d(mat.clone()),
                Transform {
                    translation: Vec3::new(x + ox, h + 0.22 * s, z + oz),
                    rotation: Quat::from_rotation_y(yaw)
                        * Quat::from_rotation_x(tilt_x)
                        * Quat::from_rotation_z(tilt_z),
                    scale: Vec3::new(1.0, s, 1.0),
                },
                NotShadowCaster,
            ));
        }
    }

    let rock_mesh = meshes.add(Sphere::new(1.0).mesh().ico(2).unwrap());
    let rock_mats: Vec<_> = [(0.42, 0.40, 0.37), (0.34, 0.32, 0.30), (0.48, 0.44, 0.38)]
        .iter()
        .map(|&(r, g, b)| {
            materials.add(StandardMaterial {
                base_color: Color::srgb(r, g, b),
                perceptual_roughness: 0.9,
                ..default()
            })
        })
        .collect();
    for i in 0..110 {
        let x = (hash(i, 0, 200) * 2.0 - 1.0) * span;
        let z = (hash(i, 1, 200) * 2.0 - 1.0) * span;
        if x * x + z * z < 36.0 {
            continue;
        }
        let s = 0.25 + hash(i, 2, 200) * hash(i, 2, 200) * 1.1;
        commands.spawn((
            Mesh3d(rock_mesh.clone()),
            MeshMaterial3d(rock_mats[i as usize % rock_mats.len()].clone()),
            Transform {
                translation: Vec3::new(x, height(x, z) + s * 0.15, z),
                rotation: Quat::from_euler(
                    EulerRot::YXZ,
                    hash(i, 3, 200) * PI * 2.0,
                    (hash(i, 4, 200) - 0.5) * 0.4,
                    (hash(i, 5, 200) - 0.5) * 0.4,
                ),
                scale: Vec3::new(s * 1.2, s * 0.6, s),
            },
        ));
    }
}

// ───────────────────────────── Hero model ─────────────────────────────

struct Builder<'a, 'w, 's> {
    c: &'a mut Commands<'w, 's>,
    meshes: &'a mut Assets<Mesh>,
}

impl Builder<'_, '_, '_> {
    fn node(&mut self, parent: Entity, t: Transform, limb: Option<Limb>) -> Entity {
        let mut e = self.c.spawn((t, Visibility::default()));
        if let Some(l) = limb {
            e.insert(l);
        }
        let id = e.id();
        self.c.entity(parent).add_child(id);
        id
    }

    fn part(
        &mut self,
        parent: Entity,
        mesh: impl Into<Mesh>,
        mat: &Handle<StandardMaterial>,
        t: Transform,
    ) -> Entity {
        let id = self
            .c
            .spawn((Mesh3d(self.meshes.add(mesh)), MeshMaterial3d(mat.clone()), t))
            .id();
        self.c.entity(parent).add_child(id);
        id
    }
}

fn spawn_hero(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let mut mat = |r: f32, g: f32, b: f32, metallic: f32, rough: f32| {
        materials.add(StandardMaterial {
            base_color: Color::srgb(r, g, b),
            metallic,
            perceptual_roughness: rough,
            ..default()
        })
    };
    let skin = mat(0.85, 0.66, 0.52, 0.0, 0.7);
    let steel = mat(0.72, 0.75, 0.80, 0.55, 0.38);
    let dark_steel = mat(0.40, 0.42, 0.47, 0.6, 0.45);
    let gold = mat(0.95, 0.75, 0.30, 0.7, 0.35);
    let red = mat(0.62, 0.10, 0.10, 0.0, 0.8);
    let leather = mat(0.36, 0.22, 0.12, 0.0, 0.85);
    let cloth = mat(0.20, 0.20, 0.26, 0.0, 0.9);
    let black = mat(0.04, 0.04, 0.05, 0.0, 0.5);

    let hero = Hero {
        target: None,
        speed: 5.5,
        yaw: 0.0,
    };
    let root = commands
        .spawn((
            Transform::from_xyz(0.0, height(0.0, 0.0), 0.0),
            Visibility::default(),
            hero,
        ))
        .id();
    let mut b = Builder { c: commands, meshes };
    let t = Transform::from_xyz;

    // Body: origin at hip height. Everything else hangs off this.
    let body = b.node(root, t(0.0, 1.04, 0.0), Some(Limb::Body));
    b.part(body, Cuboid::new(0.38, 0.18, 0.25), &cloth, t(0.0, -0.03, 0.0));
    b.part(body, Cuboid::new(0.41, 0.06, 0.27), &leather, t(0.0, 0.07, 0.0));
    b.part(body, Cuboid::new(0.07, 0.07, 0.02), &gold, t(0.0, 0.07, 0.14));
    // Front and back skirt flaps
    b.part(body, Cuboid::new(0.30, 0.30, 0.02), &red, t(0.0, -0.16, 0.14));
    b.part(body, Cuboid::new(0.30, 0.30, 0.02), &red, t(0.0, -0.16, -0.14));

    // Legs
    for (side, hip_limb, knee_limb) in [(1.0, Limb::HipL, Limb::KneeL), (-1.0, Limb::HipR, Limb::KneeR)] {
        let hip = b.node(body, t(0.12 * side, -0.08, 0.0), Some(hip_limb));
        b.part(hip, Sphere::new(0.08), &cloth, t(0.0, 0.0, 0.0));
        b.part(hip, Capsule3d::new(0.075, 0.30), &cloth, t(0.0, -0.225, 0.0));
        let knee = b.node(hip, t(0.0, -0.45, 0.0), Some(knee_limb));
        b.part(knee, Sphere::new(0.08), &steel, t(0.0, 0.0, 0.03));
        b.part(knee, Capsule3d::new(0.065, 0.31), &steel, t(0.0, -0.22, 0.0));
        b.part(knee, Cuboid::new(0.14, 0.12, 0.30), &leather, t(0.0, -0.45, 0.06));
        b.part(knee, Cuboid::new(0.15, 0.04, 0.31), &dark_steel, t(0.0, -0.50, 0.06));
    }

    // Torso (twists as one piece)
    let chest = b.node(body, Transform::IDENTITY, Some(Limb::Chest));
    b.part(chest, Cuboid::new(0.36, 0.18, 0.24), &dark_steel, t(0.0, 0.15, 0.0));
    b.part(chest, Cuboid::new(0.48, 0.42, 0.29), &steel, t(0.0, 0.40, 0.0));
    b.part(chest, Cuboid::new(0.14, 0.34, 0.02), &gold, t(0.0, 0.40, 0.15));
    b.part(chest, Cuboid::new(0.30, 0.10, 0.02), &red, t(0.0, 0.22, 0.14));
    b.part(chest, Cylinder::new(0.06, 0.10), &skin, t(0.0, 0.66, 0.0));
    // Pauldrons
    for side in [1.0, -1.0] {
        b.part(
            chest,
            Sphere::new(0.13),
            &steel,
            t(0.31 * side, 0.58, 0.0).with_scale(Vec3::new(1.0, 0.75, 1.1)),
        );
        b.part(
            chest,
            Cuboid::new(0.2, 0.025, 0.2),
            &gold,
            t(0.32 * side, 0.52, 0.0),
        );
    }

    // Head
    let head = b.node(chest, t(0.0, 0.80, 0.0), Some(Limb::Head));
    b.part(head, Sphere::new(0.12), &skin, t(0.0, 0.0, 0.01));
    b.part(
        head,
        Sphere::new(0.14),
        &steel,
        t(0.0, 0.045, -0.01).with_scale(Vec3::new(1.0, 0.9, 1.1)),
    );
    b.part(head, Cuboid::new(0.025, 0.12, 0.02), &steel, t(0.0, 0.0, 0.135));
    for side in [1.0, -1.0] {
        b.part(head, Sphere::new(0.015), &black, t(0.045 * side, -0.01, 0.118));
    }
    b.part(
        head,
        Cuboid::new(0.035, 0.09, 0.28),
        &red,
        t(0.0, 0.17, -0.04).with_rotation(Quat::from_rotation_x(-0.25)),
    );

    // Cape
    let cape = b.node(chest, t(0.0, 0.62, -0.16), Some(Limb::Cape));
    b.part(cape, Cuboid::new(0.44, 0.85, 0.02), &red, t(0.0, -0.42, 0.0));

    // Arms. Character's left (+X) carries the shield, right (-X) the sword.
    for (side, shoulder_limb, elbow_limb) in [
        (1.0, Limb::ShoulderL, Limb::ElbowL),
        (-1.0, Limb::ShoulderR, Limb::ElbowR),
    ] {
        let shoulder = b.node(chest, t(0.32 * side, 0.55, 0.0), Some(shoulder_limb));
        b.part(shoulder, Capsule3d::new(0.055, 0.20), &cloth, t(0.0, -0.16, 0.0));
        let elbow = b.node(shoulder, t(0.0, -0.31, 0.0), Some(elbow_limb));
        b.part(elbow, Capsule3d::new(0.05, 0.17), &steel, t(0.0, -0.14, 0.0));
        b.part(elbow, Sphere::new(0.055), &skin, t(0.0, -0.29, 0.0));

        if side > 0.0 {
            let shield = b.node(
                elbow,
                t(0.09, -0.14, 0.06).with_rotation(Quat::from_rotation_y(-0.5)),
                None,
            );
            let disc = Quat::from_rotation_z(FRAC_PI_2);
            b.part(shield, Cylinder::new(0.27, 0.04), &dark_steel, t(0.0, 0.0, 0.0).with_rotation(disc));
            b.part(shield, Cylinder::new(0.22, 0.045), &red, t(0.012, 0.0, 0.0).with_rotation(disc));
            b.part(shield, Sphere::new(0.06), &gold, t(0.04, 0.0, 0.0));
        } else {
            let sword = b.node(
                elbow,
                t(0.0, -0.29, 0.0).with_rotation(Quat::from_rotation_x(1.7)),
                None,
            );
            b.part(sword, Cylinder::new(0.02, 0.14), &leather, t(0.0, 0.0, 0.0));
            b.part(sword, Sphere::new(0.03), &gold, t(0.0, -0.08, 0.0));
            b.part(sword, Cuboid::new(0.22, 0.03, 0.045), &gold, t(0.0, 0.08, 0.0));
            b.part(sword, Cuboid::new(0.05, 0.85, 0.014), &steel, t(0.0, 0.53, 0.0));
        }
    }
}

// ───────────────────────────── Systems ─────────────────────────────

fn cursor_ground(
    window: &Window,
    camera: &Camera,
    cam_tf: &GlobalTransform,
) -> Option<Vec3> {
    let cursor = window.cursor_position()?;
    let ray = camera.viewport_to_world(cam_tf, cursor).ok()?;
    let p = pick_ground(ray)?;
    Some(Vec3::new(
        p.x.clamp(-HALF + 5.0, HALF - 5.0),
        p.y,
        p.z.clamp(-HALF + 5.0, HALF - 5.0),
    ))
}

fn issue_orders(
    buttons: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<RtsCamera>>,
    mut hero: Query<&mut Hero>,
    mut marker: Query<(&mut MoveMarker, &mut Transform)>,
) {
    let just = buttons.just_pressed(MouseButton::Right) || buttons.just_pressed(MouseButton::Left);
    let held = buttons.pressed(MouseButton::Right);
    if !just && !held {
        return;
    }
    let (Ok(window), Ok((cam, cam_tf)), Ok(mut hero)) =
        (windows.single(), camera.single(), hero.single_mut())
    else {
        return;
    };
    let Some(p) = cursor_ground(window, cam, cam_tf) else {
        return;
    };
    hero.target = Some(p);
    if just {
        if let Ok((mut m, mut tf)) = marker.single_mut() {
            m.t = MARKER_TIME;
            tf.translation = p + Vec3::Y * 0.12;
        }
    }
}

fn hero_move(
    time: Res<Time>,
    mut hero: Query<(&mut Transform, &mut Hero)>,
    mut anim: ResMut<Anim>,
) {
    let Ok((mut tf, mut hero)) = hero.single_mut() else {
        return;
    };
    let dt = time.delta_secs();
    let mut moving = false;

    if let Some(target) = hero.target {
        let to = Vec3::new(target.x - tf.translation.x, 0.0, target.z - tf.translation.z);
        let dist = to.length();
        if dist < 0.06 {
            hero.target = None;
        } else {
            let dir = to / dist;
            let desired = dir.x.atan2(dir.z);
            let mut diff = (desired - hero.yaw + PI).rem_euclid(2.0 * PI) - PI;
            let step = diff.clamp(-12.0 * dt, 12.0 * dt);
            hero.yaw += step;
            diff -= step;
            // Turn first if the target is far off the facing direction.
            if diff.abs() < 1.0 {
                let travel = (hero.speed * dt).min(dist);
                tf.translation += dir * travel;
                moving = true;
            }
        }
    }

    tf.translation.y = height(tf.translation.x, tf.translation.z);
    tf.rotation = Quat::from_rotation_y(hero.yaw);

    anim.time += dt;
    let goal = if moving { 1.0 } else { 0.0 };
    anim.blend += (goal - anim.blend) * (1.0 - (-10.0 * dt).exp());
    if moving {
        anim.phase += hero.speed * 2.4 * dt;
    }
}

fn animate_hero(anim: Res<Anim>, mut limbs: Query<(&Limb, &mut Transform)>) {
    let p = anim.phase;
    let b = anim.blend;
    let breathe = (anim.time * 1.8).sin();
    let rx = Quat::from_rotation_x;
    let ry = Quat::from_rotation_y;
    let rz = Quat::from_rotation_z;

    for (limb, mut tf) in &mut limbs {
        match limb {
            Limb::Body => {
                tf.translation.y = 1.04 + (2.0 * p).cos() * 0.025 * b + breathe * 0.004 * (1.0 - b);
                tf.rotation = rx(0.10 * b) * ry(p.sin() * 0.06 * b);
            }
            Limb::Chest => {
                tf.rotation = ry(-p.sin() * 0.16 * b) * rx(breathe * 0.012);
            }
            Limb::Head => {
                tf.rotation = rx(-0.08 * b) * ry(p.sin() * 0.1 * b);
            }
            Limb::HipL => tf.rotation = rx(-p.sin() * 0.75 * b),
            Limb::HipR => tf.rotation = rx(p.sin() * 0.75 * b),
            Limb::KneeL => tf.rotation = rx((0.05 + 1.0 * p.cos().max(0.0)) * b),
            Limb::KneeR => tf.rotation = rx((0.05 + 1.0 * (-p.cos()).max(0.0)) * b),
            Limb::ShoulderL => {
                tf.rotation = rz(0.07) * rx(p.sin() * 0.5 * b + breathe * 0.02);
            }
            Limb::ShoulderR => {
                tf.rotation = rz(-0.07) * rx(-p.sin() * 0.5 * b - 0.12 + breathe * 0.02);
            }
            Limb::ElbowL => tf.rotation = rx(-(0.5 + 0.2 * b)),
            Limb::ElbowR => tf.rotation = rx(-(0.35 + 0.35 * b) - 0.1 * p.cos() * b),
            Limb::Cape => {
                tf.rotation = rx(0.05 + 0.45 * b + (2.0 * p).sin() * 0.06 * b + breathe * 0.015);
            }
        }
    }
}

fn camera_control(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    windows: Query<&Window, With<PrimaryWindow>>,
    hero: Query<&Transform, (With<Hero>, Without<RtsCamera>)>,
    mut camera: Query<(&mut RtsCamera, &mut Transform), Without<Hero>>,
) {
    let Ok((mut cam, mut tf)) = camera.single_mut() else {
        return;
    };
    let dt = time.delta_secs();

    // Rotate
    let mut spin = 0.0;
    if keys.pressed(KeyCode::KeyQ) {
        spin += 1.0;
    }
    if keys.pressed(KeyCode::KeyE) {
        spin -= 1.0;
    }
    cam.yaw += spin * 1.6 * dt;

    let rot = Quat::from_rotation_y(cam.yaw);
    let forward = rot * Vec3::NEG_Z;
    let right = rot * Vec3::X;

    // Pan: WASD / arrows
    let mut dir = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) || keys.pressed(KeyCode::ArrowUp) {
        dir += forward;
    }
    if keys.pressed(KeyCode::KeyS) || keys.pressed(KeyCode::ArrowDown) {
        dir -= forward;
    }
    if keys.pressed(KeyCode::KeyD) || keys.pressed(KeyCode::ArrowRight) {
        dir += right;
    }
    if keys.pressed(KeyCode::KeyA) || keys.pressed(KeyCode::ArrowLeft) {
        dir -= right;
    }

    // Pan: screen-edge scrolling
    if let Ok(window) = windows.single() {
        if let (true, Some(c)) = (window.focused, window.cursor_position()) {
            let m = 3.0;
            if c.x <= m {
                dir -= right;
            }
            if c.x >= window.width() - m {
                dir += right;
            }
            if c.y <= m {
                dir += forward;
            }
            if c.y >= window.height() - m {
                dir -= forward;
            }
        }
    }

    let boost = if keys.pressed(KeyCode::ShiftLeft) { 2.0 } else { 1.0 };
    let pan_speed = (10.0 + cam.zoom * 0.9) * boost;
    cam.focus += dir.normalize_or_zero() * pan_speed * dt;

    // Pan: middle-mouse drag (grab the ground)
    if buttons.pressed(MouseButton::Middle) {
        let k = cam.zoom * 0.0022;
        cam.focus += -right * motion.delta.x * k + forward * motion.delta.y * k;
    }

    // Recenter on hero
    if keys.pressed(KeyCode::Space) {
        if let Ok(h) = hero.single() {
            let t = 1.0 - (-12.0 * dt).exp();
            cam.focus.x += (h.translation.x - cam.focus.x) * t;
            cam.focus.z += (h.translation.z - cam.focus.z) * t;
        }
    }

    // Zoom
    let scroll_y = match scroll.unit {
        MouseScrollUnit::Line => scroll.delta.y,
        MouseScrollUnit::Pixel => scroll.delta.y / 40.0,
    };
    if scroll_y != 0.0 {
        cam.zoom_target = (cam.zoom_target * 0.88f32.powf(scroll_y)).clamp(6.0, 60.0);
    }
    cam.zoom += (cam.zoom_target - cam.zoom) * (1.0 - (-10.0 * dt).exp());

    cam.focus.x = cam.focus.x.clamp(-CAMERA_LIMIT, CAMERA_LIMIT);
    cam.focus.z = cam.focus.z.clamp(-CAMERA_LIMIT, CAMERA_LIMIT);
    cam.focus.y = height(cam.focus.x, cam.focus.z);

    // Closer = lower angle, farther = more top-down.
    let t = smooth(6.0, 60.0, cam.zoom);
    let pitch = 0.62 + (1.05 - 0.62) * t;
    let offset = rot * Quat::from_rotation_x(-pitch) * Vec3::Z * cam.zoom;
    tf.translation = cam.focus + Vec3::Y * 0.8 + offset;
    tf.look_at(cam.focus + Vec3::Y * 0.8, Vec3::Y);
}

fn update_marker(
    time: Res<Time>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut marker: Query<(
        &mut MoveMarker,
        &mut Transform,
        &mut Visibility,
        &MeshMaterial3d<StandardMaterial>,
    )>,
) {
    let Ok((mut m, mut tf, mut vis, mat)) = marker.single_mut() else {
        return;
    };
    if m.t <= 0.0 {
        *vis = Visibility::Hidden;
        return;
    }
    m.t -= time.delta_secs();
    let k = (m.t / MARKER_TIME).clamp(0.0, 1.0);
    *vis = Visibility::Inherited;
    tf.scale = Vec3::splat(0.6 + 0.9 * k);
    if let Some(mat) = materials.get_mut(&mat.0) {
        mat.base_color = Color::srgba(0.3, 1.0, 0.4, k);
    }
}
