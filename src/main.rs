use std::f32::consts::{FRAC_PI_2, PI};

use bevy::{
    asset::RenderAssetUsages,
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit},
    pbr::{CascadeShadowConfigBuilder, DistanceFog, FogFalloff, NotShadowCaster},
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology},
    window::{MonitorSelection, PrimaryWindow, WindowMode},
};

// ───────────────────────────── World / terrain ─────────────────────────────

const HALF: f32 = 40.0; // terrain spans [-HALF, HALF] on both axes
const CELL: f32 = 1.0;
const EDGE_FADE: f32 = 10.0;
const CAMERA_LIMIT: f32 = 28.0;
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
    let detail = fbm(x * 0.3, z * 0.3, 5, 2) - 0.5;
    smooth(0.52, 0.62, n + detail * 0.14)
}

fn height(x: f32, z: f32) -> f32 {
    let broad = (fbm(x * 0.02, z * 0.02, 1, 3) - 0.5) * 1.8;
    let mid = (fbm(x * 0.11, z * 0.11, 2, 3) - 0.5) * 0.55;
    let fine = (fbm(x * 0.35, z * 0.35, 3, 2) - 0.5) * 0.16;
    let mud_dip = -mud_mask(x, z) * 0.28; // mud pools sit in shallow dips
    (broad + mid + fine + mud_dip) * edge_factor(x, z)
}

fn terrain_color(x: f32, z: f32, h: f32) -> Vec3 {
    let mud = mud_mask(x, z);
    let patch = fbm(x * 0.5, z * 0.5, 6, 2);
    let blades = fbm(x * 1.2, z * 1.2, 7, 1);
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

const CHUNKS: usize = 8; // terrain is split into CHUNKS x CHUNKS meshes so culling works

fn build_terrain_chunks() -> Vec<Mesh> {
    let n = (2.0 * HALF / CELL).round() as usize;
    let stride = n + 1;
    let per = n / CHUNKS;
    let at = |i: usize| -HALF + i as f32 * CELL;

    let mut heights = vec![0.0f32; stride * stride];
    for iz in 0..stride {
        for ix in 0..stride {
            heights[iz * stride + ix] = height(at(ix), at(iz));
        }
    }

    let mut out = Vec::with_capacity(CHUNKS * CHUNKS);
    for cz in 0..CHUNKS {
        for cx in 0..CHUNKS {
            let (x0, z0) = (cx * per, cz * per);
            let side = per + 1;
            let mut positions = Vec::with_capacity(side * side);
            let mut normals = Vec::with_capacity(side * side);
            let mut uvs = Vec::with_capacity(side * side);
            let mut colors = Vec::with_capacity(side * side);
            for iz in z0..=z0 + per {
                for ix in x0..=x0 + per {
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
            let mut indices = Vec::with_capacity(per * per * 6);
            for iz in 0..per {
                for ix in 0..per {
                    let i = (iz * side + ix) as u32;
                    let s = side as u32;
                    indices.extend_from_slice(&[i, i + s, i + 1, i + 1, i + s, i + s + 1]);
                }
            }
            out.push(
                Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
                    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
                    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
                    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
                    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
                    .with_inserted_indices(Indices::U32(indices)),
            );
        }
    }
    out
}

/// All grass blades, baked into one mesh per terrain chunk (a handful of draw calls
/// instead of tens of thousands of entities).
fn build_grass_chunks() -> Vec<Mesh> {
    #[derive(Default)]
    struct Bucket {
        pos: Vec<[f32; 3]>,
        nor: Vec<[f32; 3]>,
        col: Vec<[f32; 4]>,
        idx: Vec<u32>,
    }
    let mut buckets: Vec<Bucket> = (0..CHUNKS * CHUNKS).map(|_| Bucket::default()).collect();
    let chunk_size = 2.0 * HALF / CHUNKS as f32;
    let span = HALF - 8.0;
    let palette = [
        Vec3::new(0.20, 0.38, 0.10),
        Vec3::new(0.30, 0.46, 0.14),
        Vec3::new(0.42, 0.55, 0.18),
        Vec3::new(0.46, 0.46, 0.20),
    ];

    for i in 0..1800 {
        let x = (hash(i, 0, 100) * 2.0 - 1.0) * span;
        let z = (hash(i, 1, 100) * 2.0 - 1.0) * span;
        if mud_mask(x, z) > 0.3 {
            continue;
        }
        let h = height(x, z);
        let cx = (((x + HALF) / chunk_size) as usize).min(CHUNKS - 1);
        let cz = (((z + HALF) / chunk_size) as usize).min(CHUNKS - 1);
        let bucket = &mut buckets[cz * CHUNKS + cx];
        let base = palette[(hash(i, 2, 100) * palette.len() as f32) as usize % palette.len()];

        for k in 0..5 {
            let s = 0.6 + hash(i, 10 + k, 100) * 1.0;
            let a = hash(i, 20 + k, 100) * PI * 2.0;
            let lean = Vec3::new(
                (hash(i, 30 + k, 100) - 0.5) * 0.3,
                0.0,
                (hash(i, 40 + k, 100) - 0.5) * 0.3,
            );
            let c = Vec3::new(
                x + (hash(i, 50 + k, 100) - 0.5) * 0.25,
                h - 0.02,
                z + (hash(i, 60 + k, 100) - 0.5) * 0.25,
            );
            let r = Vec3::new(a.cos(), 0.0, a.sin()) * 0.045 * s;
            let tip = c + lean * s + Vec3::Y * 0.5 * s;
            let v0 = bucket.pos.len() as u32;
            for (p, shade) in [(c - r, 0.55), (c + r, 0.55), (tip, 1.15)] {
                bucket.pos.push(p.to_array());
                bucket.nor.push([0.0, 1.0, 0.0]);
                bucket.col.push(srgb_to_linear_arr((base * shade).min(Vec3::ONE)));
            }
            bucket.idx.extend_from_slice(&[v0, v0 + 1, v0 + 2]);
        }
    }

    buckets
        .into_iter()
        .filter(|b| !b.pos.is_empty())
        .map(|b| {
            Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
                .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, b.pos)
                .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, b.nor)
                .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, b.col)
                .with_inserted_indices(Indices::U32(b.idx))
        })
        .collect()
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
    attack: Option<Entity>,
    aim: Option<Vec3>,
    cooldown: f32,
    speed: f32,
    yaw: f32,
}

#[derive(Component)]
struct Health {
    cur: f32,
    max: f32,
}

#[derive(Component)]
struct Enemy {
    yaw: f32,
    phase: f32,
    blend: f32,
    cooldown: f32,
}

#[derive(Component, Clone, Copy)]
enum EKind {
    Body,
    Head,
    HipL,
    HipR,
    KneeL,
    KneeR,
}

#[derive(Component)]
struct ELimb {
    owner: Entity,
    kind: EKind,
}

#[derive(Component)]
struct Projectile {
    vel: Vec3,
    life: f32,
    damage: f32,
    from_player: bool,
}

#[derive(Component)]
struct Flash {
    life: f32,
    max: f32,
    size: f32,
}

#[derive(Component)]
struct HealthBar {
    owner: Entity,
}

#[derive(Component)]
struct HealthFill {
    width: f32,
}

#[derive(Resource)]
struct Fx {
    bullet: Handle<Mesh>,
    player_bullet: Handle<StandardMaterial>,
    enemy_bullet: Handle<StandardMaterial>,
    spark: Handle<Mesh>,
    muzzle: Handle<StandardMaterial>,
    hit: Handle<StandardMaterial>,
    boom: Handle<StandardMaterial>,
}

#[derive(Resource)]
struct Spawner {
    timer: f32,
    n: u32,
}

const HERO_HP: f32 = 100.0;
const HERO_RANGE: f32 = 15.0;
const HERO_DAMAGE: f32 = 10.0;
const HERO_FIRE_INTERVAL: f32 = 0.25;
const ENEMY_HP: f32 = 50.0;
const ENEMY_SPEED: f32 = 2.6;
const ENEMY_STOP_DIST: f32 = 10.0;
const ENEMY_RANGE: f32 = 16.0;
const ENEMY_DAMAGE: f32 = 7.0;
const ENEMY_FIRE_INTERVAL: f32 = 1.1;
const MAX_ENEMIES: usize = 6;
const SPAWN_INTERVAL: f32 = 3.5;

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
    Rifle,
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

const MARKER_TIME: f32 = 0.9;

// ───────────────────────────── App ─────────────────────────────

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "SurviveGame".into(),
                mode: WindowMode::BorderlessFullscreen(MonitorSelection::Primary),
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
        .insert_resource(Spawner { timer: 2.0, n: 0 })
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                quit_on_escape,
                issue_orders,
                hero_combat,
                hero_move,
                enemy_spawner,
                enemy_ai,
                projectiles,
                update_flashes,
                reap,
                animate_hero,
                animate_enemies,
                camera_control,
                update_health_bars,
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
    // Terrain (chunked so off-screen parts are culled)
    let terrain_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 0.95,
        reflectance: 0.08,
        ..default()
    });
    for chunk in build_terrain_chunks() {
        commands.spawn((Mesh3d(meshes.add(chunk)), MeshMaterial3d(terrain_mat.clone())));
    }

    // Grass blades
    let grass_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 1.0,
        reflectance: 0.05,
        cull_mode: None,
        double_sided: true,
        ..default()
    });
    for chunk in build_grass_chunks() {
        commands.spawn((
            Mesh3d(meshes.add(chunk)),
            MeshMaterial3d(grass_mat.clone()),
            NotShadowCaster,
        ));
    }

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
            num_cascades: 3,
            first_cascade_far_bound: 14.0,
            maximum_distance: 80.0,
            ..default()
        }
        .build(),
    ));

    // Camera
    commands.spawn((
        Camera3d::default(),
        Msaa::Sample2,
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
            zoom: 22.0,
            zoom_target: 22.0,
        },
    ));

    spawn_scatter(&mut commands, &mut meshes, &mut materials);
    let hero = spawn_hero(&mut commands, &mut meshes, &mut materials);
    spawn_health_bar(&mut commands, &mut meshes, &mut materials, hero, 1.6, Color::srgb(0.2, 0.9, 0.3));

    let glow = |c: Color, e: LinearRgba, materials: &mut Assets<StandardMaterial>| {
        materials.add(StandardMaterial {
            base_color: c,
            emissive: e,
            unlit: true,
            ..default()
        })
    };
    commands.insert_resource(Fx {
        bullet: meshes.add(Cuboid::new(0.07, 0.07, 0.8)),
        player_bullet: glow(Color::srgb(0.4, 0.9, 1.0), LinearRgba::rgb(2.0, 8.0, 10.0), &mut materials),
        enemy_bullet: glow(Color::srgb(1.0, 0.2, 0.1), LinearRgba::rgb(10.0, 1.0, 0.5), &mut materials),
        spark: meshes.add(Sphere::new(0.5).mesh().ico(1).unwrap()),
        muzzle: glow(Color::srgb(1.0, 0.95, 0.6), LinearRgba::rgb(8.0, 7.0, 3.0), &mut materials),
        hit: glow(Color::srgb(1.0, 0.8, 0.4), LinearRgba::rgb(8.0, 5.0, 1.0), &mut materials),
        boom: glow(Color::srgb(1.0, 0.5, 0.15), LinearRgba::rgb(10.0, 3.5, 0.5), &mut materials),
    });

    // Move-command marker (reused every click): ring + center dot
    let marker_mat = materials.add(StandardMaterial {
        base_color: Color::srgba(0.3, 1.0, 0.4, 1.0),
        unlit: true,
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        ..default()
    });
    let marker = commands
        .spawn((
            Mesh3d(meshes.add(Annulus::new(0.6, 0.8))),
            MeshMaterial3d(marker_mat.clone()),
            Transform::from_rotation(Quat::from_rotation_x(-FRAC_PI_2)),
            Visibility::Hidden,
            NotShadowCaster,
            MoveMarker { t: 0.0 },
        ))
        .id();
    let dot = commands
        .spawn((
            Mesh3d(meshes.add(Circle::new(0.14))),
            MeshMaterial3d(marker_mat),
            Transform::default(),
            NotShadowCaster,
        ))
        .id();
    commands.entity(marker).add_child(dot);
}

/// Grass tufts and rocks sprinkled over the plain.
fn spawn_scatter(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
) {
    let span = HALF - 8.0;
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
    for i in 0..25 {
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
) -> Entity {
    let mut mat = |r: f32, g: f32, b: f32, metallic: f32, rough: f32| {
        materials.add(StandardMaterial {
            base_color: Color::srgb(r, g, b),
            metallic,
            perceptual_roughness: rough,
            ..default()
        })
    };
    let armor = mat(0.24, 0.38, 0.52, 0.35, 0.5);
    let armor_dark = mat(0.15, 0.24, 0.34, 0.4, 0.5);
    let under = mat(0.10, 0.11, 0.13, 0.1, 0.8);
    let trim = mat(0.90, 0.62, 0.10, 0.3, 0.45);
    let gun = mat(0.17, 0.18, 0.20, 0.8, 0.4);
    let glove = mat(0.08, 0.08, 0.09, 0.0, 0.9);
    let visor = materials.add(StandardMaterial {
        base_color: Color::srgb(0.1, 0.8, 0.95),
        emissive: LinearRgba::rgb(0.3, 3.0, 4.0),
        metallic: 0.2,
        perceptual_roughness: 0.2,
        ..default()
    });
    let glow = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.5, 0.2),
        emissive: LinearRgba::rgb(4.0, 1.5, 0.3),
        ..default()
    });

    let hero = Hero {
        target: None,
        attack: None,
        aim: None,
        cooldown: 0.0,
        speed: 5.5,
        yaw: 0.0,
    };
    let root = commands
        .spawn((
            Transform::from_xyz(0.0, height(0.0, 0.0), 0.0),
            Visibility::default(),
            hero,
            Health {
                cur: HERO_HP,
                max: HERO_HP,
            },
        ))
        .id();
    let mut b = Builder { c: commands, meshes };
    let t = Transform::from_xyz;

    // Body: origin at hip height. Everything else hangs off this.
    let body = b.node(root, t(0.0, 1.04, 0.0), Some(Limb::Body));
    b.part(body, Cuboid::new(0.40, 0.18, 0.28), &under, t(0.0, -0.03, 0.0));
    b.part(body, Cuboid::new(0.44, 0.07, 0.30), &armor_dark, t(0.0, 0.07, 0.0));
    b.part(body, Cuboid::new(0.10, 0.08, 0.03), &trim, t(0.0, 0.07, 0.16));
    // Belt pouches and hip guards
    for side in [1.0, -1.0] {
        b.part(body, Cuboid::new(0.08, 0.12, 0.12), &gun, t(0.23 * side, 0.0, 0.06));
        b.part(body, Cuboid::new(0.06, 0.2, 0.26), &armor, t(0.22 * side, -0.1, -0.02));
    }
    b.part(body, Cuboid::new(0.28, 0.14, 0.04), &armor, t(0.0, -0.14, 0.15));

    // Legs
    for (side, hip_limb, knee_limb) in [(1.0, Limb::HipL, Limb::KneeL), (-1.0, Limb::HipR, Limb::KneeR)] {
        let hip = b.node(body, t(0.13 * side, -0.08, 0.0), Some(hip_limb));
        b.part(hip, Sphere::new(0.09), &under, t(0.0, 0.0, 0.0));
        b.part(hip, Capsule3d::new(0.08, 0.28), &under, t(0.0, -0.225, 0.0));
        b.part(hip, Cuboid::new(0.17, 0.26, 0.19), &armor, t(0.0, -0.2, 0.02));
        let knee = b.node(hip, t(0.0, -0.45, 0.0), Some(knee_limb));
        b.part(knee, Sphere::new(0.1), &armor_dark, t(0.0, 0.0, 0.05));
        b.part(knee, Capsule3d::new(0.065, 0.3), &under, t(0.0, -0.22, 0.0));
        b.part(knee, Cuboid::new(0.17, 0.34, 0.2), &armor, t(0.0, -0.24, 0.02));
        b.part(knee, Cuboid::new(0.17, 0.12, 0.34), &gun, t(0.0, -0.45, 0.06));
        b.part(knee, Cuboid::new(0.18, 0.06, 0.12), &armor_dark, t(0.0, -0.40, 0.17));
    }

    // Torso (twists as one piece)
    let chest = b.node(body, Transform::IDENTITY, Some(Limb::Chest));
    b.part(chest, Cuboid::new(0.36, 0.2, 0.26), &under, t(0.0, 0.15, 0.0));
    b.part(chest, Cuboid::new(0.54, 0.44, 0.34), &armor, t(0.0, 0.41, 0.0));
    b.part(chest, Cuboid::new(0.50, 0.12, 0.30), &armor_dark, t(0.0, 0.23, 0.0));
    b.part(chest, Cuboid::new(0.12, 0.30, 0.02), &trim, t(0.0, 0.42, 0.175));
    b.part(chest, Cuboid::new(0.14, 0.12, 0.10), &armor_dark, t(0.0, 0.64, 0.0)); // collar
    // Shoulder pads
    for side in [1.0, -1.0] {
        b.part(
            chest,
            Sphere::new(0.15),
            &armor,
            t(0.35 * side, 0.58, 0.0).with_scale(Vec3::new(1.0, 0.8, 1.15)),
        );
        b.part(chest, Cuboid::new(0.05, 0.1, 0.24), &trim, t(0.43 * side, 0.58, 0.0));
    }
    // Backpack with thrusters
    b.part(chest, Cuboid::new(0.34, 0.44, 0.18), &armor_dark, t(0.0, 0.36, -0.26));
    for side in [1.0, -1.0] {
        b.part(chest, Cylinder::new(0.045, 0.2), &gun, t(0.1 * side, 0.62, -0.28));
        b.part(chest, Cylinder::new(0.03, 0.03), &glow, t(0.1 * side, 0.1, -0.26));
    }
    b.part(chest, Cuboid::new(0.02, 0.5, 0.02), &gun, t(0.14, 0.9, -0.3)); // antenna

    // Head: helmet with glowing visor
    let head = b.node(chest, t(0.0, 0.80, 0.0), Some(Limb::Head));
    b.part(head, Sphere::new(0.16), &armor, t(0.0, 0.02, -0.01).with_scale(Vec3::new(1.0, 0.95, 1.1)));
    b.part(head, Cuboid::new(0.2, 0.09, 0.08), &visor, t(0.0, 0.03, 0.12));
    b.part(head, Cuboid::new(0.14, 0.07, 0.1), &armor_dark, t(0.0, -0.07, 0.1));
    b.part(head, Cuboid::new(0.04, 0.05, 0.22), &trim, t(0.0, 0.17, -0.02));
    for side in [1.0, -1.0] {
        b.part(head, Cylinder::new(0.05, 0.05), &armor_dark, t(0.15 * side, 0.0, 0.0).with_rotation(Quat::from_rotation_z(FRAC_PI_2)));
    }

    // Assault rifle, held across the chest with both hands.
    let rifle = b.node(chest, t(-0.03, 0.27, 0.46), Some(Limb::Rifle));
    b.part(rifle, Cuboid::new(0.08, 0.12, 0.42), &gun, t(0.0, 0.0, 0.0)); // receiver
    b.part(rifle, Cuboid::new(0.07, 0.14, 0.22), &armor_dark, t(0.0, -0.02, -0.3)); // stock
    b.part(rifle, Cuboid::new(0.16, 0.09, 0.28), &armor, t(0.0, -0.01, 0.34)); // handguard
    b.part(rifle, Cylinder::new(0.02, 0.4), &gun, t(0.0, 0.0, 0.7).with_rotation(Quat::from_rotation_x(FRAC_PI_2)));
    b.part(rifle, Cylinder::new(0.03, 0.08), &gun, t(0.0, 0.0, 0.92).with_rotation(Quat::from_rotation_x(FRAC_PI_2)));
    b.part(rifle, Cuboid::new(0.05, 0.2, 0.08), &gun, t(0.0, -0.14, 0.04)); // magazine
    b.part(rifle, Cuboid::new(0.06, 0.06, 0.18), &gun, t(0.0, 0.09, 0.0)); // sight
    b.part(rifle, Cuboid::new(0.085, 0.02, 0.2), &glow, t(0.0, 0.065, 0.2)); // energy strip

    // Arms: both hands on the rifle (left is +X).
    for (side, shoulder_limb, elbow_limb) in [
        (1.0, Limb::ShoulderL, Limb::ElbowL),
        (-1.0, Limb::ShoulderR, Limb::ElbowR),
    ] {
        let shoulder = b.node(chest, t(0.35 * side, 0.55, 0.0), Some(shoulder_limb));
        b.part(shoulder, Capsule3d::new(0.06, 0.2), &under, t(0.0, -0.16, 0.0));
        b.part(shoulder, Cuboid::new(0.13, 0.16, 0.13), &armor, t(0.0, -0.15, 0.0));
        let elbow = b.node(shoulder, t(0.0, -0.31, 0.0), Some(elbow_limb));
        b.part(elbow, Sphere::new(0.07), &armor_dark, t(0.0, 0.0, 0.0));
        b.part(elbow, Capsule3d::new(0.055, 0.16), &under, t(0.0, -0.14, 0.0));
        b.part(elbow, Cuboid::new(0.12, 0.2, 0.12), &armor, t(0.0, -0.15, 0.0));
        b.part(elbow, Sphere::new(0.065), &glove, t(0.0, -0.31, 0.0));
    }
    root
}

impl Builder<'_, '_, '_> {
    fn enode(&mut self, parent: Entity, t: Transform, owner: Entity, kind: Option<EKind>) -> Entity {
        let mut e = self.c.spawn((t, Visibility::default()));
        if let Some(kind) = kind {
            e.insert(ELimb { owner, kind });
        }
        let id = e.id();
        self.c.entity(parent).add_child(id);
        id
    }
}

/// A scrappy tan battle droid: thin limbs, hunched, long snout head, blaster.
fn spawn_enemy(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    pos: Vec3,
    yaw: f32,
) -> Entity {
    let mut mat = |r: f32, g: f32, b: f32, metallic: f32, rough: f32| {
        materials.add(StandardMaterial {
            base_color: Color::srgb(r, g, b),
            metallic,
            perceptual_roughness: rough,
            ..default()
        })
    };
    let tan = mat(0.76, 0.63, 0.38, 0.1, 0.55);
    let dark = mat(0.20, 0.18, 0.17, 0.5, 0.5);
    let blaster = mat(0.10, 0.10, 0.11, 0.7, 0.4);
    let eye = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.7, 0.1),
        emissive: LinearRgba::rgb(3.0, 1.5, 0.1),
        ..default()
    });

    let root = commands
        .spawn((
            Transform::from_translation(pos).with_rotation(Quat::from_rotation_y(yaw)),
            Visibility::default(),
            Enemy {
                yaw,
                phase: hash(pos.x as i32, pos.z as i32, 9) * 6.0,
                blend: 1.0,
                cooldown: 0.5 + hash(pos.z as i32, pos.x as i32, 4),
            },
            Health {
                cur: ENEMY_HP,
                max: ENEMY_HP,
            },
        ))
        .id();
    let mut b = Builder { c: commands, meshes };
    let t = Transform::from_xyz;

    let body = b.enode(root, t(0.0, 0.95, 0.0), root, Some(EKind::Body));
    b.part(body, Cuboid::new(0.22, 0.12, 0.15), &dark, t(0.0, 0.0, 0.0));
    b.part(body, Cuboid::new(0.2, 0.36, 0.12), &tan, t(0.0, 0.26, 0.0));
    b.part(body, Cuboid::new(0.27, 0.13, 0.14), &tan, t(0.0, 0.40, 0.01));
    b.part(body, Cuboid::new(0.14, 0.24, 0.1), &dark, t(0.0, 0.3, -0.12)); // backpack
    b.part(body, Cylinder::new(0.035, 0.26), &dark, t(0.0, 0.6, 0.02)); // neck

    let head = b.enode(body, t(0.0, 0.76, 0.04), root, Some(EKind::Head));
    b.part(head, Sphere::new(0.11), &tan, t(0.0, 0.0, 0.07).with_scale(Vec3::new(0.8, 0.8, 1.9)));
    b.part(head, Cuboid::new(0.05, 0.1, 0.12), &dark, t(0.0, 0.03, -0.1)); // back crest
    for side in [1.0, -1.0] {
        b.part(head, Sphere::new(0.025), &eye, t(0.06 * side, 0.02, 0.16));
    }

    // Blaster held out front in both hands.
    b.part(body, Cuboid::new(0.06, 0.1, 0.5), &blaster, t(0.05, 0.28, 0.42));
    b.part(body, Cylinder::new(0.015, 0.3), &blaster, t(0.05, 0.3, 0.8).with_rotation(Quat::from_rotation_x(FRAC_PI_2)));

    for side in [1.0, -1.0] {
        let shoulder = b.enode(body, t(0.15 * side, 0.42, 0.0), root, None);
        let upper = b.enode(
            shoulder,
            Transform::from_rotation(Quat::from_rotation_z(-0.35 * side) * Quat::from_rotation_x(-1.1)),
            root,
            None,
        );
        b.part(upper, Cylinder::new(0.025, 0.28), &dark, t(0.0, -0.14, 0.0));
        let elbow = b.enode(upper, t(0.0, -0.28, 0.0).with_rotation(Quat::from_rotation_x(-0.5)), root, None);
        b.part(elbow, Cylinder::new(0.025, 0.28), &tan, t(0.0, -0.14, 0.0));
        b.part(elbow, Sphere::new(0.04), &dark, t(0.0, -0.29, 0.0));
    }

    for (side, hip_kind, knee_kind) in [(1.0, EKind::HipL, EKind::KneeL), (-1.0, EKind::HipR, EKind::KneeR)] {
        let hip = b.enode(body, t(0.07 * side, -0.06, 0.0), root, Some(hip_kind));
        b.part(hip, Cylinder::new(0.03, 0.42), &dark, t(0.0, -0.21, 0.0));
        b.part(hip, Cuboid::new(0.07, 0.2, 0.08), &tan, t(0.0, -0.15, 0.0));
        let knee = b.enode(hip, t(0.0, -0.43, 0.0), root, Some(knee_kind));
        b.part(knee, Sphere::new(0.04), &dark, t(0.0, 0.0, 0.0));
        b.part(knee, Cylinder::new(0.025, 0.42), &dark, t(0.0, -0.21, 0.0));
        b.part(knee, Cuboid::new(0.06, 0.22, 0.07), &tan, t(0.0, -0.15, 0.01));
        b.part(knee, Cuboid::new(0.09, 0.05, 0.2), &tan, t(0.0, -0.44, 0.05));
    }
    root
}

fn spawn_health_bar(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<StandardMaterial>,
    owner: Entity,
    width: f32,
    color: Color,
) {
    let unlit = |c: Color| StandardMaterial {
        base_color: c,
        unlit: true,
        cull_mode: None,
        ..default()
    };
    let bg = commands
        .spawn((
            Mesh3d(meshes.add(Rectangle::new(width + 0.08, 0.2))),
            MeshMaterial3d(materials.add(unlit(Color::srgb(0.04, 0.04, 0.05)))),
            Transform::default(),
            NotShadowCaster,
        ))
        .id();
    let fill = commands
        .spawn((
            Mesh3d(meshes.add(Rectangle::new(width, 0.13))),
            MeshMaterial3d(materials.add(unlit(color))),
            Transform::from_xyz(0.0, 0.0, 0.004),
            HealthFill { width },
            NotShadowCaster,
        ))
        .id();
    let bar = commands
        .spawn((Transform::default(), Visibility::default(), HealthBar { owner }))
        .id();
    commands.entity(bar).add_child(bg);
    commands.entity(bar).add_child(fill);
}

// ───────────────────────────── Systems ─────────────────────────────

fn quit_on_escape(keys: Res<ButtonInput<KeyCode>>, mut exit: EventWriter<AppExit>) {
    if keys.just_pressed(KeyCode::Escape) {
        exit.write(AppExit::Success);
    }
}

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
    enemies: Query<(Entity, &Transform), With<Enemy>>,
    mut marker: Query<(&mut MoveMarker, &mut Transform), Without<Enemy>>,
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

    // Clicking on (or near) an enemy = attack order, otherwise move order.
    let mut best: Option<(Entity, f32, Vec3)> = None;
    for (e, t) in &enemies {
        let d = Vec2::new(t.translation.x - p.x, t.translation.z - p.z).length();
        if d < 1.6 && best.is_none_or(|(_, bd, _)| d < bd) {
            best = Some((e, d, t.translation));
        }
    }
    let spot = match best {
        Some((e, _, pos)) => {
            hero.attack = Some(e);
            pos
        }
        None => {
            hero.attack = None;
            hero.target = Some(p);
            p
        }
    };
    if just {
        if let Ok((mut m, mut tf)) = marker.single_mut() {
            m.t = MARKER_TIME;
            tf.translation = spot + Vec3::Y * 0.2;
        }
    }
}

fn wrap_angle(a: f32) -> f32 {
    (a + PI).rem_euclid(2.0 * PI) - PI
}

fn spawn_bullet(commands: &mut Commands, fx: &Fx, pos: Vec3, dir: Vec3, speed: f32, damage: f32, from_player: bool) {
    let mat = if from_player { &fx.player_bullet } else { &fx.enemy_bullet };
    commands.spawn((
        Mesh3d(fx.bullet.clone()),
        MeshMaterial3d(mat.clone()),
        Transform::from_translation(pos).looking_to(dir, Vec3::Y),
        Projectile {
            vel: dir * speed,
            life: 2.0,
            damage,
            from_player,
        },
        NotShadowCaster,
    ));
}

fn spawn_flash(commands: &mut Commands, fx: &Fx, mat: &Handle<StandardMaterial>, pos: Vec3, size: f32, life: f32) {
    commands.spawn((
        Mesh3d(fx.spark.clone()),
        MeshMaterial3d(mat.clone()),
        Transform::from_translation(pos).with_scale(Vec3::splat(size)),
        Flash { life, max: life, size },
        NotShadowCaster,
    ));
}

/// Decides what the hero shoots at. Idle = auto-fire at the nearest enemy in range;
/// an attack order chases the target until in range. Moving by command = no shooting.
fn hero_combat(
    time: Res<Time>,
    mut commands: Commands,
    fx: Res<Fx>,
    mut hero: Query<(&Transform, &mut Hero)>,
    enemies: Query<(Entity, &Transform), (With<Enemy>, Without<Hero>)>,
) {
    let Ok((tf, mut h)) = hero.single_mut() else {
        return;
    };
    h.cooldown = (h.cooldown - time.delta_secs()).max(0.0);
    h.aim = None;
    if let Some(e) = h.attack {
        if enemies.get(e).is_err() {
            h.attack = None;
        }
    }
    let pos = tf.translation;
    let flat = |a: Vec3| Vec2::new(a.x - pos.x, a.z - pos.z).length();

    let mut shoot_at = None;
    if let Some(e) = h.attack {
        let et = enemies.get(e).unwrap().1.translation;
        if flat(et) > HERO_RANGE * 0.9 {
            h.target = Some(et);
        } else {
            h.target = None;
            shoot_at = Some(et);
        }
    } else if h.target.is_none() {
        let mut best = HERO_RANGE;
        for (_, t) in &enemies {
            let d = flat(t.translation);
            if d < best {
                best = d;
                shoot_at = Some(t.translation);
            }
        }
    }

    let Some(at) = shoot_at else {
        return;
    };
    h.aim = Some(at);
    let desired = (at.x - pos.x).atan2(at.z - pos.z);
    if wrap_angle(desired - h.yaw).abs() < 0.2 && h.cooldown <= 0.0 {
        let muzzle = pos + Quat::from_rotation_y(h.yaw) * Vec3::new(-0.03, 1.31, 1.4);
        let dir = (at + Vec3::Y * 1.0 - muzzle).normalize();
        spawn_bullet(&mut commands, &fx, muzzle, dir, 45.0, HERO_DAMAGE, true);
        spawn_flash(&mut commands, &fx, &fx.muzzle, muzzle, 0.3, 0.06);
        h.cooldown = HERO_FIRE_INTERVAL;
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
            let mut diff = wrap_angle(desired - hero.yaw);
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
    } else if let Some(aim) = hero.aim {
        let desired = (aim.x - tf.translation.x).atan2(aim.z - tf.translation.z);
        let diff = wrap_angle(desired - hero.yaw);
        hero.yaw += diff.clamp(-12.0 * dt, 12.0 * dt);
    }

    tf.translation.x = tf.translation.x.clamp(-HALF + 5.0, HALF - 5.0);
    tf.translation.z = tf.translation.z.clamp(-HALF + 5.0, HALF - 5.0);
    tf.translation.y = height(tf.translation.x, tf.translation.z);
    tf.rotation = Quat::from_rotation_y(hero.yaw);

    anim.time += dt;
    let goal = if moving { 1.0 } else { 0.0 };
    anim.blend += (goal - anim.blend) * (1.0 - (-10.0 * dt).exp());
    if moving {
        anim.phase += hero.speed * 2.4 * dt;
    }
}

fn enemy_spawner(
    time: Res<Time>,
    mut spawner: ResMut<Spawner>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    enemies: Query<(), With<Enemy>>,
) {
    spawner.timer -= time.delta_secs();
    if spawner.timer > 0.0 || enemies.iter().count() >= MAX_ENEMIES {
        return;
    }
    spawner.timer = SPAWN_INTERVAL;
    let n = spawner.n as i32;
    spawner.n += 1;

    let e = HALF - 4.0;
    let u = (hash(n, 1, 777) * 2.0 - 1.0) * e;
    let (x, z) = match (hash(n, 2, 777) * 4.0) as i32 {
        0 => (e, u),
        1 => (-e, u),
        2 => (u, e),
        _ => (u, -e),
    };
    let pos = Vec3::new(x, height(x, z), z);
    let enemy = spawn_enemy(&mut commands, &mut meshes, &mut materials, pos, (-x).atan2(-z));
    spawn_health_bar(&mut commands, &mut meshes, &mut materials, enemy, 1.2, Color::srgb(0.95, 0.2, 0.15));
}

fn enemy_ai(
    time: Res<Time>,
    mut commands: Commands,
    fx: Res<Fx>,
    mut shots: Local<u32>,
    hero: Query<&Transform, (With<Hero>, Without<Enemy>)>,
    mut enemies: Query<(&mut Transform, &mut Enemy)>,
) {
    let Ok(hero_tf) = hero.single() else {
        return;
    };
    let dt = time.delta_secs();
    let target = hero_tf.translation;
    let others: Vec<Vec3> = enemies.iter().map(|(t, _)| t.translation).collect();

    for (mut tf, mut e) in &mut enemies {
        let to = Vec3::new(target.x - tf.translation.x, 0.0, target.z - tf.translation.z);
        let dist = to.length();
        let dir = to / dist.max(0.001);

        let desired = dir.x.atan2(dir.z);
        let diff = wrap_angle(desired - e.yaw);
        e.yaw += diff.clamp(-5.0 * dt, 5.0 * dt);

        let moving = dist > ENEMY_STOP_DIST;
        if moving {
            tf.translation += dir * ENEMY_SPEED * dt;
            e.phase += ENEMY_SPEED * 2.8 * dt;
        }
        // Don't stack on top of each other.
        for o in &others {
            let away = Vec3::new(tf.translation.x - o.x, 0.0, tf.translation.z - o.z);
            let d = away.length();
            if d > 0.01 && d < 1.3 {
                tf.translation += away / d * (1.3 - d) * 2.0 * dt;
            }
        }
        e.blend += ((if moving { 1.0 } else { 0.0 }) - e.blend) * (1.0 - (-8.0 * dt).exp());

        tf.translation.y = height(tf.translation.x, tf.translation.z);
        tf.rotation = Quat::from_rotation_y(e.yaw);

        e.cooldown -= dt;
        if e.cooldown <= 0.0 && dist < ENEMY_RANGE && diff.abs() < 0.3 {
            *shots += 1;
            let k = *shots as i32;
            let muzzle = tf.translation + Quat::from_rotation_y(e.yaw) * Vec3::new(0.05, 1.25, 0.95);
            let aim = target
                + Vec3::new(
                    (hash(k, 1, 31) - 0.5) * 1.0,
                    1.0 + (hash(k, 2, 31) - 0.5) * 0.5,
                    (hash(k, 3, 31) - 0.5) * 1.0,
                );
            let d = (aim - muzzle).normalize();
            spawn_bullet(&mut commands, &fx, muzzle, d, 26.0, ENEMY_DAMAGE, false);
            spawn_flash(&mut commands, &fx, &fx.muzzle, muzzle, 0.25, 0.06);
            e.cooldown = ENEMY_FIRE_INTERVAL * (0.8 + hash(k, 4, 31) * 0.5);
        }
    }
}

fn seg_dist(a: Vec3, b: Vec3, p: Vec3) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-6)).clamp(0.0, 1.0);
    (a + ab * t - p).length()
}

fn projectiles(
    time: Res<Time>,
    mut commands: Commands,
    fx: Res<Fx>,
    mut bullets: Query<(Entity, &mut Transform, &mut Projectile), (Without<Enemy>, Without<Hero>)>,
    mut enemies: Query<(&Transform, &mut Health), (With<Enemy>, Without<Projectile>)>,
    mut hero: Query<(&Transform, &mut Health), (With<Hero>, Without<Enemy>, Without<Projectile>)>,
) {
    let dt = time.delta_secs();
    for (entity, mut tf, mut b) in &mut bullets {
        let prev = tf.translation;
        let next = prev + b.vel * dt;
        b.life -= dt;

        let mut hit = false;
        if b.from_player {
            for (et, mut hp) in &mut enemies {
                if seg_dist(prev, next, et.translation + Vec3::Y * 1.0) < 0.55 {
                    hp.cur -= b.damage;
                    hit = true;
                    break;
                }
            }
        } else if let Ok((ht, mut hp)) = hero.single_mut() {
            if seg_dist(prev, next, ht.translation + Vec3::Y * 1.0) < 0.5 {
                hp.cur -= b.damage;
                hit = true;
            }
        }

        let ground = next.y < height(next.x, next.z);
        if hit || ground {
            spawn_flash(&mut commands, &fx, &fx.hit, next, 0.35, 0.15);
            commands.entity(entity).despawn();
        } else if b.life <= 0.0 {
            commands.entity(entity).despawn();
        } else {
            tf.translation = next;
        }
    }
}

fn update_flashes(time: Res<Time>, mut commands: Commands, mut flashes: Query<(Entity, &mut Transform, &mut Flash)>) {
    for (e, mut tf, mut f) in &mut flashes {
        f.life -= time.delta_secs();
        if f.life <= 0.0 {
            commands.entity(e).despawn();
        } else {
            tf.scale = Vec3::splat(f.size * (f.life / f.max));
        }
    }
}

/// Removes dead enemies; if the hero dies the arena resets.
fn reap(
    mut commands: Commands,
    fx: Res<Fx>,
    enemies: Query<(Entity, &Transform, &Health), With<Enemy>>,
    mut hero: Query<(&mut Transform, &mut Health, &mut Hero), Without<Enemy>>,
) {
    let hero_dead = hero.single().is_ok_and(|(_, hp, _)| hp.cur <= 0.0);
    for (e, tf, hp) in &enemies {
        if hp.cur <= 0.0 || hero_dead {
            spawn_flash(&mut commands, &fx, &fx.boom, tf.translation + Vec3::Y * 1.0, 1.4, 0.35);
            commands.entity(e).despawn();
        }
    }
    if hero_dead {
        if let Ok((mut tf, mut hp, mut h)) = hero.single_mut() {
            hp.cur = hp.max;
            tf.translation = Vec3::new(0.0, height(0.0, 0.0), 0.0);
            h.target = None;
            h.attack = None;
        }
    }
}

fn animate_enemies(enemies: Query<&Enemy>, mut limbs: Query<(&ELimb, &mut Transform)>) {
    for (l, mut tf) in &mut limbs {
        let Ok(e) = enemies.get(l.owner) else {
            continue;
        };
        let (p, b) = (e.phase, e.blend);
        let rx = Quat::from_rotation_x;
        match l.kind {
            EKind::Body => {
                tf.translation.y = 0.95 + (2.0 * p).cos() * 0.02 * b;
                tf.rotation = rx(0.2 + 0.08 * b) * Quat::from_rotation_y(p.sin() * 0.1 * b);
            }
            EKind::Head => tf.rotation = rx(-0.25 - 0.05 * b) * Quat::from_rotation_y(-p.sin() * 0.1 * b),
            EKind::HipL => tf.rotation = rx(-p.sin() * 0.6 * b + 0.1),
            EKind::HipR => tf.rotation = rx(p.sin() * 0.6 * b + 0.1),
            EKind::KneeL => tf.rotation = rx((0.15 + 0.7 * p.cos().max(0.0)) * b + 0.1),
            EKind::KneeR => tf.rotation = rx((0.15 + 0.7 * (-p.cos()).max(0.0)) * b + 0.1),
        }
    }
}

/// Health bars float just below their owner, always facing the camera.
fn update_health_bars(
    mut commands: Commands,
    camera: Query<&Transform, (With<RtsCamera>, Without<HealthBar>, Without<HealthFill>)>,
    owners: Query<(&Transform, &Health), (Without<HealthBar>, Without<HealthFill>)>,
    mut bars: Query<(Entity, &HealthBar, &mut Transform, &Children)>,
    mut fills: Query<(&mut Transform, &HealthFill), Without<HealthBar>>,
) {
    let Ok(cam) = camera.single() else {
        return;
    };
    let mut toward_cam = -(cam.rotation * Vec3::NEG_Z);
    toward_cam.y = 0.0;
    let toward_cam = toward_cam.normalize_or_zero();

    for (entity, bar, mut tf, children) in &mut bars {
        let Ok((owner_tf, hp)) = owners.get(bar.owner) else {
            commands.entity(entity).despawn();
            continue;
        };
        tf.translation = owner_tf.translation + toward_cam * 0.9 + Vec3::Y * 0.4;
        tf.rotation = cam.rotation;
        let frac = (hp.cur / hp.max).clamp(0.0, 1.0);
        for child in children.iter() {
            if let Ok((mut ftf, fill)) = fills.get_mut(child) {
                ftf.scale.x = frac.max(0.001);
                ftf.translation.x = -fill.width * 0.5 * (1.0 - frac);
            }
        }
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
                tf.rotation = rx(0.07 * b) * ry(p.sin() * 0.05 * b);
            }
            Limb::Chest => {
                tf.rotation = ry(-p.sin() * 0.08 * b) * rx(breathe * 0.012);
            }
            Limb::Head => {
                tf.rotation = rx(-0.08 * b) * ry(p.sin() * 0.1 * b);
            }
            Limb::HipL => tf.rotation = rx(-p.sin() * 0.75 * b),
            Limb::HipR => tf.rotation = rx(p.sin() * 0.75 * b),
            Limb::KneeL => tf.rotation = rx((0.05 + 1.0 * p.cos().max(0.0)) * b),
            Limb::KneeR => tf.rotation = rx((0.05 + 1.0 * (-p.cos()).max(0.0)) * b),
            Limb::ShoulderL => {
                tf.rotation = rz(-0.9) * rx(-0.8 + p.sin() * 0.05 * b + breathe * 0.015);
            }
            Limb::ShoulderR => {
                tf.rotation = rz(0.9) * rx(-0.6 - p.sin() * 0.05 * b + breathe * 0.015);
            }
            Limb::ElbowL => tf.rotation = rx(-0.5),
            Limb::ElbowR => tf.rotation = rx(-0.9),
            Limb::Rifle => {
                tf.rotation = rx(-0.03 + (2.0 * p).sin() * 0.025 * b + breathe * 0.01);
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
        cam.zoom_target = (cam.zoom_target * 0.88f32.powf(scroll_y)).clamp(6.0, 45.0);
    }
    cam.zoom += (cam.zoom_target - cam.zoom) * (1.0 - (-10.0 * dt).exp());

    cam.focus.x = cam.focus.x.clamp(-CAMERA_LIMIT, CAMERA_LIMIT);
    cam.focus.z = cam.focus.z.clamp(-CAMERA_LIMIT, CAMERA_LIMIT);
    cam.focus.y = height(cam.focus.x, cam.focus.z);

    // Closer = lower angle, farther = more top-down.
    let t = smooth(6.0, 45.0, cam.zoom);
    let pitch = 0.62 + (1.05 - 0.62) * t;
    let offset = rot * Quat::from_rotation_x(-pitch) * Vec3::Z * cam.zoom;
    tf.translation = cam.focus + Vec3::Y * 0.8 + offset;
    tf.look_at(cam.focus + Vec3::Y * 0.8, Vec3::Y);
}

fn update_marker(
    time: Res<Time>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    hero: Query<&Hero>,
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
    // Stay put (and pulse) until the hero arrives, then fade out.
    let walking = hero.single().is_ok_and(|h| h.target.is_some() && h.attack.is_none());
    if walking {
        m.t = MARKER_TIME;
    } else {
        m.t -= time.delta_secs();
    }
    let k = (m.t / MARKER_TIME).clamp(0.0, 1.0);
    *vis = Visibility::Inherited;
    let pulse = if walking {
        1.0 + 0.12 * (time.elapsed_secs() * 8.0).sin()
    } else {
        0.6 + 0.6 * k
    };
    tf.scale = Vec3::splat(pulse);
    if let Some(mat) = materials.get_mut(&mat.0) {
        mat.base_color = Color::srgba(0.3, 1.0, 0.4, k.max(0.0));
    }
}
