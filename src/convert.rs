#![cfg(target_os = "android")]

use glam::{Quat, Vec3};

use space_soup::renderer::{
    Beam, Color3, Cuboid, CuboidShape as SsCuboidShape, CuboidStyle as SsCuboidStyle, Light,
    LightKind as SsLightKind,
};
use space_soup_protocol::{
    WireColor3, WireCuboidShape, WireCuboidStyle, WireLightKind, WireRenderCuboid,
    WireRenderLaser, WireRenderLight,
};

pub(crate) fn xr_vec3(p: openxr::Vector3f) -> Vec3 {
    Vec3::new(p.x, p.y, p.z)
}

pub(crate) fn xr_quat(o: openxr::Quaternionf) -> Quat {
    Quat::from_xyzw(o.x, o.y, o.z, o.w)
}

pub(crate) fn to_space_soup_cuboid(rc: &WireRenderCuboid, offset: Vec3, yaw_inv: Quat) -> Cuboid {
    let style = match rc.style {
        WireCuboidStyle::Solid => SsCuboidStyle::Solid,
        WireCuboidStyle::Wireframe => SsCuboidStyle::Wireframe,
        WireCuboidStyle::SolidAndWire => SsCuboidStyle::SolidAndWire,
    };

    let position = yaw_inv * (Vec3::from(rc.position) - offset);
    let half_size = Vec3::from(rc.half_size);
    let mut c = match style {
        SsCuboidStyle::Solid => Cuboid::solid(position, half_size, ss_color(rc.color)),
        SsCuboidStyle::Wireframe => {
            Cuboid::wireframe(position, half_size, ss_color(rc.wire_color))
        }
        SsCuboidStyle::SolidAndWire => Cuboid::solid_and_wire(
            position,
            half_size,
            ss_color(rc.color),
            ss_color(rc.wire_color),
        ),
    };
    c.rotation = yaw_inv * Quat::from_array(rc.rotation);
    c.lightmap_key = Some(rc.id.clone());
    c.reflectivity = rc.reflectivity.clamp(0.0, 1.0);
    c.shape = match rc.shape {
        WireCuboidShape::Box => SsCuboidShape::Box,
        WireCuboidShape::Cylinder => SsCuboidShape::Cylinder,
    };
    c
}

pub(crate) fn to_space_soup_light(rl: &WireRenderLight, offset: Vec3, yaw_inv: Quat) -> Light {
    Light {
        position: yaw_inv * (Vec3::from(rl.position) - offset),
        direction: yaw_inv * Vec3::from(rl.direction),
        kind: match rl.kind {
            WireLightKind::Point => SsLightKind::Point,
            WireLightKind::Spot => SsLightKind::Spot,
        },
        color: ss_color(rl.color),
        intensity: rl.intensity,
        range: rl.range,
        cone_angle_deg: rl.cone_angle_deg,
    }
}

pub(crate) fn to_space_soup_beam(rl: &WireRenderLaser, offset: Vec3, yaw_inv: Quat) -> Beam {
    Beam {
        start: yaw_inv * (Vec3::from(rl.origin) - offset),
        end: yaw_inv * (Vec3::from(rl.end) - offset),
        width: rl.beam_width,
        color: ss_color(rl.color),
    }
}

pub(crate) fn ss_color(c: WireColor3) -> Color3 {
    Color3(c.0, c.1, c.2, c.3)
}
