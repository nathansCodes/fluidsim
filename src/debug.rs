use bevy::{color::palettes::basic::*, math::FloatPow, prelude::*, window::PrimaryWindow};

use crate::{
    controls::SimCamera,
    sim::{self, Sim},
};

#[derive(Default, Debug, PartialEq, Eq)]
pub enum ParticleColoring {
    #[default]
    Density,
    Velocity,
}

#[derive(Default, Debug, PartialEq, Eq)]
pub enum LogLevel {
    Always,
    #[default]
    IllegalValues,
}

bitflags::bitflags! {
    #[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DebugOverlay: u32 {
        const CellQuery = 0b00000001;
        const CellGrid = 0b00000010;
        const SmoothingRadius = 0b00000100;
        const VelocityArrows = 0b00001000;
    }
}

#[derive(Resource, Default)]
pub struct DebugData {
    pub step_execution_time: u128,
    pub debug_overlay: DebugOverlay,
    pub particle_colors: ParticleColoring,
    pub log_level: LogLevel,
}

#[derive(Default)]
pub struct DebugInfo {
    pub neighbors: usize,
    pub discarded_neighbors: usize,
    pub density: (f32, f32),
    pub direction: Vec2,
}

#[allow(clippy::type_complexity)]
pub fn debug_overlay(
    sim: Res<Sim>,
    debug_data: Res<DebugData>,
    q_camera: Query<(&Camera, &GlobalTransform), (With<Camera2d>, With<SimCamera>)>,
    q_window: Query<&Window, With<PrimaryWindow>>,
    mut gizmos: Gizmos,
) -> Option<DebugInfo> {
    if debug_data.debug_overlay.is_empty() {
        return None;
    }

    let Ok((cam, global_transform)) = q_camera.get_single() else {
        return None;
    };
    let Ok(window) = q_window.get_single() else {
        return None;
    };

    let mut debug_info = DebugInfo::default();

    let mouse_pos_maybe = window
        .cursor_position()
        .map(|p| cam.viewport_to_world_2d(global_transform, p).unwrap());

    if let Some(mouse_pos) = mouse_pos_maybe {
        debug_info.density = sim.density_at_point(mouse_pos);

        if debug_data.debug_overlay.intersects(DebugOverlay::CellQuery) {
            let (discarded_neighbors, particles) = sim.spatial_query_details(mouse_pos);
            debug_info.neighbors += particles.len();
            debug_info.discarded_neighbors += discarded_neighbors;

            particles
                .iter()
                .for_each(|(particle_index, _particle_cell_key)| {
                    gizmos.line_2d(mouse_pos, sim.positions[*particle_index], GREEN);
                });
        }

        if debug_data.debug_overlay.intersects(DebugOverlay::CellGrid) {
            for offset in sim::CELL_OFFSETS {
                let mouse_cell_coord = sim::pos_to_cell_coord(mouse_pos, sim.smoothing_radius);
                let offset = offset * sim.smoothing_radius;
                let cell_pos = mouse_cell_coord * sim.smoothing_radius + offset;

                gizmos.rect_2d(
                    cell_pos + Vec2::new(sim.smoothing_radius / 2.0, sim.smoothing_radius / 2.0),
                    Vec2::new(sim.smoothing_radius, sim.smoothing_radius),
                    GRAY,
                );
            }
        }
        if debug_data
            .debug_overlay
            .intersects(DebugOverlay::SmoothingRadius)
        {
            gizmos.circle_2d(mouse_pos, sim.smoothing_radius, RED);
        }
    }
    Some(debug_info)
}
