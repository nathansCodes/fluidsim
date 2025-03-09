use bevy::{color, prelude::*, window::PrimaryWindow};
use bevy_egui::{
    egui::{self, emath, panel::Side, Response, Ui, WidgetText},
    EguiContexts, EguiPlugin,
};

use crate::{
    controls::{InteractionSettings, SimCamera},
    sim::DebugData,
    Sim, SimEvents, SimState, SpawnInfo,
};

#[derive(Resource, Default)]
pub struct UiState {
    is_active: bool,
    spawn_info: SpawnInfo,
}

pub fn ui_is_active(ui_state: Res<UiState>) -> bool {
    ui_state.is_active
}

fn labeled_drag_value(
    ui: &mut Ui,
    label: impl Into<WidgetText>,
    value: &mut impl emath::Numeric,
    speed: f32,
) -> Response {
    ui.horizontal(|ui| {
        ui.add(egui::DragValue::new(value).speed(speed))
            .labelled_by(ui.label(label).id)
    })
    .response
}

fn labeled_vec2(ui: &mut Ui, label: impl Into<WidgetText>, value: &mut Vec2) -> Response {
    ui.horizontal(|ui| {
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut value.x));
            ui.add(egui::DragValue::new(&mut value.y));
        })
        .response
        .labelled_by(ui.label(label).id)
    })
    .response
}

#[allow(clippy::too_many_arguments)]
pub(super) fn ui(
    mut contexts: EguiContexts,
    mut sim: ResMut<Sim>,
    mut debug_data: ResMut<DebugData>,
    state: Res<State<SimState>>,
    mut ui_state: ResMut<UiState>,
    mut mouse_settings: ResMut<InteractionSettings>,
    mut evw: EventWriter<SimEvents>,
    q_window: Option<Single<&Window, With<PrimaryWindow>>>,
    q_camera: Single<(&Camera, &GlobalTransform), With<SimCamera>>,
) {
    let ctx = contexts.ctx_mut();

    egui::SidePanel::new(Side::Left, "Hello")
        .resizable(true)
        .min_width(300.0)
        .show(ctx, |ui| {
            ui.heading("Simulation Settings");

            labeled_vec2(ui, "Gravity", &mut sim.gravity);
            labeled_drag_value(ui, "Particle Size", &mut sim.particle_radius, 1.0);
            labeled_drag_value(ui, "Smoothing Radius", &mut sim.smoothing_radius, 1.0);
            labeled_drag_value(ui, "Pressure Multiplier", &mut sim.pressure_multiplier, 1.0);
            labeled_drag_value(
                ui,
                "Near Pressure Multiplier",
                &mut sim.near_pressure_multiplier,
                0.05,
            );
            labeled_drag_value(ui, "Viscosity", &mut sim.viscosity, 0.05);
            labeled_drag_value(ui, "Target Density", &mut sim.target_density, 0.001);
            labeled_drag_value(ui, "Time Step", &mut sim.delta, 0.001);
            labeled_vec2(ui, "Bounds size", &mut sim.bounds_size);

            ui.separator();
            ui.heading("Debug");

            egui::ComboBox::from_label("Particle Color Mode")
                .selected_text(format!("{:?}", debug_data.particle_colors))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut debug_data.particle_colors,
                        crate::sim::ParticleColoring::Density,
                        "Density",
                    );
                    ui.selectable_value(
                        &mut debug_data.particle_colors,
                        crate::sim::ParticleColoring::Velocity,
                        "Velocity",
                    );
                });
            egui::ComboBox::from_label("Log Level")
                .selected_text(format!("{:?}", debug_data.log_level))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut debug_data.log_level,
                        crate::sim::LogLevel::IllegalValues,
                        "Illegal Values",
                    );
                    ui.selectable_value(
                        &mut debug_data.log_level,
                        crate::sim::LogLevel::Always,
                        "Always",
                    );
                });

            ui.separator();
            ui.heading("Mouse Settings");

            ui.checkbox(&mut mouse_settings.custom, "Customize Settings");
            ui.add_enabled_ui(mouse_settings.custom, |ui| {
                labeled_drag_value(ui, "Radius", &mut mouse_settings.radius, 0.05);
                labeled_drag_value(ui, "Force Multiplier", &mut mouse_settings.force, 0.05);
            });

            ui.separator();

            ui.add_enabled_ui(*state.get() == SimState::Prepare, |ui| {
                ui.heading("Spawn Parameters");

                labeled_drag_value(
                    ui,
                    "Number of Particles",
                    &mut ui_state.spawn_info.num_particles,
                    1.0,
                );
                labeled_drag_value(
                    ui,
                    "Particle Spacing",
                    &mut ui_state.spawn_info.spacing,
                    1.0,
                );

                labeled_vec2(ui, "Center", &mut ui_state.spawn_info.center);

                if ui.button("Start Simulation").clicked() {
                    evw.send(SimEvents::StartSimEvent(ui_state.spawn_info));
                }
            });

            if *state.get() != SimState::Prepare && ui.button("Reset Simulation").clicked() {
                evw.send(SimEvents::ResetSim);
            }

            egui::TopBottomPanel::new(egui::panel::TopBottomSide::Bottom, "stuff").show(
                ui.ctx(),
                |ui| {
                    let Some(window) = q_window else {
                        return;
                    };
                    if let Some(position) = window.cursor_position() {
                        let Ok(world_position) =
                            q_camera.0.viewport_to_world_2d(q_camera.1, position)
                        else {
                            return;
                        };
                        ui.horizontal(|ui| {
                            ui.label(format!(
                                "Density at mouse pointer: {}",
                                sim.density_at_point(world_position).0,
                            ));
                            ui.label(format!("{} ms", debug_data.step_execution_time,));
                        });
                    }
                },
            );
        });

    ui_state.is_active = ctx.is_pointer_over_area() | ctx.dragged_id().is_some();
}

fn draw_bounds(sim: Res<Sim>, mut gizmos: Gizmos) {
    gizmos.rect_2d(
        Isometry2d::IDENTITY,
        sim.bounds_size,
        color::LinearRgba::RED,
    );
}

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(EguiPlugin)
            .insert_resource(UiState::default())
            .add_systems(Update, (ui, draw_bounds))
            .add_event::<SimEvents>();
    }
}
