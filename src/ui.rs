use std::ops::BitXorAssign;

use bevy::{
    color,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
    window::PrimaryWindow,
};
use bevy_egui::{
    egui::{
        self, emath, panel::Side, Align2, CollapsingResponse, Response, SelectableLabel, Ui,
        WidgetText,
    },
    EguiContexts, EguiPlugin,
};

use crate::{
    controls::{InteractionSettings, SimCamera},
    debug::{DebugData, DebugInfo, LogLevel, ParticleColoring},
    sim::{Device, Sim, SimEvents, SimState, SpawnInfo},
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

fn bitflags_combobox<F: bitflags::Flags + BitXorAssign + Clone + Copy>(
    ui: &mut Ui,
    label: impl Into<WidgetText>,
    value: &mut F,
) {
    let current_name = if value.is_empty() {
        "None".to_string()
    } else if value.is_all() {
        "All".to_string()
    } else {
        F::FLAGS
            .iter()
            .filter_map(|f| {
                if f.value().intersects(*value) {
                    Some(f.name())
                } else {
                    None
                }
            })
            .fold(None, |acc: Option<String>, n| {
                if let Some(acc) = acc {
                    Some(acc + ", " + n)
                } else {
                    Some(n.to_string())
                }
            })
            .unwrap()
    };

    egui::ComboBox::from_label(label)
        .selected_text(current_name)
        .show_ui(ui, |ui| {
            let label = SelectableLabel::new(value.is_empty(), "None");
            if ui.add(label).clicked() {
                *value = F::empty();
            }
            for flag in F::FLAGS {
                if ui
                    .add(SelectableLabel::new(
                        value.intersects(*flag.value()),
                        flag.name(),
                    ))
                    .clicked()
                {
                    *value ^= *flag.value();
                }
            }
        });
}

fn collapsing_open<R>(
    ui: &mut Ui,
    label: impl Into<WidgetText>,
    add_body: impl FnOnce(&mut Ui) -> R,
) -> CollapsingResponse<R> {
    egui::CollapsingHeader::new(label)
        .default_open(true)
        .show(ui, add_body)
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
    device: Res<State<Device>>,
    mut next_device: ResMut<NextState<Device>>,
    diagnostics: Res<DiagnosticsStore>,
) {
    let ctx = contexts.ctx_mut();

    egui::SidePanel::new(Side::Left, "Hello")
        .resizable(true)
        .min_width(300.0)
        .show(ctx, |ui| {
            collapsing_open(ui, "Simulation Settings", |ui| {
                egui::ComboBox::from_label("Gravity Mode")
                    .selected_text(if sim.planetary_gravity {
                        "Planetary"
                    } else {
                        "Uniform"
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut sim.planetary_gravity, false, "Uniform");
                        ui.selectable_value(&mut sim.planetary_gravity, true, "Planetary");
                    });

                if sim.planetary_gravity {
                    labeled_drag_value(ui, "Gravity", &mut sim.gravity.y, 10.0);
                    labeled_vec2(ui, "Position", &mut sim.planet_position);
                    labeled_drag_value(ui, "Radius", &mut sim.planet_radius, 0.1);
                } else {
                    labeled_vec2(ui, "Gravity", &mut sim.gravity);
                }

                labeled_drag_value(ui, "Particle Size", &mut sim.particle_diameter, 0.005);
                labeled_drag_value(ui, "Smoothing Radius", &mut sim.smoothing_radius, 0.025);
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
            });

            collapsing_open(ui, "Debug", |ui| {
                bitflags_combobox(ui, "Debug Overlay", &mut debug_data.debug_overlay);

                egui::ComboBox::from_label("Particle Color Mode")
                    .selected_text(format!("{:?}", debug_data.particle_colors))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut debug_data.particle_colors,
                            ParticleColoring::Density,
                            "Density",
                        );
                        ui.selectable_value(
                            &mut debug_data.particle_colors,
                            ParticleColoring::Velocity,
                            "Velocity",
                        );
                    });

                egui::ComboBox::from_label("Log Level")
                    .selected_text(format!("{:?}", debug_data.log_level))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut debug_data.log_level,
                            LogLevel::IllegalValues,
                            "Illegal Values",
                        );
                        ui.selectable_value(&mut debug_data.log_level, LogLevel::Always, "Always");
                    });
            });

            collapsing_open(ui, "Mouse Settings", |ui| {
                labeled_drag_value(ui, "Radius", &mut mouse_settings.radius, 0.05);
                labeled_drag_value(ui, "Force Multiplier", &mut mouse_settings.force, 0.05);
            });

            collapsing_open(ui, "Spawn Parameters", |ui| {
                ui.add_enabled_ui(*state.get() == SimState::Prepare, |ui| {
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

                    egui::ComboBox::from_label("Device")
                        .selected_text(format!("{:?}", device.get()))
                        .show_ui(ui, |ui| {
                            if ui.button("CPU").clicked() {
                                next_device.set(Device::CPU);
                                eprintln!("set device to cpu");
                            }
                            if ui.button("GPU").clicked() {
                                next_device.set(Device::GPU);
                                eprintln!("set device to gpu");
                            }
                        })
                })
            });

            match *state.get() {
                SimState::Prepare => {
                    if ui.button("Start Simulation").clicked() {
                        evw.send(SimEvents::StartSimEvent(ui_state.spawn_info, false));
                    }
                    if ui.button("Start Paused").clicked() {
                        evw.send(SimEvents::StartSimEvent(ui_state.spawn_info, true));
                    }
                }
                _ => {
                    if ui.button("Reset Simulation").clicked() {
                        evw.send(SimEvents::ResetSim);
                    }
                }
            }

            egui::TopBottomPanel::new(egui::panel::TopBottomSide::Bottom, "stuff").show(
                ui.ctx(),
                |ui| {
                    ui.horizontal(|ui| {
                        let Some(window) = q_window else {
                            return;
                        };

                        if let Some(value) = diagnostics
                            .get(&FrameTimeDiagnosticsPlugin::FPS)
                            .and_then(|fps| fps.smoothed())
                        {
                            ui.label(value.round().to_string() + "fps");
                        }

                        if let Some(position) = window.cursor_position() {
                            let Ok(world_position) =
                                q_camera.0.viewport_to_world_2d(q_camera.1, position)
                            else {
                                return;
                            };
                            ui.label(format!(
                                "Density at mouse pointer: {}",
                                sim.density_at_point(world_position).x,
                            ));
                            ui.label(format!("{} ms", debug_data.step_execution_time));
                        }
                    })
                },
            );
        });

    ui_state.is_active = ctx.is_pointer_over_area() | ctx.dragged_id().is_some();
}

pub fn debug_info(
    In(debug_info): In<Option<DebugInfo>>,
    mut contexts: EguiContexts,
    sim: Res<Sim>,
) {
    let Some(debug_info) = debug_info else {
        return;
    };

    egui::Window::new("Debug Info")
        .anchor(Align2::RIGHT_TOP, egui::Vec2::new(-8., 8.))
        .show(contexts.ctx_mut(), |ui| {
            ui.label(format!("Neighbored particles: {}", debug_info.neighbors));
            ui.label(format!(
                "Neighbors outside of radius: {}",
                debug_info.discarded_neighbors
            ));
            ui.label(format!(
                "Particles discarded in search: {}",
                sim.positions.len() - (debug_info.neighbors + debug_info.discarded_neighbors)
            ));
            ui.label(format!("Density: {}", debug_info.density.x));
            ui.label(format!("Near density: {}", debug_info.density.y));
        });
}

fn draw_bounds(sim: Res<Sim>, mut gizmos: Gizmos) {
    gizmos.rect_2d(
        Isometry2d::IDENTITY,
        sim.bounds_size,
        color::LinearRgba::RED,
    );
    if sim.planetary_gravity {
        gizmos.circle_2d(
            Isometry2d::new(sim.planet_position, Rot2::IDENTITY),
            sim.planet_radius,
            color::LinearRgba::BLUE,
        );
    }
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
