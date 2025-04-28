mod controls;
mod debug;
mod sim;
mod ui;

use bevy::prelude::*;
use bevy::{app::App, DefaultPlugins};

use controls::ControlsPlugin;
use sim::SimPlugin;
use ui::UiPlugin;

fn main() {
    App::new()
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    present_mode: bevy::window::PresentMode::Immediate,
                    ..default()
                }),
                ..default()
            }),
            UiPlugin,
            ControlsPlugin,
            SimPlugin,
        ))
        .run();
}
