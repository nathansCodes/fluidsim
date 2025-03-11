use bevy::{
    input::mouse::{MouseScrollUnit, MouseWheel},
    prelude::*,
    window::PrimaryWindow,
};

use crate::{ui, SimState};

#[derive(Component)]
pub struct SimCamera;

// This is used for zooming into the cursor instead of the cursor location.
// The cursor's world position cannot be calculated immediately after updating the
// projection's scale because the camera only gets updated in
// `bevy::render::camera::camera_system`, not immediately after changing it.
// This is why I update a fake camera that doesn't render, and only update the real camera's scale
// and position in the next update cycle.
#[derive(Component)]
struct FakeCam;

#[derive(States, Clone, PartialEq, Eq, Hash, Debug)]
pub enum InteractionMode {
    None,
    Repel,
    Attract,
}

#[derive(Resource)]
pub struct InteractionSettings {
    pub force: f32,
    pub radius: f32,
}

impl Default for InteractionSettings {
    fn default() -> Self {
        Self {
            force: 5.0,
            radius: 3.5,
        }
    }
}

fn setup(mut cmds: Commands) {
    cmds.spawn(Camera2d).insert(SimCamera);
    cmds.spawn((Camera::default(), OrthographicProjection::default_2d()))
        .insert(FakeCam);
}

fn setup_cam_zoom(mut cam: Single<&mut OrthographicProjection, With<SimCamera>>) {
    cam.scale = 0.05;
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn cam_controller(
    q_camera: Single<
        (
            &mut OrthographicProjection,
            &Camera,
            &mut Transform,
            &GlobalTransform,
        ),
        (With<Camera2d>, With<SimCamera>),
    >,
    q_fake_camera: Single<
        (&mut OrthographicProjection, &Camera),
        (With<FakeCam>, Without<SimCamera>, Without<Camera2d>),
    >,
    q_window: Option<Single<&Window, With<PrimaryWindow>>>,
    mut wheel: EventReader<MouseWheel>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut cursor_moved: EventReader<CursorMoved>,
    mut zoom_diff: Local<Option<Vec2>>,
    kb: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut next_interaction_mode: ResMut<NextState<InteractionMode>>,
) {
    let Some(window) = q_window else {
        return;
    };

    let (mut projection, cam, mut transform, global_transform) = q_camera.into_inner();

    if mouse.pressed(MouseButton::Left) {
        if kb.pressed(KeyCode::ShiftLeft) {
            next_interaction_mode.set(InteractionMode::Attract);
            return;
        } else if kb.pressed(KeyCode::AltLeft) {
            next_interaction_mode.set(InteractionMode::Repel);
            return;
        }
    }

    next_interaction_mode.set(InteractionMode::None);

    let dt = time.delta_secs();

    let mut frame_delta = Vec2::ZERO;

    let (mut fake_projection, fake_cam) = q_fake_camera.into_inner();

    let mut log_scale = projection.scale.ln();

    if let Some(prev_cursor_pos) = *zoom_diff {
        let Ok(current_cursor_pos) = fake_cam
            .viewport_to_world_2d(global_transform, window.cursor_position().unwrap())
            else {
            return;
        };

        projection.scale = fake_projection.scale;
        frame_delta += prev_cursor_pos - current_cursor_pos;

        *zoom_diff = None;
    }

    for ev in wheel.read() {
        log_scale -= ev.y
            * dt
            * match ev.unit {
                MouseScrollUnit::Line => 10.0,
                MouseScrollUnit::Pixel => 7.0,
            };
        fake_projection.scale = log_scale.exp();
        *zoom_diff = Some(
            cam.viewport_to_world_2d(global_transform, window.cursor_position().unwrap())
                .unwrap(),
        );
    }

    if mouse.pressed(MouseButton::Left) {
        for ev in cursor_moved.read() {
            if let Some(delta) = ev.delta {
                frame_delta += Vec2::new(-delta.x, delta.y) * log_scale.exp();
            }
        }
    }

    let dt = time.delta_secs();

    let cam_speed: f32 = 300.0 * projection.scale;
    let dist = cam_speed * dt;

    if kb.pressed(KeyCode::KeyW) {
        frame_delta.y += dist;
    }
    if kb.pressed(KeyCode::KeyA) {
        frame_delta.x -= dist;
    }
    if kb.pressed(KeyCode::KeyS) {
        frame_delta.y -= dist;
    }
    if kb.pressed(KeyCode::KeyD) {
        frame_delta.x += dist;
    }

    transform.translation += frame_delta.extend(0.0);
}

fn keyboard_controls(
    kb: Res<ButtonInput<KeyCode>>,
    state: Res<State<SimState>>,
    mut next_state: ResMut<NextState<SimState>>,
) {
    if kb.just_pressed(KeyCode::Space) {
        next_state.set(match state.get() {
            SimState::Running | SimState::Step => SimState::Paused,
            SimState::Paused => SimState::Running,
            SimState::Prepare => SimState::Prepare,
        });
    }
    if kb.just_pressed(KeyCode::ArrowRight) {
        next_state.set(match state.get() {
            SimState::Running | SimState::Step | SimState::Paused => SimState::Step,
            SimState::Prepare => SimState::Prepare,
        });
    }
}

#[derive(SystemSet, Clone, PartialEq, Eq, Debug, Hash)]
pub struct ControlSystemSet;

pub struct ControlsPlugin;

impl Plugin for ControlsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ClearColor(Color::BLACK))
            .init_resource::<InteractionSettings>()
            .insert_state(InteractionMode::None)
            .add_systems(Startup, (setup, setup_cam_zoom).chain())
            .add_systems(
                PostUpdate,
                (
                    cam_controller.run_if(not(ui::ui_is_active)),
                    keyboard_controls,
                )
                    .after(bevy::render::camera::CameraUpdateSystem)
                    .after(TransformSystem::TransformPropagate),
            );
    }
}
