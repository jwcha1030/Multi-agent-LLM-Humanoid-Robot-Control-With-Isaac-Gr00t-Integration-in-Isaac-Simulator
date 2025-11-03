"""
Action Replay Script - Execute recorded trajectories in Isaac Sim

This script allows you to load and replay recorded action sequences.
Useful for testing individual subtasks or composing them via LLM planning.
"""

from isaacsim import SimulationApp

# Simulation parameters
LOAD_WORLD_FILE = "./sim_environments/gr1_NutPouring.usd"
RESULT_VIDEO_FILE = "./results/action_replay.mp4"

# Actions to replay (set these or pass as arguments)
# ACTIONS_TO_REPLAY = [
#     "pick_red_beaker",  # Example action names
#     # "pour_nut",
#     # "grab_yellow_bowl",
#     # "place_on_scale"
# ]
ACTIONS_TO_REPLAY = [
    "pick_up_yellow_bowl",
    "move_item_in_right_hand_over_to_measuring_scale",
    "drop_item_in_right_hand_on_measuring_scale"
    # "grab_yellow_bowl",
    # "place_on_scale"
]

simulation_app = SimulationApp({
    "headless": False,
    "create_new_stage": False,
    "open_usd": LOAD_WORLD_FILE,
    "sync_loads": True,
})

# Setup camera
CAMERA_HEIGHT = 200
CAMERA_FOCAL_LENGTH = 1.2
CAMERA_FORWARD_DIST = 0.25
CAMERA_ANGLE = 70
    

from isaacsim.core.api import World
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.sensors.camera.camera import Camera
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.utils.types import ArticulationAction
import cv2
import gr1_config
from action_library import ActionLibrary
import time


def replay_action_sequence(action_label: str, 
                          gr1: Robot,
                          gr1_controller: ArticulationController,
                          world: World,
                          video_writer=None,
                          camera=None):
    """
    Replay a single recorded action sequence.
    
    Args:
        action_label: Name/label of the action to replay
        gr1: Robot instance
        gr1_controller: Articulation controller
        world: World instance
        video_writer: Optional video writer
        camera: Optional camera for video recording
    
    Returns:
        bool: True if successful, False otherwise
    """
    library = ActionLibrary()
    trajectory = library.get_trajectory(action_label)

    if trajectory is None:
        print(f"Action '{action_label}' not found in library")
        return False
    
    print(f"\n{'='*60}")
    print(f"REPLAYING: {action_label}")
    print(f"   Frames: {len(trajectory)}")
    print(f"{'='*60}\n")
    
    for frame_idx, joint_positions in enumerate(trajectory):
        # Apply the recorded joint positions
        gr1_controller.apply_action(ArticulationAction(joint_positions=joint_positions))
        
        # Step the simulation
        world.step(render=True)
        
        # Record video if camera and writer provided
        if camera is not None and video_writer is not None:
            obs = camera.get_rgba()
            image = obs[:, :, :3]
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr)
        
        # Print progress every 50 frames
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{len(trajectory)}")
        
        # Small delay for smoother visualization
        time.sleep(0.01)

    print(f"Finished replaying: {action_label}\n")
    return True


def main():
    print("=" * 60)
    print("ACTION REPLAY SYSTEM")
    print("=" * 60)
    
    # List available actions
    library = ActionLibrary()
    library.list_actions()
    
    if not library.actions:
        print("\n⚠️  No actions found in library!")
        print("Please record some actions first using run_simulation_with_recorder.py")
        simulation_app.close()
        return

    ## 1. Setup scene
    print("\n## 1. Setup scene")
    world = World()
    scene: Scene = world.scene
    gr1: Robot = scene.add(Robot(
        prim_path="/World/gr1",
        name="gr1",
    ))
    gr1_articulation_controller: ArticulationController = gr1.get_articulation_controller()
    
    camera = Camera(
        prim_path="/World/gr1/head_yaw_link/camera",
        name="camera",
        translation=np.array([CAMERA_FORWARD_DIST, 0.0, 0.07]),
        frequency=60,
        resolution=(256, CAMERA_HEIGHT),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, CAMERA_ANGLE, 0]), degrees=True),
    )

    camera.set_focal_length(CAMERA_FOCAL_LENGTH)
    camera.set_clipping_range(0.1, 2)
    
    ## 2. Setup post-load
    world.reset()
    print("## 2. Setup post-load")
    camera.initialize()
    camera.add_motion_vectors_to_frame()
    gr1_articulation_controller.set_gains(
        kps=np.array([3000.0] * 54),
        kds=np.array([100.0] * 54)
    )
    
    # Set initial position
    gr1.set_joint_positions(positions=gr1_config.default_joint_position)
    
    # Initialize the world
    print("Initializing world...")
    for _ in range(100):
        world.step(render=True)
    
    ## 3. Setup video recording
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(RESULT_VIDEO_FILE, fourcc, 30, (256, CAMERA_HEIGHT), isColor=True)
    if not video.isOpened():
        print("Warning: Could not open video writer")
        video = None
    
    ## 4. Replay actions
    print("\n## 3. Replaying actions")
    
    # If no actions specified, ask user
    actions_to_replay = ACTIONS_TO_REPLAY
    if not actions_to_replay or actions_to_replay == ["pick_red_beaker"]:
        print("\nEnter action labels to replay (comma-separated), or 'all' for all actions:")
        user_input = input("> ").strip()
        
        if user_input.lower() == 'all':
            actions_to_replay = list(library.actions.keys())
        else:
            actions_to_replay = [label.strip() for label in user_input.split(',') if label.strip()]
    
    # Replay each action
    for action_label in actions_to_replay:
        success = replay_action_sequence(
            action_label=action_label,
            gr1=gr1,
            gr1_controller=gr1_articulation_controller,
            world=world,
            video_writer=video,
            camera=camera
        )
        
        if not success:
            continue
        
        # Wait a bit between actions
        print("Waiting 3 seconds before next action...")
        for _ in range(90):  # 3 seconds at 30 FPS
            world.step(render=True)
            if video and camera:
                obs = camera.get_rgba()
                image = obs[:, :, :3]
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                video.write(bgr)
    
    # Cleanup
    if video:
        video.release()
        print(f"\nVideo saved to: {RESULT_VIDEO_FILE}")
    
    print("\n" + "=" * 60)
    print("ACTION REPLAY COMPLETE")
    print("=" * 60)
    
    simulation_app.close()


if __name__ == "__main__":
    main()
