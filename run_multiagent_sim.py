"""
Isaac Sim Integration for Multi-Agent System

This script connects the multi-agent LLM system with Isaac Sim,
allowing AI agents to control the robot in simulation.
"""

from isaacsim import SimulationApp

# Simulation parameters
VERIFICATION_FREE_PASS = False  # Set to True to auto-approve all actions
LOAD_WORLD_FILE = "./sim_environments/gr1_NutPouring.usd"
RESULT_VIDEO_FILE = "./results/multiagent_execution.mp4"
GR00T_TASK = "Pick up the red beaker and tilt it to pour out 1 green nut into yellow bowl. Pick up the yellow bowl and place it on the metallic measuring scale."
INFERENCE_SERVER_URL = "http://localhost:9876/inference"
GPT_MODEL_NAME = "gpt-5-mini"  # Vision-capable model for verification
# IMPORTANT: Configure before importing other Isaac Sim modules
simulation_app = SimulationApp({
    "headless": False,
    "create_new_stage": False,
    "open_usd": LOAD_WORLD_FILE,
    "sync_loads": True,
})

from isaacsim.core.api import World
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.sensors.camera.camera import Camera
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.utils.types import ArticulationAction
import gr1_config
import gr1_gr00t_utils
from action_library import ActionLibrary
from multiagent_system import MultiAgentRobotSystem
import time
import cv2
import os


class VideoWriter:
    """
    Video writer for capturing simulation frames.
    """
    
    def __init__(self, output_path: str, fps: int = 30, resolution: tuple = (256, 200)):
        """
        Initialize video writer.
        
        Args:
            output_path: Path to save the video file
            fps: Frames per second
            resolution: Video resolution (width, height)
        """
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.frames = []
        self.writer = None
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    def add_frame(self, image: np.ndarray):
        """
        Add a frame to the video.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        """
        # Convert RGB to BGR for OpenCV
        if image.shape[2] == 3:
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            frame = image
        
        # Ensure uint8 format
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
        
        self.frames.append(frame)
    
    def save(self):
        """Save all frames to video file."""
        if not self.frames:
            print("No frames to save!")
            return
        
        print(f"\nSaving video with {len(self.frames)} frames...")
        
        # Get frame dimensions
        height, width = self.frames[0].shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (width, height)
        )
        
        # Write all frames
        for frame in self.frames:
            self.writer.write(frame)
        
        # Release writer
        self.writer.release()
        print(f"Video saved to: {self.output_path}")
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.writer is not None:
            self.writer.release()


class SimulatorBridge:
    """
    Bridges the multi-agent system with Isaac Sim.
    Handles action execution and state feedback.
    """
    
    def __init__(self, 
                 world: World,
                 robot: Robot,
                 controller: ArticulationController,
                 camera: Camera,
                 video_writer: VideoWriter = None):
        self.world = world
        self.robot = robot
        self.controller = controller
        self.camera = camera
        self.library = ActionLibrary()
        self.video_writer = video_writer

        # For GR00T inference
        self.inference_url = INFERENCE_SERVER_URL
        self.groot_task = GR00T_TASK

    def callback(self, command: str, data: any) -> dict:
        """
        Callback function for multi-agent system.
        
        Args:
            command: Either "use_groot" or "execute_action"
            data: Action label (for execute_action)
            
        Returns:
            Dictionary with execution result
        """
        if command == "use_groot":
            return self._execute_groot_inference()
        elif command == "execute_action":
            return self._execute_action_from_library(data)
        else:
            return {"success": False, "error": "Unknown command"}
    
    def _capture_image(self, record_frame: bool = True) -> np.ndarray:
        """
        Capture current camera image.
        
        Args:
            record_frame: If True and video_writer is available, add frame to video
        
        Returns:
            RGB image as numpy array
        """
        obs = self.camera.get_rgba()
        image = obs[:, :, :3]  # RGB only
        
        # Record frame to video if enabled
        if record_frame and self.video_writer is not None:
            self.video_writer.add_frame(image)
        
        return image
    
    def _execute_groot_inference(self) -> dict:
        """Execute the complete trained task using GR00T inference."""
        print(f"\n{'='*60}")
        print("EXECUTING TRAINED TASK WITH GR00T")
        print(f"{'='*60}\n")
        
        try:
            # Capture before image
            image_before = self._capture_image()
            
            # Run inference for multiple steps
            steps = 40  # Adjust based on your trained task length
            
            for step in range(steps):
                self.world.step(render=True)
                
                # Capture frame for video
                if self.video_writer is not None:
                    frame = self._capture_image(record_frame=True)
                
                # Get current state
                obs = self._capture_image()
                current_joint_positions = self.robot.get_joint_positions()
                
                # Request GR00T inference
                print(f"Step {step+1}/{steps}: Calling GR00T inference")
                groot_input = gr1_gr00t_utils.make_gr00t_input(
                    task=self.groot_task,
                    obs=obs,
                    joint_positions=current_joint_positions
                )
                
                groot_output = gr1_gr00t_utils.request_gr00t_inference(
                    payload=groot_input,
                    url=self.inference_url
                )
                
                # Execute 16 timesteps of actions
                for timestep in range(16):
                    action_joint_position = gr1_gr00t_utils.make_joint_position_from_gr00t_output(
                        groot_output,
                        timestep=timestep
                    )
                    
                    self.controller.apply_action(
                        ArticulationAction(joint_positions=action_joint_position)
                    )

                    if timestep < 15:
                        self.world.step(render=True)
                        # Capture frame for video
                        if self.video_writer is not None:
                            self._capture_image(record_frame=True)
            
            # Capture after image
            image_after = self._capture_image()
            
            print("GR00T execution complete\n")
            
            return {
                "success": True,
                "image_before": image_before,
                "image_after": image_after
            }
            
        except Exception as e:
            print(f"GR00T execution failed: {e}\n")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _execute_action_from_library(self, action_label: str) -> dict:
        """Execute a recorded action from the library."""
        print(f"\n{'='*60}")
        print(f" EXECUTING ACTION: {action_label}")
        print(f"{'='*60}\n")
        
        try:
            # Clean up any previous verification images from old runs
            import glob
            import os
            old_images = glob.glob("./results/verification_images/*.jpg")
            if old_images:
                print(f"Cleaning up {len(old_images)} old verification images from previous runs...")
                for old_image in old_images:
                    try:
                        os.remove(old_image)
                    except:
                        pass
            
            # Capture before image
            image_before = self._capture_image()
            
            # Get trajectory
            trajectory = self.library.get_trajectory(action_label)
            
            if trajectory is None:
                return {
                    "success": False,
                    "error": f"Action '{action_label}' not found"
                }
            
            # Execute trajectory
            print(f"Executing {len(trajectory)} frames...")
            
            for frame_idx, joint_positions in enumerate(trajectory):
                self.controller.apply_action(
                    ArticulationAction(joint_positions=joint_positions)
                )
                self.world.step(render=True)
                
                # Capture frame for video
                if self.video_writer is not None:
                    self._capture_image(record_frame=True)
                
                # Small delay for visualization
                time.sleep(0.01)
                
                if frame_idx % 50 == 0:
                    print(f"  Frame {frame_idx}/{len(trajectory)}")
            
            # Wait a moment after action
            for _ in range(30):  # 1 second
                self.world.step(render=True)
                # Capture frame for video
                if self.video_writer is not None:
                    self._capture_image(record_frame=True)
            
            # Capture after image
            image_after = self._capture_image()
            
            print(f"Action complete\n")
            
            return {
                "success": True,
                "image_before": image_before,
                "image_after": image_after
            }
            
        except Exception as e:
            print(f"Action execution failed: {e}\n")
            return {
                "success": False,
                "error": str(e)
            }


def main():
    """Main execution loop."""
    
    print("="*80)
    print("MULTI-AGENT ROBOT CONTROL WITH ISAAC SIM")
    print("="*80)
    
    # Camera settings
    CAMERA_HEIGHT = 200
    CAMERA_FOCAL_LENGTH = 1.2
    CAMERA_FORWARD_DIST = 0.25
    CAMERA_ANGLE = 70
    
    ## 1. Setup scene
    print("\n## 1. Setting up Isaac Sim scene...")
    world = World()
    scene: Scene = world.scene
    robot: Robot = scene.add(Robot(
        prim_path="/World/gr1",
        name="gr1",
    ))
    controller: ArticulationController = robot.get_articulation_controller()
    
    # Setup camera
    camera = Camera(
        prim_path="/World/gr1/head_yaw_link/camera",
        name="camera",
        translation=np.array([CAMERA_FORWARD_DIST, 0.0, 0.07]),
        frequency=60,
        resolution=(256, CAMERA_HEIGHT),
        orientation=rot_utils.euler_angles_to_quats(
            np.array([0, CAMERA_ANGLE, 0]), degrees=True
        ),
    )
    camera.set_focal_length(CAMERA_FOCAL_LENGTH)
    camera.set_clipping_range(0.1, 2)
    
    ## 2. Initialize
    print("## 2. Initializing world...")
    world.reset()
    camera.initialize()
    camera.add_motion_vectors_to_frame()
    controller.set_gains(
        kps=np.array([3000.0] * 54),
        kds=np.array([100.0] * 54)
    )
    
    # Set initial position
    robot.set_joint_positions(positions=gr1_config.default_joint_position)
    
    # Warm up simulation
    for _ in range(100):
        world.step(render=True)
    
    print("Isaac Sim ready\n")
    
    ## 3. Setup multi-agent system
    print("## 3. Initializing multi-agent system...")
    agent_system = MultiAgentRobotSystem(
        model_name=GPT_MODEL_NAME,
        library_dir="./action_library",
        supervisor_free_pass=VERIFICATION_FREE_PASS
    )
    
    # Create video writer
    print("## 3.5. Setting up video recording...")
    video_writer = VideoWriter(
        output_path=RESULT_VIDEO_FILE,
        fps=30,
        resolution=(256, CAMERA_HEIGHT)
    )
    print(f"Video will be saved to: {RESULT_VIDEO_FILE}\n")
    
    # Create simulator bridge
    sim_bridge = SimulatorBridge(world, robot, controller, camera, video_writer)
    
    print("Multi-agent system ready\n")
    
    ## 4. Get task from user
    print("="*80)
    print("TASK INPUT")
    print("="*80)
    print("\nAvailable example tasks:")
    print("1. Pick up the red beaker and tilt it to pour out 1 green nut into yellow bowl. Pick up the yellow bowl and place it on the metallic measuring scale.")
    print("2. Pick up the yellow bowl and place it on the measuring scale")
    print("3. Grab the red beaker and move it to the blue box")
    print("4. Custom task\n")
    
    choice = input("Select (1-4) or press Enter for task 2: ").strip()
    
    if choice == "1":
        user_task = "Pick up the red beaker and tilt it to pour out 1 green nut into yellow bowl. Pick up the yellow bowl and place it on the metallic measuring scale."
    elif choice == "3":
        user_task = "Grab the red beaker and move it to the blue box"
    elif choice == "4":
        user_task = input("\nEnter your task: ").strip()
    else:
        print("\nDefaulting to task 2")
        user_task = "Pick up the yellow bowl and place it on the measuring scale"
    
    ## 5. Capture environment image for planning
    print("\n## 5. Capturing environment state...")
    environment_image = camera.get_rgba()[:, :, :3]  # RGB only
    print("Environment image captured for vision-based planning\n")
    
    # Record initial frame to video
    video_writer.add_frame(environment_image)

    ## 6. Execute task with multi-agent system
    result = agent_system.run_task(
        user_task=user_task,
        simulator_callback=sim_bridge.callback,
        environment_image=environment_image
    )
    
    ## 7. Display results
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"\nTask: {user_task}")
    print(f"Success: {'YES' if result['success'] else 'NO'}")
    print(f"Strategy Used: {result['strategy']}")
    
    if result['strategy'] == 'action_library':
        print(f"\nActions Executed:")
        for log in result.get('execution_log', []):
            status = '[SUCCESS]' if log['success'] else '[FAILED]'
            print(f"  {status} Subtask {log['subtask_id']}: {log['description']}")
            if log.get('action'):
                print(f"     Action: {log['action']}")
                print(f"     Attempts: {log['attempts']}")
    
    print(f"\nFinal Robot State:")
    final_state = result.get('final_state', {})
    print(f"  Left hand: {final_state.get('left_hand_holding', 'empty')}")
    print(f"  Right hand: {final_state.get('right_hand_holding', 'empty')}")
    print(f"  Objects on scale: {final_state.get('objects_on_scale', [])}")
    
    print("\n" + "="*80)
    
    # Keep simulation running for observation
    print("\nSimulation will continue for 5 seconds for observation...")
    for _ in range(150):  # 5 seconds at 30 Hz
        world.step(render=True)
        # Capture final frames
        obs = camera.get_rgba()[:, :, :3]
        video_writer.add_frame(obs)
    
    # Save video
    print("\n## 8. Finalizing video...")
    video_writer.save()
    
    simulation_app.close()
    print("\nSimulation closed. Goodbye!")


if __name__ == "__main__":
    main()
