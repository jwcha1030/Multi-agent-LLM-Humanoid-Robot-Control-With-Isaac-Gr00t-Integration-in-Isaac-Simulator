"""
Action Library Manager for recording and replaying robot trajectories.

Usage:
1. During simulation, press 'R' to start recording, 'S' to stop and save
2. You'll be prompted to name the action segment
3. Later, load and replay actions using ActionLibrary.load() and ActionLibrary.replay()
"""

import json
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import pickle


class ActionRecorder:
    """Records robot joint positions and actions during simulation."""
    
    def __init__(self):
        self.is_recording = False
        self.current_recording = {
            "frames": [],
            "metadata": {}
        }
        self.library_dir = "./action_library"
        os.makedirs(self.library_dir, exist_ok=True)
        
    def start_recording(self, label: str = None):
        """Start recording a new action sequence."""
        self.is_recording = True
        self.current_recording = {
            "frames": [],
            "metadata": {
                "label": label,
                "start_time": datetime.now().isoformat(),
                "num_frames": 0
            }
        }
        print(f"\n{'='*60}")
        print(f"🔴 RECORDING STARTED: {label if label else 'Unlabeled'}")
        print(f"{'='*60}\n")
        
    def record_frame(self, 
                     joint_positions: np.ndarray,
                     action_joint_position: np.ndarray,
                     gr00t_output: dict,
                     timestep: int,
                     step: int,
                     episode: int,
                     camera_image: np.ndarray = None):
        """Record a single frame of robot state and action."""
        if not self.is_recording:
            return
            
        frame_data = {
            "episode": episode,
            "step": step,
            "timestep": timestep,
            "current_joint_positions": joint_positions.tolist(),
            "action_joint_position": action_joint_position.tolist(),
            "gr00t_output": {
                k: v for k, v in gr00t_output.items()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Optionally save camera image (can be large)
        if camera_image is not None:
            frame_data["camera_shape"] = camera_image.shape
            # Store as compressed numpy array to save space
            
        self.current_recording["frames"].append(frame_data)
        self.current_recording["metadata"]["num_frames"] = len(self.current_recording["frames"])
        
    def stop_recording(self, label: str = None) -> str:
        """Stop recording and save to file."""
        if not self.is_recording:
            print("⚠️  No recording in progress")
            return None
            
        self.is_recording = False
        
        if label:
            self.current_recording["metadata"]["label"] = label
        
        self.current_recording["metadata"]["end_time"] = datetime.now().isoformat()
        
        # Generate filename
        label_str = self.current_recording["metadata"].get("label", "unlabeled")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{label_str}_{timestamp}.json"
        filepath = os.path.join(self.library_dir, filename)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(self.current_recording, f, indent=2)
        
        num_frames = self.current_recording["metadata"]["num_frames"]
        print(f"\n{'='*60}")
        print(f"✅ RECORDING SAVED: {label_str}")
        print(f"   Frames: {num_frames}")
        print(f"   File: {filepath}")
        print(f"{'='*60}\n")
        
        return filepath
        
    def cancel_recording(self):
        """Cancel current recording without saving."""
        if self.is_recording:
            self.is_recording = False
            self.current_recording = {"frames": [], "metadata": {}}
            print("\n❌ Recording cancelled\n")


class ActionLibrary:
    """Load and replay recorded action sequences."""
    
    def __init__(self, library_dir: str = "./action_library"):
        self.library_dir = library_dir
        self.actions = {}
        self.load_library()
        
    def load_library(self):
        """Load all actions from the library directory."""
        if not os.path.exists(self.library_dir):
            return
            
        for filename in os.listdir(self.library_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.library_dir, filename)
                with open(filepath, 'r') as f:
                    action_data = json.load(f)
                    label = action_data["metadata"].get("label", filename[:-5])
                    self.actions[label] = action_data
                    
        print(f"Loaded {len(self.actions)} actions from library")
        
    def list_actions(self):
        """List all available actions."""
        print(f"\n{'='*60}")
        print("Available Actions in Library:")
        print(f"{'='*60}")
        for label, action_data in self.actions.items():
            num_frames = action_data["metadata"]["num_frames"]
            print(f"  • {label}: {num_frames} frames")
        print(f"{'='*60}\n")
        
    def get_action(self, label: str) -> Optional[Dict]:
        """Get action data by label."""
        # Try exact match first
        if label in self.actions:
            return self.actions[label]
            
        # Try fuzzy match
        for key in self.actions.keys():
            if label.lower() in key.lower():
                return self.actions[key]
                
        print(f"⚠️  Action '{label}' not found in library")
        return None
        
    def get_trajectory(self, label: str) -> Optional[List[np.ndarray]]:
        """Get just the joint position trajectory for an action."""
        action_data = self.get_action(label)
        if action_data is None:
            return None
            
        trajectory = []
        for frame in action_data["frames"]:
            joint_pos = np.array(frame["action_joint_position"], dtype=float)
            trajectory.append(joint_pos)
        return trajectory
        
    def save_action(self, label: str, action_data: dict):
        """Save a new action to the library."""
        filename = f"{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.library_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(action_data, f, indent=2)
            
        self.actions[label] = action_data
        print(f"✅ Saved action '{label}' to library")


def print_recorder_instructions():
    """Print keyboard controls for the recorder."""
    print(f"\n{'='*60}")
    print("ACTION RECORDER - Keyboard Controls:")
    print(f"{'='*60}")
    print("  R - Start Recording")
    print("  S - Stop Recording and Save")
    print("  C - Cancel Recording")
    print("  L - List all saved actions")
    print(f"{'='*60}\n")
