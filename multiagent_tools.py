"""
Multi-Agent Action Tools for Robot Control

This module provides tools that agents can use to:
1. Query available actions from the action library
2. Execute recorded actions
3. Call GR00T for inference on untrained tasks
4. Get robot and world state
"""

import json
import os
import base64
import numpy as np
from typing import List, Dict, Optional, Any
from action_library import ActionLibrary
from PIL import Image
import cv2

class RobotState:
    """Tracks current state of the robot and objects."""
    
    def __init__(self):
        self.left_hand_holding = None
        self.right_hand_holding = None
        self.objects_on_table = []
        self.objects_on_scale = []
        self.last_action = None
        self.action_history = []
        
    def update(self, action: str, success: bool):
        """Update state based on action execution."""
        self.last_action = action
        self.action_history.append({
            "action": action,
            "success": success
        })
        
        # Parse action to update object locations
        action_lower = action.lower()
        
        # Pick actions
        if "pick" in action_lower and "red beaker" in action_lower:
            if success:
                self.left_hand_holding = "red_beaker"
        elif "pick" in action_lower and "yellow bowl" in action_lower:
            if success:
                self.right_hand_holding = "yellow_bowl"
        
        # Drop actions
        elif "drop" in action_lower:
            if "left hand" in action_lower and success:
                self.left_hand_holding = None
            elif "right hand" in action_lower and success:
                if "scale" in action_lower:
                    self.objects_on_scale.append(self.right_hand_holding)
                self.right_hand_holding = None
        
        # Switch hands
        elif "switch" in action_lower and success:
            temp = self.left_hand_holding
            self.left_hand_holding = self.right_hand_holding
            self.right_hand_holding = temp
    
    def to_dict(self) -> Dict:
        """Return state as dictionary."""
        return {
            "left_hand_holding": self.left_hand_holding,
            "right_hand_holding": self.right_hand_holding,
            "objects_on_scale": self.objects_on_scale,
            "last_action": self.last_action
        }


class ActionLibraryTool:
    """Tool for querying and executing actions from the library."""
    
    def __init__(self, library_dir: str = "./action_library"):
        self.library = ActionLibrary(library_dir)
        self.robot_state = RobotState()
        
    def list_available_actions(self) -> str:
        """List all available actions with descriptions."""
        if not self.library.actions:
            return "No actions available in library."
        
        result = "Available Actions:\n\n"
        for label, action_data in self.library.actions.items():
            num_frames = action_data["metadata"]["num_frames"]
            description = self._generate_description(label)
            result += f"• {label}\n"
            result += f"  Description: {description}\n"
            result += f"  Duration: {num_frames} frames\n\n"
        
        return result
    
    def _generate_description(self, label: str) -> str:
        """Generate human-readable description from action label."""
        label_lower = label.lower()
        
        descriptions = {
            "pick_red_beaker": "Grasp the red beaker with left hand from the table",
            "pick_up_yellow_bowl": "Grasp the yellow bowl with right hand from the table",
            "tilt_and_pour": "Tilt the item in left hand to pour contents into yellow bowl",
            "move_item_in_right_hand_over_to_measuring_scale": "Move right hand (holding item) above the measuring scale",
            "move_item_in_left_hand_over_the_blue_box": "Move left hand (holding item) above the blue box",
            "drop_item_in_right_hand_on_measuring_scale": "Release item from right hand onto the measuring scale",
            "drop_item_in_left_hand_into_box": "Release item from left hand into the blue box",
            "switch_item_in_left_to_right_hand": "Transfer item from left hand to right hand",
            "drop_item_in_right_hand_on_the_table": "Place item from right hand back on the table"
        }
        
        # Try exact match first
        for key, desc in descriptions.items():
            if key in label_lower:
                return desc
        
        # Generate generic description
        return f"Robot action: {label.replace('_', ' ')}"
    
    def get_action_requirements(self, action_label: str) -> Dict:
        """Get prerequisites and expected outcomes for an action."""
        action_lower = action_label.lower()

        requirements = {
            "prerequisites": [],
            "expected_outcome": "",
            "hand_used": None,
            "object_interaction": None
        }
        
        # Analyze action
        if "pick" in action_lower:
            if "red" in action_lower and "beaker" in action_lower:
                requirements["prerequisites"] = ["red_beaker_on_table", "left_hand_empty"]
                requirements["expected_outcome"] = "red_beaker_in_left_hand"
                requirements["hand_used"] = "left"
                requirements["object_interaction"] = "red_beaker"
            elif "yellow" in action_lower and "bowl" in action_lower:
                requirements["prerequisites"] = ["yellow_bowl_on_table", "right_hand_empty"]
                requirements["expected_outcome"] = "yellow_bowl_in_right_hand"
                requirements["hand_used"] = "right"
                requirements["object_interaction"] = "yellow_bowl"
        
        elif "drop" in action_lower or "place" in action_lower:
            if "right" in action_lower and "hand" in action_lower:
                requirements["prerequisites"] = ["right_hand_holding_object"]
                requirements["hand_used"] = "right"
                if "scale" in action_lower:
                    requirements["expected_outcome"] = "object_on_scale"
            elif "left" in action_lower and "hand" in action_lower:
                requirements["prerequisites"] = ["left_hand_holding_object"]
                requirements["hand_used"] = "left"
                if "box" in action_lower:
                    requirements["expected_outcome"] = "object_in_box"
        
        elif "move" in action_lower:
            if "right" in action_lower and "hand" in action_lower:
                requirements["prerequisites"] = ["right_hand_holding_object"]
                requirements["hand_used"] = "right"
            elif "left" in action_lower and "hand" in action_lower:
                requirements["prerequisites"] = ["left_hand_holding_object"]
                requirements["hand_used"] = "left"
        
        elif "switch" in action_lower:
            requirements["prerequisites"] = ["left_hand_holding_object", "right_hand_empty"]
            requirements["expected_outcome"] = "object_switched_to_right_hand"
        return requirements
    
    def search_actions_for_task(self, task_description: str) -> List[Dict]:
        """Search for actions that match a task description."""
        task_lower = task_description.lower()
        matching_actions = []
        
        for label in self.library.actions.keys():
            label_lower = label.lower()
            
            # Calculate relevance score
            score = 0
            keywords = task_lower.split()
            
            for keyword in keywords:
                if keyword in label_lower:
                    score += 1
            
            if score > 0:
                matching_actions.append({
                    "action": label,
                    "relevance_score": score,
                    "description": self._generate_description(label),
                    "requirements": self.get_action_requirements(label)
                })

        # Sort by relevance
        matching_actions.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return matching_actions
    
    def get_robot_state(self) -> Dict:
        """Get current state of robot hands and objects."""
        return self.robot_state.to_dict()
    
    def encode_image_to_base64(self, image: np.ndarray) -> str:
        """Encode numpy image to base64 for LLM analysis."""
        # rgb_np: HxWx3, uint8 (your camera frame)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        assert ok, "JPEG encode failed"
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    
    def analyze_images_with_openai(
        self, 
        before_image: np.ndarray, 
        after_image: np.ndarray, 
        subtask: Dict,
        action: str,
        api_key: str,
        model_name: str = "gpt-4o"
    ) -> str:
        """
        Use OpenAI Vision API to analyze before/after images and return detailed text description.
        
        Args:
            before_image: Image before action execution
            after_image: Image after action execution
            subtask: Subtask information
            action: Action that was executed
            api_key: OpenAI API key
            model_name: Vision-capable model name (gpt-4o-mini, gpt-4o, gpt-4-vision-preview)
            
        Returns:
            Detailed textual analysis of visual changes between images
        """
        from openai import OpenAI
        model_name = "gpt-4o" #using only gpt-4o for now

        print(f"  Encoding images to base64...")
        # Encode images to base64
        before_b64 = self.encode_image_to_base64(before_image)
        after_b64 = self.encode_image_to_base64(after_image)
        print(f"    Before image: {len(before_b64)} chars")
        print(f"    After image: {len(after_b64)} chars")
        
        # Create OpenAI client
        client = OpenAI(api_key=api_key)
        print(f"  Using model: {model_name}")
        
        # Prepare comprehensive vision analysis prompt
        prompt = f"""You are an expert vision system analyzing robot manipulation actions.

            TASK CONTEXT:
            - Subtask Goal: {subtask.get('description', '')}
            - Action Executed: {action}
            - Expected Outcome: {subtask.get('hand_state_after', {})}

            INSTRUCTIONS:
            Analyze the BEFORE and AFTER images to provide a detailed visual report. Focus on:

            1. **Object Locations**: Where are key objects (beakers, bowls, hands, scale, box) in each image?
            2. **Hand States**: What is each hand holding (or is it empty) in BEFORE vs AFTER?
            3. **Visual Changes**: What specifically changed between the two images?
            4. **Object Integrity**: Are objects intact, dropped, spilled, or damaged?
            5. **Spatial Relationships**: Position changes relative to surfaces (table, scale, box)?
            6. **Goal Achievement**: Did the visual changes match what was expected?

            Provide a comprehensive visual description that someone without seeing the images could use to determine if the action succeeded.

            Be extremely detailed and specific. Include:
            - Exact positions (left/right, top/bottom, near/far)
            - Colors and shapes of objects
            - Hand positions and grasping states  
            - Any objects on surfaces (table, scale, box)
            - Any signs of failure (objects on floor, spills, collisions, dropped items)

            Format your response as a detailed narrative report with clear sections."""

        try:
            # Call OpenAI Vision API
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": before_b64,
                                    "detail": "high"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": after_b64,
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_completion_tokens=1000  # Use max_completion_tokens for newer models
            )
            
            # Debug: Check full response structure
            print(f"  DEBUG - OpenAI Response:")
            print(f"    Model: {response.model}")
            print(f"    Finish Reason: {response.choices[0].finish_reason}")
            print(f"    Message Content Type: {type(response.choices[0].message.content)}")
            print(f"    Message Content Length: {len(response.choices[0].message.content) if response.choices[0].message.content else 0}")
            
            visual_analysis = response.choices[0].message.content
            
            if not visual_analysis or visual_analysis.strip() == "":
                # Provide more diagnostic information
                error_msg = f"Visual analysis returned empty response.\n"
                error_msg += f"  Model: {response.model}\n"
                error_msg += f"  Finish reason: {response.choices[0].finish_reason}\n"
                error_msg += f"  Content type: {type(visual_analysis)}\n"
                error_msg += f"  Content repr: {repr(visual_analysis)}\n"
                print(f"  ERROR: {error_msg}")
                return error_msg
            
            print(f"  SUCCESS - Visual analysis received ({len(visual_analysis)} chars)")
            return visual_analysis
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"  ERROR in OpenAI Vision API call: {str(e)}")
            print(f"  Traceback: {error_details}")
            return f"Visual analysis failed: {str(e)}\n{error_details}"


# Tool function wrappers for CrewAI
def list_actions_tool() -> str:
    """
    List all available robot actions from the action library.
    Returns detailed information about each action including description and duration.
    """
    tool = ActionLibraryTool()
    return tool.list_available_actions()


def search_actions_tool(task_description: str) -> str:
    """
    Search for actions that match a given task description.
    
    Args:
        task_description: Description of the task to find matching actions for
        
    Returns:
        JSON string of matching actions with relevance scores
    """
    tool = ActionLibraryTool()
    results = tool.search_actions_for_task(task_description)
    return json.dumps(results, indent=2)


def get_action_requirements_tool(action_label: str) -> str:
    """
    Get prerequisites and expected outcomes for a specific action.
    
    Args:
        action_label: Label of the action to query
        
    Returns:
        JSON string with prerequisites, expected outcomes, hand usage, etc.
    """
    tool = ActionLibraryTool()
    requirements = tool.get_action_requirements(action_label)
    return json.dumps(requirements, indent=2)


def get_robot_state_tool() -> str:
    """
    Get current state of the robot including what objects are held in each hand.
    
    Returns:
        JSON string with robot state information
    """
    tool = ActionLibraryTool()
    state = tool.get_robot_state()
    return json.dumps(state, indent=2)
