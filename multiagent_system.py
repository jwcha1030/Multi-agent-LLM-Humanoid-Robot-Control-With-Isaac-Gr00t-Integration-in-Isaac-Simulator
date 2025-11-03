"""
Multi-Agent Robot Control System using CrewAI

This system coordinates three AI agents to plan, execute, and verify robot tasks:
1. Planner: Breaks down high-level tasks into subtasks
2. Implementer: Selects and executes appropriate actions
3. Supervisor: Verifies successful execution using vision

Architecture:
- Uses CrewAI for agent coordination
- Integrates with Isaac Sim for execution
- Uses GPT-4 Vision for verification
- Maintains state tracking throughout execution
"""

import os
import json
import time
import glob
from typing import List, Dict, Optional
from dotenv import load_dotenv
import numpy as np
from PIL import Image

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI as CommunityChat

# Local imports
from action_library import ActionLibrary
from multiagent_tools import ActionLibraryTool, RobotState

# Load environment variables
load_dotenv()
MAX_RETRIES = 2


class MultiAgentRobotSystem:
    """
    Main multi-agent system for robot task execution.
    """

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        library_dir: str = "./action_library",
        supervisor_free_pass: bool = False,
    ):
        """
        Initialize the multi-agent system.

        Args:
            model_name: OpenAI model to use (gpt-4o, gpt-4, gpt-4-turbo, gpt-5, gpt5-mini)
            library_dir: Path to action library directory
            supervisor_free_pass: If True, supervisor auto-approves all actions without verification
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.model_name = model_name
        self.library_dir = library_dir
        self.supervisor_free_pass = supervisor_free_pass
        self.action_tool = ActionLibraryTool(library_dir)
        self.robot_state = RobotState()

        # Initialize LLM
        self.llm = ChatOpenAI(model=model_name, temperature=0.3, api_key=self.api_key)

        # Create tools as callable objects
        self._create_tools()

        # Create agents
        self.planner_agent = self._create_planner_agent()
        self.implementer_agent = self._create_implementer_agent()
        self.supervisor_agent = self._create_supervisor_agent()

        # Execution tracking
        self.execution_log = []
        self.current_plan = None

    def _create_tools(self):
        """Create tool functions that agents can use."""
        # We need to create these as bound methods wrapped in tool decorator
        # The key is to create them BEFORE agents are created

        @tool("List Available Actions")
        def list_actions():
            """List all available robot actions from the action library."""
            return self.action_tool.list_available_actions()

        @tool("Search Actions")
        def search_actions(task_description: str):
            """
            Search for actions matching a task description.
            Returns list of relevant actions with scores.
            """
            results = self.action_tool.search_actions_for_task(task_description)
            return json.dumps(results, indent=2)

        @tool("Get Action Requirements")
        def get_action_requirements(action_label: str):
            """
            Get prerequisites and expected outcomes for an action.
            """
            requirements = self.action_tool.get_action_requirements(action_label)
            return json.dumps(requirements, indent=2)

        @tool("Get Initial Robot State")
        def get_robot_state():
            """Get current state of robot hands and objects."""
            state = self.robot_state.to_dict()
            return json.dumps(state, indent=2)

        # Store as instance variables
        self.list_actions_tool = list_actions
        self.search_actions_tool = search_actions
        self.get_action_requirements_tool = get_action_requirements
        self.get_robot_state_tool = get_robot_state

    def _create_planner_agent(self) -> Agent:
        """Create the Planner agent."""
        return Agent(
            role="Task Planner",
            goal="Break down high-level robot tasks into executable subtasks",
            backstory="""You are an expert robotics task planner with deep understanding 
            of robot manipulation. You analyze complex tasks and decompose them into sequential 
            subtasks that a robot can execute. You consider hand availability, physical constraints, 
            and spatial relationships between objects.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.list_actions_tool, self.get_robot_state_tool]
        )

    def _create_implementer_agent(self) -> Agent:
        """Create the Implementer/Coder agent."""
        return Agent(
            role="Action Implementer",
            goal="Select and execute the best action from the library for each subtask",
            backstory="""You are a robot action implementer. Given a subtask, you search 
            through available actions, analyze their requirements and outcomes, and select 
            the most appropriate one. You understand prerequisites like which hand is holding 
            what object, and you track state changes after each action.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[
                self.list_actions_tool,
                self.search_actions_tool,
                self.get_action_requirements_tool,
                self.get_robot_state_tool,
            ],
        )

    def _create_supervisor_agent(self) -> Agent:
        """Create the Supervisor agent."""
        return Agent(
            role="Task Supervisor",
            goal="Verify successful execution of each subtask using vision and reasoning",
            backstory="""You are a careful supervisor with vision capabilities. You analyze 
            camera images before and after each action to verify success. You use chain-of-thought 
            reasoning to determine if the robot achieved the desired outcome. If an action fails, 
            you can request retry up to N times before declaring failure.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=[self.get_robot_state_tool],
            multimodal=True,
        )

    def plan_task(
        self, user_task: str, environment_image: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Use Planner agent to decompose task into subtasks.

        Args:
            user_task: High-level task description from user
            environment_image: Optional camera image (not currently used for planning)

        Returns:
            Dictionary with plan including subtasks and strategy
        """
        print(f"\n{'='*60}")
        print("PLANNER AGENT: Creating execution plan")
        print(f"{'='*60}\n")

        # Check if this is the exact trained task
        is_trained_task = self._is_gr00t_pretrained_task(user_task)
        print(f"Is trained task: {is_trained_task}\n")

        planning_task = Task(
            description=f"""
            Analyze this robot task and create an execution plan:
            
            Task: {user_task}
            GR00T pretrained Task : {"pick up the red beaker and tilt it to pour out 1 green nut into yellow bowl. pick up the yellow bowl and place it on the metallic measuring scale"}
            Current robot state: {json.dumps(self.robot_state.to_dict(), indent=2)}

            Instructions:
            1. Determine if this is the exactly the same with GR00T-pretrained task: 
               "Pick up the red beaker and tilt it to pour out 1 green nut into yellow bowl. 
                Pick up the yellow bowl and place it on the metallic measuring scale."
               
            2. If it IS the exactly the same with the pretrained task:
               - Set strategy to "use_gr00t_inference"
               - Create a single subtask: "Run complete trained task via GR00T"

            3. If it is NOT the pretrained task:
               - Set strategy to "use_action_library"
               - Break down into specific subtasks
               - For each subtask, consider:
                 * Which hand should be used
                 * What object is being manipulated
                 * Prerequisites (e.g., hand must be empty/holding object)
                 * State changes after completion
                 * There is no need for confirming empty hands for the first task. Immediately proceed to action.
            
            4. Track hand state throughout:
               - Which hand holds what object after each step
               - When hands need to be empty
               - When objects need to be transferred between hands
            
            Output Format (JSON):
            {{
                "strategy": "use_gr00t_inference" or "use_action_library",
                "reasoning": "Explanation of strategy choice",
                "subtasks": [
                    {{
                        "id": 1,
                        "description": "Clear description",
                        "hand_state_before": {{"left": null, "right": null}},
                        "hand_state_after": {{"left": "object_name", "right": null}},
                        "prerequisites": ["list", "of", "conditions"]
                    }}
                ],
                "expected_outcome": "Final state description"
            }}
            """,
            expected_output="JSON plan with strategy, subtasks, and state tracking",
            agent=self.planner_agent,
        )

        # Create crew and execute
        crew = Crew(
            agents=[self.planner_agent],
            tasks=[planning_task],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        # Parse result
        try:
            plan = json.loads(str(result))
        except json.JSONDecodeError:
            # Fallback parsing
            plan = {
                "strategy": "use_action_library",
                "reasoning": "Could not parse plan JSON",
                "subtasks": self._fallback_parse_subtasks(str(result)),
                "expected_outcome": "Task completion",
            }

        self.current_plan = plan
        return plan

    def _is_gr00t_pretrained_task(self, task: str) -> bool:
        """Check if task is the exact trained GR00T task."""
        trained_task = "pick up the red beaker and tilt it to pour out 1 green nut into yellow bowl. pick up the yellow bowl and place it on the metallic measuring scale"
        return trained_task in task.lower()

    def _fallback_parse_subtasks(self, result_text: str) -> List[Dict]:
        """Fallback parser if JSON parsing fails."""
        # Simple heuristic parsing
        subtasks = []
        lines = result_text.split("\n")
        task_id = 1

        for line in lines:
            line = line.strip()
            if line and (
                line[0].isdigit() or line.startswith("-") or line.startswith("*")
            ):
                # Remove numbering/bullets
                description = line.lstrip("0123456789.-* ")
                if description:
                    subtasks.append(
                        {
                            "id": task_id,
                            "description": description,
                            "hand_state_before": {"left": None, "right": None},
                            "hand_state_after": {"left": None, "right": None},
                            "prerequisites": [],
                        }
                    )
                    task_id += 1

        return subtasks

    def execute_plan(
        self, plan: Dict, simulator_callback=None, max_retries: int = 3
    ) -> Dict:
        """
        Execute the plan using Implementer and Supervisor agents.

        Args:
            plan: Plan from planner agent
            simulator_callback: Callback function to execute actions in simulator
            max_retries: Maximum retry attempts per subtask

        Returns:
            Execution report with success/failure status
        """
        print(f"\n{'='*60}")
        print("EXECUTING PLAN")
        print(f"{'='*60}\n")

        strategy = plan.get("strategy", "use_action_library")

        if strategy == "use_gr00t_inference":
            return self._execute_with_groot(simulator_callback)
        else:
            return self._execute_with_action_library(
                plan, simulator_callback, max_retries
            )

    def _execute_with_groot(self, simulator_callback) -> Dict:
        """Execute using GR00T inference for trained task."""
        print("\nStrategy: Using GR00T inference for trained task\n")

        if simulator_callback:
            success = simulator_callback("use_groot", None)
            return {
                "success": success,
                "strategy": "gr00t_inference",
                "message": "Executed complete trained task via GR00T",
            }
        else:
            return {
                "success": False,
                "strategy": "gr00t_inference",
                "message": "No simulator callback provided",
            }

    def _execute_with_action_library(
        self, plan: Dict, simulator_callback, max_retries: int
    ) -> Dict:
        """Execute plan using action library."""

        print("\nStrategy: Using action library\n")

        # Clean up all previous verification images from previous runs
        verification_dir = "./results/verification_images"
        if os.path.exists(verification_dir):
            old_images = glob.glob(f"{verification_dir}/*.jpg")
            if old_images:
                print(
                    f"Cleaning up {len(old_images)} old verification images from previous runs..."
                )
                for old_image in old_images:
                    try:
                        os.remove(old_image)
                    except:
                        pass
                print("Cleanup complete.\n")

        subtasks = plan.get("subtasks", [])
        execution_log = []

        for subtask in subtasks:
            subtask_id = subtask.get("id", 0)
            description = subtask.get("description", "")

            print(f"\n--- Subtask {subtask_id}: {description} ---\n")

            # Implementer selects action
            selected_action = self._implementer_select_action(subtask)

            if not selected_action:
                execution_log.append(
                    {
                        "subtask_id": subtask_id,
                        "description": description,
                        "action": None,
                        "success": False,
                        "error": "No suitable action found",
                    }
                )
                continue

            # Execute with retries
            success = False
            attempts = 0
            initial_before_image = None  # Store the first BEFORE image

            while attempts < max_retries and not success:
                attempts += 1
                print(f"\nAttempt {attempts}/{max_retries}")

                # Execute action
                if simulator_callback:
                    execution_result = simulator_callback(
                        "execute_action", selected_action
                    )
                else:
                    execution_result = {"success": True}  # Mock for testing

                # Save the initial BEFORE image from first attempt
                if attempts == 1 and "image_before" in execution_result:
                    initial_before_image = execution_result["image_before"]

                # Replace BEFORE image with the initial one for subsequent attempts
                if attempts > 1 and initial_before_image is not None:
                    execution_result["image_before"] = initial_before_image

                # Supervisor verification
                verification = self._supervisor_verify(
                    subtask,
                    selected_action,
                    execution_result,
                    free_pass=self.supervisor_free_pass,
                )

                success = verification["success"]

                if not success and attempts < max_retries:
                    print(
                        f"Verification failed: {verification.get('reasoning', 'Unknown')}"
                    )
                    print("Retrying...\n")

            # Update robot state
            if success:
                self.robot_state.update(selected_action, True)
                print(f"Subtask {subtask_id} completed successfully\n")
            else:
                print(f"Subtask {subtask_id} failed after {max_retries} attempts\n")

            execution_log.append(
                {
                    "subtask_id": subtask_id,
                    "description": description,
                    "action": selected_action,
                    "attempts": attempts,
                    "success": success,
                    "verification": verification,
                }
            )

            # Stop if failed
            if not success:
                break

        # Overall success
        all_success = all(log["success"] for log in execution_log)

        return {
            "success": all_success,
            "strategy": "action_library",
            "execution_log": execution_log,
            "final_state": self.robot_state.to_dict(),
        }

    def _implementer_select_action(self, subtask: Dict) -> Optional[str]:
        """Use Implementer agent to select best action for subtask."""
        print("IMPLEMENTER: Selecting action...")

        description = subtask.get("description", "")

        selection_task = Task(
            description=f"""
            Select the best action from the library for this subtask:
            
            Subtask: {description}
            Prerequisites: {subtask.get('prerequisites', [])}
            Current robot state: {json.dumps(self.robot_state.to_dict())}
            
            Instructions:
            1. Search for actions matching this subtask
            2. Check action requirements (prerequisites, hand usage, etc.)
            3. Verify current robot state meets prerequisites
            4. Select the single best matching action
            5. Return ONLY the action label (e.g., "pick_up_yellow_bowl_1")
            
            Output: Just the action label, nothing else
            """,
            expected_output="Action label from library",
            agent=self.implementer_agent,
        )

        crew = Crew(
            agents=[self.implementer_agent],
            tasks=[selection_task],
            process=Process.sequential,
            verbose=False,
        )

        result = str(crew.kickoff()).strip()

        # Clean up result
        result = result.strip("\"'")

        # Verify action exists
        if result in self.action_tool.library.actions:
            print(f"-> Selected action: {result}\n")
            return result
        else:
            print(f"-> Action not found in library: {result}\n")
            return None

    def _supervisor_verify(
        self,
        subtask: Dict,
        action: str,
        execution_result: Dict,
        free_pass: bool = False,
    ) -> Dict:
        """
        Use Supervisor agent to verify execution success.

        Args:
            subtask: Subtask information
            action: Action that was executed
            execution_result: Result from action execution
            free_pass: If True, automatically verify as successful without LLM analysis

        Returns:
            Dictionary with verification result
        """
        print("SUPERVISOR: Verifying execution...")

        # Get images if available
        before_image = execution_result.get("image_before")
        after_image = execution_result.get("image_after")

        # Prepare vision context
        if before_image is not None and after_image is not None:
            print("  Using vision analysis (before/after images)...")

            # Save images for manual verification
            os.makedirs("./results/verification_images", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            action_clean = action.replace("/", "_").replace(" ", "_")

            before_path = (
                f"./results/verification_images/{timestamp}_{action_clean}_BEFORE.jpg"
            )
            after_path = (
                f"./results/verification_images/{timestamp}_{action_clean}_AFTER.jpg"
            )

            # Convert numpy arrays to PIL Images and save
            if before_image.dtype != np.uint8:
                before_image_uint8 = (before_image * 255).astype(np.uint8)
            else:
                before_image_uint8 = before_image

            if after_image.dtype != np.uint8:
                after_image_uint8 = (after_image * 255).astype(np.uint8)
            else:
                after_image_uint8 = after_image

            Image.fromarray(before_image_uint8).save(before_path)
            Image.fromarray(after_image_uint8).save(after_path)
            
            # Verify images were saved successfully
            if not os.path.exists(before_path) or not os.path.exists(after_path):
                print("  Warning: Failed to save verification images")
                return {
                    "success": False,
                    "reasoning": "Failed to save verification images to disk",
                    "confidence": 0.0,
                }
            
            print(f"  Saved verification images:")
            print(f"    Before: {before_path}")
            print(f"    After:  {after_path}")

            # Check free-pass mode
            if free_pass:
                print("  FREE-PASS MODE: Skipping LLM verification, auto-approving")
                return {
                    "success": True,
                    "reasoning": "Free-pass mode enabled - verification skipped",
                    "confidence": 1.0,
                }

            # Use OpenAI Vision API directly for fast image analysis (via ActionLibraryTool)
            print("  Using OpenAI Vision API to analyze images...")
            visual_analysis = self.action_tool.analyze_images_with_openai(
                before_image=before_image,
                after_image=after_image,
                subtask=subtask,
                action=action,
                api_key=self.api_key,
                model_name=self.model_name
            )
            print(f"  Obtained visual analysis ({len(visual_analysis)} chars)")
            
            # Now use Supervisor agent with text-based visual context (much faster)
            print("  Using Supervisor agent for verification decision...")
            verification_task = Task(
                description=f"""Verify if the robot action was successful based on visual analysis from the camera system.

                TASK DETAILS:
                - Subtask Goal: {subtask.get('description', '')}
                - Action Executed: {action}
                - Expected Outcome: {subtask.get('hand_state_after', {})}
                - Robot State Before Action: {json.dumps(self.robot_state.to_dict())}

                VISUAL ANALYSIS FROM CAMERA:
                {visual_analysis}

                INSTRUCTIONS:
                You are an expert supervisor verifying robot action execution. You have received a detailed visual report from the vision system comparing BEFORE and AFTER images.

                Using the visual analysis above, determine if the action succeeded:

                1. What was the specific goal of this subtask?
                2. What visual changes should occur if the action succeeded?
                3. According to the visual analysis, did those expected changes occur?
                4. Check for failure indicators mentioned in the visual report:
                - Object dropped or fell
                - Object in wrong location  
                - Hand is empty when it should be holding something
                - Hand is holding something when it should be empty
                - Object collision or damage
                5. Does the visual analysis confirm the expected outcome was achieved?

                Be precise in your assessment. If any critical element is missing or wrong in the visual report, mark as failure.

                OUTPUT FORMAT (JSON only, no other text):
                {{
                    "success": true or false,
                    "reasoning": "Your step-by-step reasoning based on the visual analysis and expected outcome",
                    "confidence": 0.0 to 1.0 (0.0 = completely uncertain, 1.0 = absolutely certain)
                }}""",
                expected_output="JSON verification result with success boolean, detailed reasoning string, and confidence float",
                agent=self.supervisor_agent,
            )

            crew = Crew(
                agents=[self.supervisor_agent],
                tasks=[verification_task],
                process=Process.sequential,
                verbose=False,  # Set to True for debugging multimodal tool usage
            )
            
            try:
                result = crew.kickoff()
                
                # Parse JSON response with improved robustness
                import re
                result_str = str(result)
                
                # Try direct JSON parsing first
                try:
                    verification = json.loads(result_str)
                except json.JSONDecodeError:
                    # Extract JSON from text using regex
                    json_match = re.search(
                        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", result_str, re.DOTALL
                    )
                    if json_match:
                        try:
                            verification = json.loads(json_match.group())
                        except json.JSONDecodeError:
                            verification = None
                    else:
                        verification = None
                
                # Validate and use fallback if needed
                if verification and isinstance(verification, dict):
                    # Ensure required fields exist
                    if "success" not in verification:
                        verification["success"] = False
                    if "reasoning" not in verification:
                        verification["reasoning"] = "No reasoning provided"
                    if "confidence" not in verification:
                        verification["confidence"] = 0.5
                else:
                    # Complete fallback
                    verification = {
                        "success": False,
                        "reasoning": f"Could not parse verification response: {result_str[:200]}",
                        "confidence": 0.3,
                    }

                status = "VERIFIED" if verification["success"] else "FAILED"
                confidence_pct = int(verification.get("confidence", 0.5) * 100)
                print(f"  {status} (confidence: {confidence_pct}%)")
                print(f"  Reasoning: {verification.get('reasoning', '')}\n")
                return verification

            except Exception as e:
                print(f"  Error in vision analysis: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "success": False,
                    "reasoning": f"Vision analysis failed: {str(e)}",
                    "confidence": 0.0,
                }
        else:
            # No images available - use state-based verification with CrewAI
            print("  No images available, using state-based verification only...")

            # Check free-pass mode (even without images)
            if free_pass:
                print("  FREE-PASS MODE: Skipping verification, auto-approving")
                return {
                    "success": True,
                    "reasoning": "Free-pass mode enabled - verification skipped",
                    "confidence": 1.0,
                }

            # Use CrewAI task for non-vision verification
            verification_task = Task(
                description=f"""
                Verify if this robot action was successfully executed:
                
                Subtask Goal: {subtask.get('description', '')}
                Action Executed: {action}
                Expected Outcome: {subtask.get('hand_state_after', {})}
                
                Initial Robot State (before action): {json.dumps(self.robot_state.to_dict())}
                
                Note: No images are available. You must infer success based on:
                - The action that was executed
                - The expected outcome
                - Common success patterns for this type of action
                
                Use chain-of-thought reasoning:
                1. What was the goal of this subtask?
                2. What should have changed after the action?
                3. Based on the action type, is it likely to have succeeded?
                4. Are there known failure modes for this action?
                
                Output Format (JSON):
                {{
                    "success": true/false,
                    "reasoning": "Step-by-step explanation",
                    "confidence": 0.0-1.0
                }}
                """,
                expected_output="JSON verification result",
                agent=self.supervisor_agent,
            )

            crew = Crew(
                agents=[self.supervisor_agent],
                tasks=[verification_task],
                process=Process.sequential,
                verbose=False,
            )

            result = crew.kickoff()

            # Parse result
            try:
                verification = json.loads(str(result))
            except:
                # Fallback
                verification = {
                    "success": True,  # Optimistic default
                    "reasoning": "Could not parse verification",
                    "confidence": 0.5,
                }

            status = "VERIFIED" if verification["success"] else "FAILED"
            print(f"{status}: {verification.get('reasoning', '')}\n")

            return verification

    def run_task(
        self,
        user_task: str,
        simulator_callback=None,
        environment_image: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Complete workflow: plan and execute task.

        Args:
            user_task: High-level task from user
            simulator_callback: Function to execute actions in simulator
            environment_image: Optional camera image for vision-based planning

        Returns:
            Complete execution report
        """
        print(f"\n{'='*80}")
        print("MULTI-AGENT ROBOT CONTROL SYSTEM")
        print(f"{'='*80}")
        print(f"\nUser Task: {user_task}\n")

        # Step 1: Plan (with optional vision)
        plan = self.plan_task(user_task, environment_image=environment_image)

        print(f"\n{'='*60}")
        print("PLAN CREATED")
        print(f"{'='*60}")
        print(json.dumps(plan, indent=2))
        print()

        # Step 2: Execute
        result = self.execute_plan(plan, simulator_callback, max_retries=MAX_RETRIES)

        print(f"\n{'='*80}")
        print("TASK EXECUTION COMPLETE")
        print(f"{'='*80}")
        print(
            f"Success: {result['success']['success'] if not isinstance(result['success'],bool) else result['success']}"
        )
        print(f"Strategy: {result['strategy']}")
        print(
            f"\nFinal Robot State: {json.dumps(result.get('final_state', {}), indent=2)}"
        )
        print(f"{'='*80}\n")

        return result


if __name__ == "__main__":
    # Test the system without simulator
    system = MultiAgentRobotSystem()

    # Example task
    task = "Pick up the yellow bowl and place it on the measuring scale"

    result = system.run_task(task)

    print("\nExecution Result:")
    print(json.dumps(result, indent=2))
