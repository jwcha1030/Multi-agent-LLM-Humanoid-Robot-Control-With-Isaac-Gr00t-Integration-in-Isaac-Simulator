"""
Test Multi-Agent System (No Simulator Required)

This script tests the multi-agent system's planning and reasoning
without requiring Isaac Sim to be running.
"""

import os
from multiagent_system import MultiAgentRobotSystem
from dotenv import load_dotenv

def custom_break():
    input("Press Enter to continue...")

# Load environment variables
load_dotenv()

def mock_simulator_callback(command: str, data: any) -> dict:
    """
    Mock callback that simulates successful execution.
    Use this to test agent reasoning without Isaac Sim.
    """
    print(f"\nMOCK EXECUTION:")
    print(f"   Command: {command}")
    print(f"   Data: {data}")
    
    # Simulate successful execution
    return {
        "success": True,
        "image_before": None,  # Would be actual image in real scenario
        "image_after": None
    }


def test_planning_only():
    """Test just the planning phase without execution."""
    print("="*80)
    print("TEST 1: Planning Only (No Execution)")
    print("="*80)
    
    system = MultiAgentRobotSystem(
        model_name="gpt-5-mini",
        library_dir="./action_library"
    )

    test_tasks = [
        "Pick up the red beaker and tilt it to pour out 1 green nut into yellow bowl. Pick up the yellow bowl and place it on the metallic measuring scale.",
        "Pick up the yellow bowl and place it on the measuring scale",
        "Grab the red beaker and move it to the blue box"
    ]
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n{'─'*80}")
        print(f"Task {i}: {task}")
        print('─'*80)
        
        # Just plan, don't execute
        plan_result = system.plan_task(task)
        
        print(f"\nPlan Result:")
        print(f"   Strategy: {plan_result.get('strategy', 'unknown')}")
        
        if plan_result['strategy'] == 'action_library':
            subtasks = plan_result.get('subtasks', [])
            print(f"   Subtasks: {len(subtasks)}")
            for j, subtask in enumerate(subtasks, 1):
                print(f"      {j}. {subtask['description']}")


def test_with_mock_execution():
    """Test complete flow with mock execution."""
    print("\n\n" + "="*80)
    print("TEST 2: Complete Flow with Mock Execution")
    print("="*80)
    
    system = MultiAgentRobotSystem(
        model_name="gpt-5-mini",
        library_dir="./action_library"
    )
    
    task = "Pick up the yellow bowl and place it on the measuring scale"
    
    print(f"\nTask: {task}\n")
    
    result = system.run_task(
        user_task=task,
        simulator_callback=mock_simulator_callback
    )
    
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    print(f"\nSuccess: {'YES' if result['success'] else 'NO'}")
    print(f"Strategy: {result['strategy']}")
    
    if result['strategy'] == 'action_library':
        print(f"\nExecution Log:")
        for log in result.get('execution_log', []):
            status = '[SUCCESS]' if log['success'] else '[FAILED]'
            print(f"  {status} Subtask {log['subtask_id']}: {log['description']}")
            if log.get('action'):
                print(f"     Action: {log['action']}")
                print(f"     Attempts: {log['attempts']}")
    
    print(f"\nFinal State:")
    final_state = result.get('final_state', {})
    print(f"  Left hand: {final_state.get('left_hand_holding', 'empty')}")
    print(f"  Right hand: {final_state.get('right_hand_holding', 'empty')}")


def test_action_library_search():
    """Test action library search functionality."""
    print("\n\n" + "="*80)
    print("TEST 3: Action Library Search")
    print("="*80)
    
    from multiagent_tools import ActionLibraryTool
    
    tool = ActionLibraryTool(library_dir="./action_library")
    
    # Test 1: List all actions
    print("\nAll Available Actions:")
    actions = tool.list_available_actions()
    print(actions)

    # Test 2: Search for specific task
    print("\n\nSearch Results for 'pick up yellow bowl':")
    results = tool.search_actions_for_task("pick up yellow bowl")
    print(results)

    
    # Test 3: Get action requirements
    print("\n\nRequirements for 'pick_up_yellow_bowl_1':")
    reqs = tool.get_action_requirements("pick_up_yellow_bowl_1")
    print(reqs)



def main():
    """Run all tests."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment")
        print("Please create a .env file with your OpenAI API key:")
        print("  echo 'OPENAI_API_KEY=sk-proj-...' > .env")
        return
    
    print("Testing Multi-Agent System")
    print(f"API Key found: {os.getenv('OPENAI_API_KEY')[:20]}...")

    try:
        # Run tests
        test_action_library_search()
        test_planning_only()
        test_with_mock_execution()
        
        print("\n\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)
        print("\nNext steps:")
        print("1. Review the agent reasoning above")
        print("2. Install dependencies: isaac-py -m pip install -r requirements_multiagent.txt")
        print("3. Run with simulator: isaac-py run_multiagent_sim.py")
        
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
