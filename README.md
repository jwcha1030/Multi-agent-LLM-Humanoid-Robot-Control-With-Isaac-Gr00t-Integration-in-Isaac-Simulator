# Multi-agent LLM Humanoid Robot Control With Isaac GR00T Integration In Isaac Simulator

A sophisticated multi-agent AI system that enables humanoid robots to execute complex manipulation tasks by intelligently combining vision-language-action models (NVIDIA GR00T) with recorded action trajectories, orchestrated by LLM agents using CrewAI.

## Project Overview

This project addresses a fundamental limitation in robotic AI: **GR00T models trained on complete task episodes cannot generalize to partial tasks or novel compositions**. The solution uses a multi-agent LLM system to intelligently decompose tasks and compose atomic actions.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER TASK INPUT                            │
│   "Pick up the yellow bowl and place it on the measuring scale"     │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│  PLANNER AGENT (LLM)                                                │
│  • Determines strategy: GR00T inference vs Action Library           │
│  • Decomposes task into sequential subtasks                         │
│  • Analyzes available actions and robot state                       │
│  Output: {strategy, subtasks[], reasoning}                          │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│  IMPLEMENTER AGENT (GPT-4o)                                         │
│  • Searches action library for matching actions                     │
│  • Checks prerequisites (hand state, object availability)           │
│  • Selects optimal action for current subtask                       │
│  Output: action_label                                               │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│  ISAAC SIM EXECUTION                                                │
│  • GR00T Inference Server (trained task) OR                         │
│  • Action Library Replay (novel task composition)                   │
│  • Captures before/after camera images                              │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│  SUPERVISOR AGENT (Two-Stage Vision Verification)                   │
│  Stage 1: OpenAI Vision API analyzes before/after images            │
│           → Returns detailed textual visual analysis                │
│  Stage 2: Supervisor Agent processes text description               │
│           → Makes verification decision with chain-of-thought       │
│  • Triggers retry (max 3) if verification fails                     │
│  Output: {success, reasoning, confidence}                           │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Features

- **Intelligent Strategy Selection**: Automatically chooses between GR00T inference (for trained tasks) or action composition (for novel tasks)
- **Vision-Based Verification**: Supervisor uses GPT vision models to verify task success by comparing before/after images
- **State Tracking**: Maintains robot hand state and object locations throughout execution
- **Automatic Retry Logic**: Up to 3 retry attempts with visual verification after each attempt
- **Action Library**: Record and reuse successful robot trajectories as atomic actions
- **Complete Audit Trail**: Detailed execution logs with success/failure reasoning

## Prerequisites

### Required Software

1. **NVIDIA Isaac Sim**: Download and install from [Isaac Sim Installation Guide](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/index.html)

2. **NVIDIA GR00T SDK**: Follow installation instructions at [Isaac GR00T GitHub](https://github.com/NVIDIA/Isaac-GR00T/tree/main?tab=readme-ov-file)

### Required Environments

You need **two separate conda environments**:

1. **`gr00t` environment**: For GR00T training and inference server
   - NVIDIA GR00T SDK
   - PyTorch with CUDA
   - FastAPI, Uvicorn
   - HuggingFace Transformers

2. **`isaacsim` environment**: For Isaac Sim simulation and multi-agent control
   - NVIDIA Isaac Sim
   - CrewAI and LangChain
   - OpenAI Python SDK
   - OpenCV, NumPy, PIL

### API Keys

- **OpenAI API Key**: Required for LLM agents (gpt-4o-mini or newer recommended for vision capabilities)

### Isaac Sim Python Command

Throughout this guide, we use `isaac-py` as a shorthand for the Isaac Sim Python executable. You should replace this with the actual path to your Isaac Sim Python:

```bash
# Example: Create an alias (add to ~/.bashrc or ~/.zshrc)
alias isaac-py="/path/to/isaac-sim/python.sh"

# Or use the full path directly:
/path/to/isaac-sim/python.sh run_multiagent_sim.py
```

## Quick Start

### 1. Environment Setup

```bash
# Activate Isaac Sim environment
conda activate isaacsim
cd /path/to/multi_agent_llm_humanoid_manipulation

# Install multi-agent dependencies (replace isaac-py with your Isaac Sim Python path)
/path/to/isaac-sim/python.sh -m pip install crewai langchain-openai python-dotenv pillow opencv-python

# Create .env file with your OpenAI API key
echo "OPENAI_API_KEY=sk-proj-your-key-here" > .env
```

### 2. Dataset Preparation

Download a pre-trained task dataset from HuggingFace:

```bash
# Switch to gr00t environment
conda activate gr00t
cd /path/to/multi_agent_llm_humanoid_manipulation

# Edit download_dataset.py to select dataset
# Options: "Nut-Pouring-task" or "Exhaust-Pipe-Sorting-task"
python download_dataset.py
```

The dataset will be downloaded to `./datasets/[DATASET_NAME]/`

### 3. Model Fine-tuning (Optional)

Fine-tune NVIDIA's GR00T-N1.5-3B model on your task:

```bash
# Make sure you're in gr00t environment
conda activate gr00t

# Configure fine-tuning parameters in run_finetune.py:
# - DATASET_PATH: Path to downloaded dataset
# - BATCH_SIZE: 32 (adjust based on GPU memory)
# - MAX_STEPS: 30000 (training iterations)
# - TUNE_PROJECTOR: True (recommended)
# - TUNE_DIFFUSION_MODEL: False (keep frozen for stability)

python run_finetune.py
```

**Training Progress:**
- Checkpoints saved every 10,000 steps to `./finetuned/`
- Monitor via Weights & Biases if configured
- Typical training time: 4-8 hours on single GPU

### 4. Start GR00T Inference Server

Launch the inference server for trained task execution:

```bash
# Terminal 1: gr00t environment
conda activate gr00t

# Edit run_inference_server.py to set MODEL_PATH:
# - For pre-trained: "nvidia/GR00T-N1.5-3B"
# - For fine-tuned: "./finetuned/your_model_name"

python run_inference_server.py
```

The server starts on `http://localhost:9876` with endpoints:
- `POST /inference`: Execute trained task

Keep this terminal running during simulation.

### 5. Run Multi-Agent Robot Control

Launch Isaac Sim with the multi-agent LLM system:

```bash
# Terminal 2: isaacsim environment
conda activate isaacsim
cd /path/to/multi_agent_llm_humanoid_manipulation

# Run the multi-agent simulation (replace isaac-py with your Isaac Sim Python path)
/path/to/isaac-sim/python.sh run_multiagent_sim.py
```

**What Happens:**
1. Isaac Sim loads the environment (table, objects, robot)
2. User is prompted to select or enter a task
3. Multi-agent system plans and executes the task
4. Camera captures before/after images for verification
5. Video recording saved to `./results/multiagent_execution.mp4`

## Project Structure

```
multi_agent_llm_humanoid_manipulation/
├── multiagent_system.py          # Core multi-agent orchestration (3 AI agents)
├── multiagent_tools.py           # Tool layer (action search, state tracking, vision analysis)
├── run_multiagent_sim.py         # Isaac Sim + multi-agent integration
├── run_inference_server.py       # GR00T FastAPI inference server
├── run_finetune.py              # GR00T model fine-tuning script
├── run_simulation.py            # Basic Isaac Sim rollout (GR00T only)
├── run_action_replay.py         # Replay recorded action sequences
├── download_dataset.py          # Download training datasets
├── test_multiagent.py           # Test multi-agent system without simulator
│
├── action_library.py            # Action recording and retrieval system
├── gr1_config.py               # Robot configuration (joint limits, etc.)
├── gr1_gr00t_utils.py          # GR00T input/output formatting utilities
│
├── action_library/             # Recorded robot action trajectories (JSON)
│   ├── pick_up_yellow_bowl_1.json
│   ├── drop_item_in_right_hand_on_measuring_scale_1.json
│   └── ... (11 actions total)
│
├── datasets/                   # Training datasets (LeRobot format)
│   ├── Nut-Pouring-task/
│   └── Exhaust-Pipe-Sorting-task/
│
├── finetuned/                 # Fine-tuned GR00T model checkpoints
│   └── gr1_arms_only.Nut_pouring_batch32_nodiffusion/
│
├── results/                   # Execution outputs
│   ├── multiagent_execution.mp4    # Recorded simulation video
│   └── verification_images/        # Before/after images for each action
│
├── sim_environments/          # Isaac Sim USD scene files
│   ├── gr1_NutPouring.usd
│   ├── gr1_exhaust_pipe.usd
│   └── gr1_default.usd
│
├── assets/                    # 3D models and textures
│   ├── GR-1/                 # Humanoid robot models
│   └── PackingTable/         # Environment props
│
├── .env                      # API keys (OPENAI_API_KEY)
├── requirements_multiagent.txt  # Multi-agent dependencies
└── README.md                 # This file
```

## Usage Examples

### Example 1: Trained Task (GR00T Inference)

```bash
/path/to/isaac-sim/python.sh run_multiagent_sim.py
# Select option 1 or enter exact task:
# "Pick up the red beaker and tilt it to pour out 1 green nut into 
#  yellow bowl. Pick up the yellow bowl and place it on the metallic 
#  measuring scale."
```

**Execution Flow:**
1. Planner recognizes exact match with trained task
2. Strategy: `use_gr00t_inference`
3. Single action: Complete task via GR00T model
4. Supervisor verifies with vision

### Example 2: Novel Task (Action Library Composition)

```bash
/path/to/isaac-sim/python.sh run_multiagent_sim.py
# Select option 2 or enter:
# "Pick up the yellow bowl and place it on the measuring scale"
```

**Execution Flow:**
1. Planner decomposes into 3 subtasks:
   - Subtask 1: Pick up yellow bowl (right hand)
   - Subtask 2: Move hand over measuring scale
   - Subtask 3: Drop bowl onto scale

2. Implementer selects actions:
   - `pick_up_yellow_bowl_1.json`
   - `move_item_in_right_hand_over_to_measuring_scale_1.json`
   - `drop_item_in_right_hand_on_measuring_scale_1.json`

3. Supervisor verifies each subtask with before/after images

### Example 3: Complex Composition

```bash
# "Grab the red beaker and move it to the blue box"
```

**Execution Flow:**
1. Pick red beaker with left hand
2. Switch item to right hand (required for blue box action)
3. Move right hand over blue box
4. Drop item into box

## Configuration Options

### Multi-Agent System (`run_multiagent_sim.py`)

```python
# Line 13-14: Verification mode
VERIFICATION_FREE_PASS = False  # Set True to skip vision verification

# Line 48-52: Model selection
agent_system = MultiAgentRobotSystem(
    model_name="gpt-4o",  # Options: gpt-4o-turbo, gpt-4o, or newer models
    library_dir="./action_library",
    supervisor_free_pass=VERIFICATION_FREE_PASS
)
```

### GR00T Inference Server (`run_inference_server.py`)

```python
MODEL_PATH = "nvidia/GR00T-N1.5-3B"  # Pre-trained
# MODEL_PATH = "./finetuned/your_model"  # Fine-tuned

# Robot configuration
EMBODIMENT_CONFIG = "fourier_gr1_arms_only"  # Arms only
# EMBODIMENT_CONFIG = "fourier_gr1_arms_waist"  # Arms + waist
```

### Fine-tuning (`run_finetune.py`)

```python
# Key parameters:
BATCH_SIZE = 32                    # Adjust based on GPU memory
MAX_STEPS = 30000                  # Training iterations
LEARNING_RATE = 1e-4               # Learning rate
TUNE_PROJECTOR = True              # Recommended
TUNE_DIFFUSION_MODEL = False       # Keep frozen for stability
```

## Action Library (11 Actions)

Current available actions for composition:

**Picking:**
- `pick_red_beaker_1`
- `pick_up_yellow_bowl_1`, `pick_up_yellow_bowl_2`

**Manipulation:**
- `tilt_and_pour_item_in_left_hand`
- `switch_item_in_left_to_right_hand_1`, `switch_item_in_left_to_right_hand_2`

**Movement:**
- `move_item_in_right_hand_over_to_measuring_scale_1`
- `move_item_in_left_hand_over_the_blue_box_1`

**Placing:**
- `drop_item_in_right_hand_on_measuring_scale_1`
- `drop_item_in_left_hand_into_box_1`
- `drop_item_in_right_hand_on_the_table_1`

### Recording New Actions

```bash
# Start Isaac Sim with action recorder
/path/to/isaac-sim/python.sh run_action_recorder.py  # (if available)

# In simulation:
# 1. Press 'R' to start recording
# 2. Perform the action with controller
# 3. Press 'S' to stop and save
# 4. Enter descriptive label (e.g., "pick_blue_cup")
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **OpenAI API Error** | Check `.env` file has valid `OPENAI_API_KEY=sk-...` |
| **GR00T Server Connection Failed** | Ensure `run_inference_server.py` is running: `curl http://localhost:9876/health` |
| **Action Not Found** | List actions: `python -c "from action_library import ActionLibrary; ActionLibrary().list_actions()"` |
| **Vision Verification Always Fails** | Set `VERIFICATION_FREE_PASS=True` or use vision-capable model (gpt-4o-mini or newer) |
| **Model Path Error** | Verify `MODEL_PATH` in `run_inference_server.py` matches fine-tuned directory |
| **Import Error (CrewAI)** | Install: `/path/to/isaac-sim/python.sh -m pip install crewai langchain-openai` |
| **Conda Environment Wrong** | Check: `which python` matches expected environment |

### Debug Mode

```bash
# Test multi-agent system without Isaac Sim
python test_multiagent.py

# Check action library
python -c "from action_library import ActionLibrary; ActionLibrary().list_actions()"

# Test GR00T server
curl -X POST http://localhost:9876/inference \
  -H "Content-Type: application/json" \
  -d '{"task": "test", "obs": [], "state": {}}'
```

## Key Concepts

### Strategy Selection Logic

The Planner agent automatically decides execution strategy:

```python
if task == "exact trained task":
    strategy = "use_gr00t_inference"  # Run complete task via GR00T
else:
    strategy = "use_action_library"   # Compose from recorded actions
```

### State Tracking

Robot state is maintained throughout execution:

```python
{
    "left_hand_holding": "red_beaker",   # or None
    "right_hand_holding": None,           # or "object_name"
    "objects_on_scale": ["yellow_bowl"],  # list of objects
    "last_action": "pick_red_beaker_1"
}
```

### Vision-Based Verification (Optimized Two-Stage Approach)

Supervisor uses a fast two-stage verification process:

1. **Capture Images**: Before/after images saved to `./results/verification_images/`
2. **Stage 1 - Vision Analysis** (Fast): OpenAI Vision API directly analyzes images
   - Uses `ActionLibraryTool.analyze_images_with_openai()`
   - Returns detailed textual description of visual changes
   - Single API call for efficiency
3. **Stage 2 - Decision Making** (Fast): Supervisor agent processes text
   - Reads visual analysis description
   - Applies chain-of-thought reasoning
   - No image processing overhead
4. **Output**: `{success: bool, reasoning: str, confidence: float}`

**Performance**: ~5-15 seconds per verification (vs 40-60s with direct image processing)

## Advanced Features

### Custom Agent Behavior

Edit agent prompts in `multiagent_system.py`:

```python
def _create_planner_agent(self):
    return Agent(
        role="Your Custom Role",
        goal="Your Custom Goal",
        backstory="Your Custom Instructions",
        # ... rest of configuration
    )
```

### Retry Logic Adjustment

Change maximum retry attempts in `multiagent_system.py`:

```python
MAX_RETRIES = 3  # Default, change to 1-5
```

### Video Recording

All executions automatically save video to:
```
./results/multiagent_execution.mp4
```

Configuration in `run_multiagent_sim.py`:
```python
RESULT_VIDEO_FILE = "./results/your_custom_name.mp4"
```

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{groot-multiagent-isaac,
  title={Multi-agent LLM Humanoid Robot Control with Isaac GR00T},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/<your-name>/Multi-agent-LLM-Humanoid-Robot-Control-With-Isaac-Gr00t-Integration-in-Isaac-Simulator}}
}
```

## License

This project uses NVIDIA's GR00T model and Isaac Sim, subject to their respective licenses.

## Contributing

Contributions welcome! Areas for improvement:
- Additional action recordings for larger task coverage
- Enhanced vision-based verification prompts
- Multi-view camera support for better spatial understanding
- Integration with additional LLM providers (Anthropic, etc.)
- Performance optimization for faster planning
- Finetune GR00t with more dataset tasks. Use it as action generator (i.e., populate the action library)

