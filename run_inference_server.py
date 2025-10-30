# this uses the gr00t conda environment

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np

import os
import torch
import gr00t
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import DATA_CONFIG_MAP

# Gr00t initialize
MODEL_PATH = "nvidia/GR00T-N1.5-3B"
# MODEL_PATH = "./finetuned/gr1_arms_only.Nut_pouring_batch32_nodiffusion/"
EMBODIMENT_TAG = "gr1"
EMBODIMENT_CONFIG = "fourier_gr1_arms_only"



device = "cuda"
data_config = DATA_CONFIG_MAP[EMBODIMENT_CONFIG]
modality_config = data_config.modality_config()
modality_transform = data_config.transform()
policy = Gr00tPolicy(
    model_path=MODEL_PATH,
    embodiment_tag=EMBODIMENT_TAG,
    modality_config=modality_config,
    modality_transform=modality_transform,
    device=device,
)

# Create a FastAPI app instance
app = FastAPI()

# Define a Pydantic model for the request body
class InferenceRequest(BaseModel):
    task: str
    obs: list
    state: dict


@app.post("/inference")
def run_inference(request: InferenceRequest):
    """
    Accepts a JSON payload and processes it for inference.
    """
    #print(f"Received inference request:")
    #print(f"  task: {request.task}")
    #print(f"  obs: {np.array(request.obs, dtype=np.uint8)}")
    #print(f"  state: {request.state}")


    step_data = {}
    step_data["video.ego_view"] = np.array(request.obs, dtype=np.uint8).reshape((1,256, 256, 3))
    for joint_part_name, joint_state in request.state.items():
        step_data[f"state.{joint_part_name}"] = np.array(joint_state, dtype=float).reshape((1, len(joint_state)))
    step_data["annotation.human.action.task_description"] = [request.task]
    
    print(f"Received Task: {request.task}")
  
    # run the model
    predicted_action = policy.get_action(step_data)
    
    return_data = {}
    for name, value in predicted_action.items():
        return_data[name] = value.tolist()
    return return_data


# Optional: Run the server directly using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=9876)