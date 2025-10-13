# this uses the gr00t conda environment


import warnings
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.data.schema import EmbodimentTag
from gr00t.data.dataset import LeRobotSingleDataset, LeRobotMixtureDataset
import torch
from gr00t.model.gr00t_n1 import GR00T_N1_5
from transformers import TrainingArguments
from gr00t.experiment.runner import TrainRunner



PRE_TRAINED_MODEL_PATH = "nvidia/GR00T-N1.5-3B"
EMBODIMENT_TAG = EmbodimentTag.GR1
EMBODIMENT_CONFIG = "fourier_gr1_arms_only"
TUNE_LLM = False            
TUNE_VISUAL = False          
TUNE_PROJECTOR = True # whether to tune projector model
TUNE_DIFFUSION_MODEL = False # whether to tune diffusion model
DATASET_PATH = "./datasets/gr1_arms_only.Exhaust_pipe_sort_task"
DATASET_VIDEO_BACKEND = "decord" # torchvision_av #this is important!
MODEL_COMPUTE_DTYPE = "bfloat16"
FINETUNED_OUTPUT_DIRECTORY = "./finetuned/gr1_arms_only.Exhaust_pipe_sort_batch32_nodiffusion"
BATCH_SIZE = 32
MAX_STEPS = 40000
SAVE_STEPS = 10000 # save the model in this steps
GRADIENT_ACCUMULATION_STEPS = 1
RUN_NAME = "gr1_arms_only.Exhaust_pipe_sort_batch32_nodiffusion" # for reporting to wandb
LEARNING_RATE = 1e-4




def main(): 
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    warnings.simplefilter("ignore", category=FutureWarning)
    data_config = DATA_CONFIG_MAP[EMBODIMENT_CONFIG]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()


    train_dataset = LeRobotSingleDataset(
        dataset_path=DATASET_PATH,
        modality_configs=modality_config,
        embodiment_tag=EMBODIMENT_TAG,
        video_backend=DATASET_VIDEO_BACKEND,
        video_backend_kwargs=None,
        transforms=modality_transform, # apply transform in the dataset loader
    )

    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=PRE_TRAINED_MODEL_PATH,
        tune_llm=TUNE_LLM,  # backbone's LLM
        tune_visual=TUNE_VISUAL,  # backbone's vision tower
        tune_projector=TUNE_PROJECTOR,  # action head's projector
        tune_diffusion_model=TUNE_DIFFUSION_MODEL,  # action head's DiT
    )


    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = MODEL_COMPUTE_DTYPE
    model.config.compute_dtype = MODEL_COMPUTE_DTYPE
    model.to(device)



    training_args = TrainingArguments(
        output_dir=FINETUNED_OUTPUT_DIRECTORY,
        overwrite_output_dir=True,
        run_name=RUN_NAME,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        dataloader_num_workers=16,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=True,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=LEARNING_RATE,
        weight_decay=1e-5,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=MAX_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=8,
        report_to="wandb",
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
    )

    experiment.train()
    

if __name__ == "__main__":
    main()