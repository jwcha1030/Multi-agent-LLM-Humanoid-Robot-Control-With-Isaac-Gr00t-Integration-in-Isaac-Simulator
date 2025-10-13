import os
import time
import numpy as np
from scipy.spatial.transform import Rotation
from isaacsim.simulation_app import SimulationApp

# 1. Set simulation (headless mode)
CONFIG = {"headless": True}
simulation_app = SimulationApp(CONFIG)

# 2. Import Sim/Omniverse modules
import omni.usd
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.objects import DynamicCuboid, DynamicSphere
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import add_reference_to_stage

# 3. Initialize Isaac Sim World
# World is a primary class that manage the simulation environment
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()  # add ground

# 4. Add custom objects

# reference frame primitive
ground_prim = XFormPrim(
    prim_path="/World/MyGroup", name="MyGroup", position=[-1.5, 0.0, 0.0]
)

# primitive shapes:
cube_prim = DynamicCuboid(
    prim_path="/World/MyCube",  # hierarchical path in USD stage
    name="MyCube",
    position=np.array([-0.5, 0.5, 1.0]),
    scale=np.array([0.2, 0.2, 0.2]),
    color=np.array([0.8, 0.1, 0.1]),
)
sphere_prim = DynamicSphere(
    prim_path="/World/MySphere",
    name="MySphere",
    position=np.array([-0.5, 0.0, 1.0]),
    radius=0.1,
    color=np.array([0.1, 0.1, 0.8]),
)

# More complex primitives from omniverse assets:
# Table
# table_asset_path = "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1/Props/Mounts/HeavyDutyPackingTable/HeavyDutyPackingTable.usd"
table_asset_path = os.path.abspath(
    "./assets/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/SM_HeavyDutyPackingTable_C02_01_physics.usd"
)
add_reference_to_stage(
    usd_path=table_asset_path, prim_path="/World/MyGroup/HeavyDutyTable"
)

rotation_axis = np.array([0.0, 0.0, 1.0])

table_prim = XFormPrim("/World/MyGroup/HeavyDutyTable")
table_prim.set_local_pose(
    translation=np.array([0.90, 0.0, -0.0678]),
    orientation=Rotation.from_rotvec(np.pi / 2 * rotation_axis).as_quat(
        scalar_first=True
    ),
)

# GR1T2
# robot_asset_path = "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1/Robots/GR1T2/GR1T2_fourier_hand_6dof.usd"
robot_asset_path = os.path.abspath(
    "./assets/GR-1/GR1T2_fourier_hand_6dof/GR1T2_fourier_hand_6dof.usd"
)
robot_prim_path = "/World/GR1T2_Hand"
add_reference_to_stage(usd_path=robot_asset_path, prim_path=robot_prim_path)
robot_prim = XFormPrim(robot_prim_path)
robot_prim.set_local_pose(
    translation=np.array([0.0, 0.0, 0.93988]),
    orientation=Rotation.from_rotvec(np.pi * rotation_axis).as_quat(scalar_first=True),
)
robot = Articulation(
    prim_path=robot_prim_path,
    name="GR1T2_Robot",
)

world.scene.add(
    robot
)  # unlike XFromPrim, DynamicCuboid, etc., it is strongly recommended to use .add method


# 5. Save state
SAVE_PATH = "sim_environments/custom_scene.usd"

# apply the changes in the scene
world.reset()

print(f"Saving the USD stage to: {SAVE_PATH}")
omni.usd.get_context().save_as_stage(SAVE_PATH)

time.sleep(1.0)

simulation_app.close()
print("USD stage created and saved. Isaac Sim closed.")
