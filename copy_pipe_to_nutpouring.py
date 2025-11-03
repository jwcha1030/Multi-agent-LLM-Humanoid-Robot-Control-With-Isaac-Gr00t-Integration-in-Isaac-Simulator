"""
Script to copy the pipe object from gr1_exhaust_pipe.usd to gr1_nutpouring_2.usd
Uses USD Python API to perform the copy operation.
"""

from pxr import Usd, UsdGeom, Sdf
import os

# File paths
SOURCE_USD = "./sim_environments/gr1_exhaust_pipe.usd"
TARGET_USD = "./sim_environments/gr1_nutpouring_pipe.usd"
BACKUP_USD = "./sim_environments/gr1_nutpouring_pipe_backup.usd"

def find_pipe_prim(stage):
    """Find the main pipe object (Xform) in the stage, excluding materials."""
    # We want the main /World/pipe Xform object that contains the geometry
    pipe_prim = stage.GetPrimAtPath("/World/pipe")
    if pipe_prim and pipe_prim.IsValid():
        print(f"Found pipe object: {pipe_prim.GetPath()} (type: {pipe_prim.GetTypeName()})")
        return pipe_prim
    else:
        print("No /World/pipe object found!")
        return None

def copy_prim_to_stage(source_prim, target_stage, target_parent_path="/World"):
    """Copy a prim and all its children to the target stage."""
    # Get the source stage and layer
    source_stage = source_prim.GetStage()
    
    # Create the target prim path
    prim_name = source_prim.GetName()
    target_path = f"{target_parent_path}/{prim_name}"
    
    # Check if target already exists and remove it
    if target_stage.GetPrimAtPath(target_path):
        print(f"Removing existing prim at {target_path}")
        target_stage.RemovePrim(target_path)
    
    # Copy the prim by exporting from source and importing to target
    # This preserves all properties, attributes, and children
    source_layer = source_stage.GetRootLayer()
    target_layer = target_stage.GetRootLayer()
    
    # Use Sdf to copy the prim spec
    source_spec = source_layer.GetPrimAtPath(source_prim.GetPath())
    if source_spec:
        Sdf.CopySpec(source_layer, source_prim.GetPath(), 
                     target_layer, target_path)
        print(f"✓ Successfully copied {source_prim.GetPath()} to {target_path}")
        
        # Print info about what was copied
        copied_prim = target_stage.GetPrimAtPath(target_path)
        children = list(copied_prim.GetChildren())
        print(f"  - Copied {len(children)} child prim(s)")
        for child in children:
            print(f"    • {child.GetName()} ({child.GetTypeName()})")
        
        return target_path
    else:
        print(f"Error: Could not find prim spec for {source_prim.GetPath()}")
        return None

def main():
    print("=" * 60)
    print("USD Pipe Object Copy Tool")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(SOURCE_USD):
        print(f"Error: Source file not found: {SOURCE_USD}")
        return
    if not os.path.exists(TARGET_USD):
        print(f"Error: Target file not found: {TARGET_USD}")
        return
    
    # Create backup of target file
    print(f"\nCreating backup: {BACKUP_USD}")
    import shutil
    shutil.copy2(TARGET_USD, BACKUP_USD)
    
    # Open source stage
    print(f"\nOpening source file: {SOURCE_USD}")
    source_stage = Usd.Stage.Open(SOURCE_USD)
    
    # Find pipe object in source
    print("\nSearching for pipe object in source file...")
    pipe_prim = find_pipe_prim(source_stage)
    
    if not pipe_prim:
        print("No pipe object found in source file!")
        return
    
    # Open target stage
    print(f"\nOpening target file: {TARGET_USD}")
    target_stage = Usd.Stage.Open(TARGET_USD)
    
    # Copy the pipe prim
    print("\nCopying pipe object to target file...")
    result = copy_prim_to_stage(pipe_prim, target_stage)
    if not result:
        print("Failed to copy pipe object!")
        return
    
    # Save the target stage
    print("\nSaving changes to target file...")
    target_stage.Save()
    print("✓ Changes saved successfully!")
    
    print("\n" + "=" * 60)
    print("Copy operation completed!")
    print(f"Original file backed up to: {BACKUP_USD}")
    print("=" * 60)

if __name__ == "__main__":
    main()
