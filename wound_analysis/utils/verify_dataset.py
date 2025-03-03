#!/usr/bin/env python3
"""
Verification script for dataset access in Docker container.
Run this script inside the container to verify dataset access.
"""

import os
import sys
from pathlib import Path

def verify_dataset_access():
    """Verify access to dataset files and directories"""
    
    print("=" * 50)
    print("DATASET VERIFICATION SCRIPT")
    print("=" * 50)
    
    # 1. Check current directory and environment
    print("\nCURRENT DIRECTORY:", os.getcwd())
    print("PYTHONPATH:", os.environ.get('PYTHONPATH', 'Not set'))
    
    # 2. Check dataset directory
    dataset_path = Path('/app/dataset')
    print(f"\nDATASET PATH: {dataset_path}")
    print(f"EXISTS: {dataset_path.exists()}")
    
    if dataset_path.exists():
        print("\nDATASET DIRECTORY CONTENTS:")
        for item in dataset_path.iterdir():
            print(f"  - {item.name} {'(DIR)' if item.is_dir() else f'(FILE, {item.stat().st_size} bytes)'}")
    
    # 3. Check impedance directory
    impedance_path = dataset_path / 'impedance_frequency_sweep'
    print(f"\nIMPEDANCE PATH: {impedance_path}")
    print(f"EXISTS: {impedance_path.exists()}")
    
    if impedance_path.exists():
        print("\nIMPEDANCE DIRECTORY CONTENTS:")
        for item in impedance_path.iterdir():
            print(f"  - {item.name} {'(DIR)' if item.is_dir() else f'(FILE, {item.stat().st_size} bytes)'}")
    
    # 4. Check main CSV file
    csv_path = dataset_path / 'SmartBandage-Data_for_llm.csv'
    print(f"\nMAIN CSV PATH: {csv_path}")
    print(f"EXISTS: {csv_path.exists()}")
    
    if csv_path.exists():
        print(f"SIZE: {csv_path.stat().st_size} bytes")
        print(f"READABLE: {os.access(csv_path, os.R_OK)}")
    
    # 5. Check permissions
    print("\nPERMISSIONS:")
    if dataset_path.exists():
        print(f"Dataset dir: {oct(dataset_path.stat().st_mode)[-3:]}")
    if impedance_path.exists():
        print(f"Impedance dir: {oct(impedance_path.stat().st_mode)[-3:]}")
    if csv_path.exists():
        print(f"CSV file: {oct(csv_path.stat().st_mode)[-3:]}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    verify_dataset_access()
