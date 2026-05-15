#!/usr/bin/env python
"""
Debug script to check what images are in the H5 file for vert2_PHC camera.
"""

import h5py
import sys

if len(sys.argv) < 2:
    print("Usage: python debug_vert2_phc.py <h5_file>")
    sys.exit(1)

h5_file = sys.argv[1]

with h5py.File(h5_file, 'r') as h5file:
    print(f"=== Inspecting {h5_file} ===\n")
    
    # Check for vert1 images
    if 'data/cam_vert1/images' in h5file:
        print("✓ data/cam_vert1/images found")
        group = h5file['data/cam_vert1/images']
        if 'img_names' in group.attrs:
            img_names = [s.decode('utf-8') for s in group.attrs['img_names']]
            print(f"  Image names: {img_names}")
        else:
            print(f"  No img_names attribute")
            print(f"  Available keys: {list(group.keys())}")
    else:
        print("✗ data/cam_vert1/images NOT found")
    
    print()
    
    # Check for vert2 images (absorption)
    if 'data/cam_vert2/images' in h5file:
        print("✓ data/cam_vert2/images found")
        group = h5file['data/cam_vert2/images']
        if 'img_names' in group.attrs:
            img_names = [s.decode('utf-8') for s in group.attrs['img_names']]
            print(f"  Image names: {img_names}")
        else:
            print(f"  No img_names attribute")
            print(f"  Available keys: {list(group.keys())}")
    else:
        print("✗ data/cam_vert2/images NOT found")
    
    print()
    
    # Check alternative paths
    for path in ['data/cam_vert2_PHC/images', 'data/cam_vert2_phc/images', 'data/vert2_PHC/images']:
        if path in h5file:
            print(f"✓ {path} found")
            group = h5file[path]
            if 'img_names' in group.attrs:
                img_names = [s.decode('utf-8') for s in group.attrs['img_names']]
                print(f"  Image names: {img_names}")
            else:
                print(f"  No img_names attribute, available keys: {list(group.keys())}")
        else:
            print(f"✗ {path} NOT found")
    
    print("\n=== Full data structure ===")
    def print_structure(name, obj):
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name} (Dataset: {obj.shape})")
        elif isinstance(obj, h5py.Group):
            print(f"{indent}{name}/ (Group)")
            if 'img_names' in obj.attrs:
                img_names = [s.decode('utf-8') for s in obj.attrs['img_names']]
                print(f"{indent}  -> img_names: {img_names}")
    
    h5file.visititems(print_structure)
