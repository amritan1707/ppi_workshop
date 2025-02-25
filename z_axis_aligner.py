import numpy as np
from Bio import PDB
import argparse
import math
import os
import copy

def find_center(model, chains=[]):
    """Calculate the center of mass for specified chains using CA atoms."""
    if not chains:
        chains = [chain.id for chain in model.get_chains()]
    
    ca_atoms = []
    for chain in model.get_chains():
        if chain.id in chains:
            for residue in chain:
                if "CA" in residue:
                    ca_atoms.append(residue["CA"].get_coord())
    
    if not ca_atoms:
        raise ValueError("No CA atoms found in specified chains")
    
    # Calculate center of mass
    center = np.mean(ca_atoms, axis=0)
    return center

def get_rotation_matrix(center, align_to='z', invert=False):
    """Calculate rotation matrix to align center vector to specified axis."""
    # Normalize center vector
    center_norm = center / np.linalg.norm(center)
    
    # Create target axis vector
    align_axis = np.zeros(3)
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    align_axis[axis_map[align_to]] = 1
    
    # Create rotation axis (cross product of vectors)
    rotation_axis = np.cross(center_norm, align_axis)
    if np.all(np.abs(rotation_axis) < 1e-10):
        return np.eye(3)
    
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    # Calculate rotation angle
    cos_angle = np.dot(center_norm, align_axis)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    if invert:
        angle = -angle
    
    # Rodriguez rotation formula
    K = np.array([
        [0, -rotation_axis[2], rotation_axis[1]],
        [rotation_axis[2], 0, -rotation_axis[0]],
        [-rotation_axis[1], rotation_axis[0], 0]
    ])
    rotation_matrix = (np.eye(3) + np.sin(angle) * K + 
                      (1 - np.cos(angle)) * np.matmul(K, K))
    
    return rotation_matrix

def apply_rotation(structure, rotation_matrix):
    """Apply rotation matrix to all atoms in structure."""
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coord = atom.get_coord()
                    new_coord = np.dot(rotation_matrix, coord)
                    atom.set_coord(new_coord)

def check_file_exists(filepath):
    """Check if a file exists and is readable."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    if not os.path.isfile(filepath):
        raise ValueError(f"Not a file: {filepath}")
    if not os.access(filepath, os.R_OK):
        raise PermissionError(f"File is not readable: {filepath}")

def main(args):
    # Validate input file
    try:
        check_file_exists(args.input_pdb)
    except Exception as e:
        print(f"Error with input file: {e}")
        return

    # Validate output directory exists
    output_dir = os.path.dirname(args.output_pdb)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            print(f"Error creating output directory: {e}")
            return

    # Parse chain list
    sym_chains = args.sym_chains.split(',')
    
    try:
        # Load structure
        parser = PDB.PDBParser(QUIET=True)
        with open(args.input_pdb, 'r') as f:
            structure = parser.get_structure('protein', f)
        
        # Create a deep copy of the structure to prevent modifications to original
        structure_copy = copy.deepcopy(structure)
        
        # Validate chains exist in structure
        available_chains = {chain.id for chain in structure_copy[0].get_chains()}
        missing_chains = set(sym_chains) - available_chains
        if missing_chains:
            raise ValueError(f"Chains not found in structure: {', '.join(missing_chains)}")

        # Get center of symmetric subunit from first model
        center = find_center(structure_copy[0], sym_chains)
        
        # Calculate rotation matrix to align with z-axis
        rot_mat = get_rotation_matrix(center)
        apply_rotation(structure_copy, rot_mat)
        
        # Check alignment and rotate in opposite direction if needed
        new_center = find_center(structure_copy[0], sym_chains)
        new_center_xy = new_center.copy()
        new_center_xy[2] = 0
        
        # If the projection onto xy-plane is too large, try inverting the rotation
        if np.linalg.norm(new_center_xy) > 0.1 * np.linalg.norm(new_center):
            # Reset to original coordinates
            structure_copy = copy.deepcopy(structure)
            # Apply inverted rotation
            rot_mat = get_rotation_matrix(center, invert=True)
            apply_rotation(structure_copy, rot_mat)
        
        # Save structure
        io = PDB.PDBIO()
        io.set_structure(structure_copy)
        
        # Ensure we preserve all PDB records
        io.save(args.output_pdb, preserve_atom_numbering=True)
        
        print(f"Successfully aligned structure to z-axis and saved to {args.output_pdb}")
        
    except Exception as e:
        print(f"Error processing structure: {e}")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align PDB structure symmetric axis to z-axis')
    parser.add_argument('--input_pdb', type=str, required=True, help='Input PDB file')
    parser.add_argument('--sym_chains', type=str, required=True, help='Comma-separated list of chain IDs')
    parser.add_argument('--output_pdb', type=str, required=True, help='Output PDB file')
    args = parser.parse_args()
    
    main(args)