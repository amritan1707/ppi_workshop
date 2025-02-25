import numpy as np
import math
import argparse
from scipy.spatial.transform import Rotation as R
from renumber_pdb import from_pdb_file, to_pdb

def get_sym_vec_no_pyr(pdb_file, order=3, chains=[]):
    """
    Get symmetry vector from a PDB file without using PyRosetta.
    
    Args:
        pdb_file: Path to the PDB file
        order: Symmetry order
        chains: List of chain IDs to consider
        
    Returns:
        sym_vec: Symmetry vector (numpy array)
    """
    protein = from_pdb_file(pdb_file)
    
    if chains == [] or chains == ['']:
        chains = list(protein.chain_id_mapping.keys())
        
    res1_ca = np.stack([protein.atom_positions[:, 1, :][protein.chain_index == i][9] for i in range(order)])
    res2_ca = np.stack([protein.atom_positions[:, 1, :][protein.chain_index == i][19] for i in range(order)])
    
    center1 = np.mean(res1_ca, axis=0)
    center2 = np.mean(res2_ca, axis=0)
    
    sym_vec = center2 - center1
    
    return sym_vec
    
def get_rot_mat_no_pyr(sym_vec, axis='z', invert=False):
    """
    Get rotation matrix to align symmetry vector with the specified axis.
    
    Args:
        sym_vec: Symmetry vector (numpy array)
        axis: Target axis ('x', 'y', or 'z')
        invert: Whether to invert the target axis
        
    Returns:
        rot_mat: Rotation matrix (3x3 numpy array)
    """
    norm_sym_vec = sym_vec / np.linalg.norm(sym_vec)
    axis_vec = np.zeros(3, dtype=np.float32)
    
    if axis == 'z':
        axis_vec[2] = 1.0
    elif axis == 'y':
        axis_vec[1] = 1.0
    else:
        axis_vec[0] = 1.0
        
    if invert:
        axis_vec *= -1
    
    # Handle the case when vectors are already aligned
    if np.allclose(norm_sym_vec, axis_vec) or np.allclose(norm_sym_vec, -axis_vec):
        return np.eye(3)
    
    rotation_axis = np.cross(norm_sym_vec, axis_vec)
    rotation_ang = np.arccos(np.clip(np.dot(norm_sym_vec, axis_vec), -1.0, 1.0))
    
    # Use scipy.spatial.transform.Rotation for more robust rotation
    return R.from_rotvec(rotation_ang * rotation_axis / np.linalg.norm(rotation_axis)).as_matrix()

def do_rotation_no_pyr(atom_positions, rot_mat):
    """
    Apply rotation matrix to atom positions.
    
    Args:
        atom_positions: Atom positions (numpy array)
        rot_mat: Rotation matrix (3x3 numpy array)
        
    Returns:
        rotated_positions: Rotated atom positions (numpy array)
    """
    return atom_positions @ rot_mat.T

def parse_pdb(pdb_file):
    """
    Parse a PDB file to extract chains, residues, and atom coordinates.
    This is a simplified function for when renumber_pdb module is not available.
    
    Args:
        pdb_file: Path to the PDB file
        
    Returns:
        chains: Dictionary mapping chain IDs to residues
        atom_positions: Dictionary mapping (chain_id, res_id, atom_name) to coordinates
    """
    chains = {}
    atom_positions = {}
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                res_id = int(line[22:26])
                chain_id = line[21]
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                if chain_id not in chains:
                    chains[chain_id] = set()
                chains[chain_id].add(res_id)
                
                atom_positions[(chain_id, res_id, atom_name)] = np.array([x, y, z])
    
    return chains, atom_positions

def write_pdb(pdb_file, output_file, rot_mat):
    """
    Read a PDB file, apply a rotation matrix, and write the rotated structure.
    This is a simplified function for when renumber_pdb module is not available.
    
    Args:
        pdb_file: Path to the input PDB file
        output_file: Path to the output PDB file
        rot_mat: Rotation matrix (3x3 numpy array)
    """
    with open(pdb_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                coords = np.array([x, y, z])
                rotated_coords = np.dot(coords, rot_mat.T)
                
                new_line = (line[:30] + 
                           f"{rotated_coords[0]:8.3f}" + 
                           f"{rotated_coords[1]:8.3f}" + 
                           f"{rotated_coords[2]:8.3f}" + 
                           line[54:])
                f_out.write(new_line)
            else:
                f_out.write(line)

def main(args):
    """
    Main function to align a protein structure along a symmetry axis.
    
    Args:
        args: Command line arguments
    """
    # Parse arguments
    args.sym_chains = args.sym_chains.split(',')
    
    try:
        # Try to use the renumber_pdb module if available
        sym_vec = get_sym_vec_no_pyr(args.input_pdb, args.order, chains=args.sym_chains)
        rot_mat = get_rot_mat_no_pyr(sym_vec, args.axis, invert=args.invert)
        
        protein = from_pdb_file(args.input_pdb)
        protein.atom_positions = do_rotation_no_pyr(protein.atom_positions, rot_mat)
        
        pdb_str = to_pdb(protein)
        with open(args.output_pdb, 'w') as f:
            f.write(pdb_str)
    except:
        # Fallback to manual PDB parsing if renumber_pdb module is not available
        print("Warning: renumber_pdb module not available, using basic PDB parsing instead.")
        
        # Get symmetry vector manually
        chains, atom_positions = parse_pdb(args.input_pdb)
        
        if args.sym_chains == ['']:
            chain_ids = list(chains.keys())[:args.order]
        else:
            chain_ids = args.sym_chains[:args.order]
        
        # Extract CA atoms from residues 10 and 20 of each chain
        res1_ca = np.stack([atom_positions[(chain, 10, 'CA')] for chain in chain_ids])
        res2_ca = np.stack([atom_positions[(chain, 20, 'CA')] for chain in chain_ids])
        
        center1 = np.mean(res1_ca, axis=0)
        center2 = np.mean(res2_ca, axis=0)
        
        sym_vec = center2 - center1
        rot_mat = get_rot_mat_no_pyr(sym_vec, args.axis, invert=args.invert)
        
        # Write rotated PDB file
        write_pdb(args.input_pdb, args.output_pdb, rot_mat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align protein structure along symmetry axis without PyRosetta")
    parser.add_argument("--input_pdb", type=str, help="Input PDB file")
    parser.add_argument("--sym_chains", type=str, default='', help="Comma-separated list of chain IDs for symmetry calculation")
    parser.add_argument("--order", type=int, default=3, help="Symmetry order")
    parser.add_argument("--axis", type=str, default="z", help="Target axis (x, y, or z)")
    parser.add_argument("--output_pdb", type=str, help="Output PDB file")
    parser.add_argument("--invert", action="store_true", help="Invert target axis")
    args = parser.parse_args()

    main(args)