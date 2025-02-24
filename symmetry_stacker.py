from Bio import PDB
import numpy as np
import argparse

# Global axis mapping
AXIS_MAP = {'x': 0, 'y': 1, 'z': 2}


def find_center(structure, chains='', res_num=10):
    """Calculate the center of mass for specified chains and residue number."""
    if not chains:
        chains = [chain.id for chain in structure.get_chains()]
    
    coordinates = []
    for chain_id in chains:
        for chain in structure.get_chains():
            if chain.id == chain_id:
                try:
                    # Get CA atom coordinates of the specified residue number
                    for residue in chain:
                        if residue.id[1] == res_num:  # residue number
                            coordinates.append(residue['CA'].get_coord())
                            break
                except KeyError:
                    print(f"Warning: Could not find CA atom in chain {chain_id} residue {res_num}")
    
    if not coordinates:
        raise ValueError("No valid coordinates found for specified chains and residue number")
    
    return np.mean(coordinates, axis=0)


def find_min_max_on_axis(structure, axis='z'):
    """Find minimum and maximum values along specified axis."""
    axis_idx = AXIS_MAP[axis.lower()]
    
    coordinates = []
    for atom in structure.get_atoms():
        if atom.name == 'CA':  # Only consider alpha carbons
            coordinates.append(atom.get_coord()[axis_idx])
    
    return min(coordinates), max(coordinates)


def do_translation(structure, trans):
    """Apply translation to structure."""
    for atom in structure.get_atoms():
        atom.set_coord(atom.get_coord() + trans)
    return structure


def append_structures(structure1, structure2):
    """Combine two structures with unique chain IDs."""
    # Get existing chain IDs
    existing_chains = set(chain.id for chain in structure1.get_chains())
    available_chains = [c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' 
                       if c not in existing_chains]
    
    # Create new chains in structure1 with new IDs
    model = structure1[0]  # Assuming first model
    chain_id_idx = 0
    
    for chain in structure2.get_chains():
        if chain_id_idx >= len(available_chains):
            raise ValueError("Ran out of available chain IDs")
            
        new_chain = chain.copy()
        new_chain.id = available_chains[chain_id_idx]
        model.add(new_chain)
        chain_id_idx += 1
    
    return structure1


def main(args):
    # Parse chain arguments
    chains1 = args.chains1.split(',') if args.chains1 else ''
    chains2 = args.chains2.split(',') if args.chains2 else ''
    
    # Setup parser
    parser = PDB.PDBParser(QUIET=True)
    
    # Read structures
    structure1 = parser.get_structure('struct1', args.input_pdb1)
    center1 = find_center(structure1, chains1, args.res_num1)
    if args.recenter_on_axis:
        min_val, max_val = find_min_max_on_axis(structure1, args.axis)
        center1[AXIS_MAP[args.axis.lower()]] = (min_val + max_val) / 2
    
    structure2 = parser.get_structure('struct2', args.input_pdb2)
    center2 = find_center(structure2, chains2, args.res_num2)
    if args.recenter_on_axis:
        min_val, max_val = find_min_max_on_axis(structure2, args.axis)
        center2[AXIS_MAP[args.axis.lower()]] = (min_val + max_val) / 2
    
    # Calculate translation vector
    trans_vec = center1 - center2
    axis_idx = AXIS_MAP[args.axis.lower()]
    trans_vec[axis_idx] -= args.separation
    
    # Apply translation and combine structures
    structure2 = do_translation(structure2, trans_vec)
    combined_structure = append_structures(structure1, structure2)
    
    # Save result
    io = PDB.PDBIO()
    io.set_structure(combined_structure)
    io.save(args.output_pdb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stack protein structures with symmetry')
    parser.add_argument("--input_pdb1", type=str, help="First input PDB file")
    parser.add_argument("--chains1", type=str, default='', help="Comma-separated chain IDs for first structure")
    parser.add_argument("--res_num1", type=int, default=10, help="Residue number for first structure center calculation")
    parser.add_argument("--input_pdb2", type=str, help="Second input PDB file")
    parser.add_argument("--chains2", type=str, default='', help="Comma-separated chain IDs for second structure")
    parser.add_argument("--res_num2", type=int, default=10, help="Residue number for second structure center calculation")
    parser.add_argument("--output_pdb", type=str, help="Output PDB file")
    parser.add_argument("--separation", type=float, help="Separation distance along specified axis")
    parser.add_argument("--axis", type=str, default='z', help="Axis for separation (x, y, or z)")
    parser.add_argument("--recenter_on_axis", action='store_true', help="Recenter structures along specified axis")
    args = parser.parse_args()
    
    main(args)