#!/usr/bin/env python3

import sys
import Bio.PDB as PDB
import numpy as np
from scipy.spatial import KDTree
import pandas as pd
from typing import Tuple, List, Dict

class InterfaceAnalyzer:
    def __init__(self, pdb_file: str, chain1: str, chain2: str):
        """
        Initialize analyzer with PDB file and chains defining the interface
        
        Args:
            pdb_file: Path to PDB file
            chain1: First chain ID
            chain2: Second chain ID
        """
        self.parser = PDB.PDBParser(QUIET=True)
        self.structure = self.parser.get_structure('protein', pdb_file)
        self.chain1 = self.structure[0][chain1]
        self.chain2 = self.structure[0][chain2]
        
        # Define constants
        self.CONTACT_CUTOFF = 5.0  # Angstroms
        self.BURIAL_CUTOFF = 0.2   # Fraction of surface area
        
    def find_interface_residues(self) -> Tuple[List[str], List[str]]:
        """
        Identify residues at the interface using distance-based criterion
        
        Returns:
            Two lists of residue IDs at interface from each chain
        """
        # Get coordinates of all atoms
        coords1 = np.array([atom.coord for atom in self.chain1.get_atoms()])
        coords2 = np.array([atom.coord for atom in self.chain2.get_atoms()])
        
        # Build KD tree for efficient neighbor search
        tree = KDTree(coords2)
        
        # Find contacts
        interface_res1 = set()
        interface_res2 = set()
        
        for atom in self.chain1.get_atoms():
            neighbors = tree.query_ball_point(atom.coord, self.CONTACT_CUTOFF)
            if neighbors:
                interface_res1.add(str(atom.get_parent().id[1]))
                for idx in neighbors:
                    interface_res2.add(str(list(self.chain2.get_atoms())[idx].get_parent().id[1]))
                    
        return list(interface_res1), list(interface_res2)
        
    def analyze_interface_composition(self, interface_res1: List[str], 
                                   interface_res2: List[str]) -> pd.DataFrame:
        """
        Analyze amino acid composition and properties of interface residues
        
        Args:
            interface_res1: List of interface residues from chain 1
            interface_res2: List of interface residues from chain 2
            
        Returns:
            DataFrame with interface composition statistics
        """
        # Define residue properties
        aa_properties = {
            'polar': ['SER', 'THR', 'ASN', 'GLN'],
            'charged': ['LYS', 'ARG', 'HIS', 'ASP', 'GLU'],
            'hydrophobic': ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'TYR']
        }
        
        stats = {
            'chain': [],
            'total_residues': [],
            'polar_count': [],
            'charged_count': [],
            'hydrophobic_count': []
        }
        
        # Analyze each chain
        for chain_id, interface_res in [('1', interface_res1), ('2', interface_res2)]:
            chain = self.chain1 if chain_id == '1' else self.chain2
            
            polar_count = 0
            charged_count = 0
            hydrophobic_count = 0
            
            for res_id in interface_res:
                res_name = chain[int(res_id)].get_resname()
                if res_name in aa_properties['polar']:
                    polar_count += 1
                elif res_name in aa_properties['charged']:
                    charged_count += 1
                elif res_name in aa_properties['hydrophobic']:
                    hydrophobic_count += 1
                    
            stats['chain'].append(chain_id)
            stats['total_residues'].append(len(interface_res))
            stats['polar_count'].append(polar_count)
            stats['charged_count'].append(charged_count)
            stats['hydrophobic_count'].append(hydrophobic_count)
            
        return pd.DataFrame(stats)

def main():
    """Example usage of InterfaceAnalyzer"""
    # Initialize analyzer with PDB file

    filename = sys.argv[1]
    chain1 = sys.argv[2] 
    chain2 = sys.argv[3]
    analyzer = InterfaceAnalyzer(filename, chain1, chain2)
    
    # Find interface residues
    interface_res1, interface_res2 = analyzer.find_interface_residues()
    print(f"Found {len(interface_res1)} interface residues in chain 1:", interface_res1)
    print(f"Found {len(interface_res2)} interface residues in chain 2:", interface_res2)
    
    # Analyze interface composition
    composition_stats = analyzer.analyze_interface_composition(interface_res1, interface_res2)
    print("\nInterface composition:")
    print(composition_stats)

if __name__ == "__main__":
    main()