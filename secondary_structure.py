import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import os, sys
from typing import Dict, List, Tuple

class SecondaryStructureFilter:
    """Filter protein structures based on secondary structure composition using Biotite."""
    
    def calculate_ss_percentages(self, pdb_file: str) -> Dict[str, float]:
        """
        Calculate percentages of secondary structure elements using Biotite.
        
        Args:
            pdb_file (str): Path to PDB file
            
        Returns:
            Dict[str, float]: Dictionary with percentages for each secondary structure type
        """
        try:
            # Load PDB file
            pdb_file = pdb.PDBFile.read(pdb_file)
            
            # Get structure - only use first model if multiple models exist
            structure = pdb_file.get_structure(model=1)
            
            # Select only protein atoms
            mask = struc.filter_amino_acids(structure)
            structure = structure[mask]
            
            if len(structure) == 0:
                print(f"No protein atoms found in {pdb_file}")
                return None
                
            try:
                # Calculate secondary structure
                sse = struc.annotate_sse(structure)
                
                # Count secondary structure elements
                total_residues = len(sse)
                if total_residues == 0:
                    print(f"No residues found for secondary structure calculation in {pdb_file}")
                    return None
                    
                ss_counts = {
                    'helix': sum(1 for ss in sse if ss == 'a'),  # alpha helix
                    'sheet': sum(1 for ss in sse if ss == 'b'),  # beta sheet
                    'loop': sum(1 for ss in sse if ss == 'c')    # coil
                }
                
                # Calculate percentages
                ss_percentages = {
                    ss_type: (count / total_residues) * 100 
                    for ss_type, count in ss_counts.items()
                }
                
                return ss_percentages
                
            except IndexError as e:
                print(f"Error calculating secondary structure for {pdb_file}: {str(e)}")
                print("This might be due to missing backbone atoms or unusual residue numbering")
                return None
                
        except Exception as e:
            print(f"Error processing {pdb_file}: {str(e)}")
            return None
    
    def filter_structures(self, 
                         pdb_dir: str,
                         min_helix: float = 0,
                         max_helix: float = 100,
                         min_sheet: float = 0,
                         max_sheet: float = 100,
                         min_loop: float = 0,
                         max_loop: float = 100) -> List[Tuple[str, Dict[str, float]]]:
        """
        Filter PDB structures based on secondary structure composition criteria.
        
        Args:
            pdb_dir (str): Directory containing PDB files
            min_helix (float): Minimum percentage of helical content
            max_helix (float): Maximum percentage of helical content
            min_sheet (float): Minimum percentage of beta sheet content
            max_sheet (float): Maximum percentage of beta sheet content
            min_loop (float): Minimum percentage of loop content
            max_loop (float): Maximum percentage of loop content
            
        Returns:
            List[Tuple[str, Dict[str, float]]]: List of (pdb_file, ss_percentages) that meet criteria
        """
        filtered_structures = []
        
        for filename in os.listdir(pdb_dir):
            if filename.endswith('.pdb'):
                pdb_path = os.path.join(pdb_dir, filename)
                ss_percentages = self.calculate_ss_percentages(pdb_path)
                
                if ss_percentages is None:
                    continue
                    
                # Check if structure meets all criteria
                meets_criteria = (
                    min_helix <= ss_percentages['helix'] <= max_helix and
                    min_sheet <= ss_percentages['sheet'] <= max_sheet and
                    min_loop <= ss_percentages['loop'] <= max_loop
                )
                
                if meets_criteria:
                    filtered_structures.append((filename, ss_percentages))
        
        return filtered_structures

# Example usage
if __name__ == "__main__":
    # Initialize the filter
    ss_filter = SecondaryStructureFilter()

    if len(sys.argv) < 4:
        raise Exception("Usage: python secondary_structure.py <filename> <min_helix_percent> <min_sheet_percent>")
    filename = sys.argv[1]
    min_helix_percent = int(sys.argv[2])
    min_sheet_percent = int(sys.argv[3])
    
    # Example: Filter structures with 30-50% helix content and at least 20% sheet content
    results = ss_filter.filter_structures(
        pdb_dir=filename,
        min_helix=min_helix_percent,
        min_sheet=min_sheet_percent
    )
    
    # Print results
    for pdb_file, ss_percentages in results:
        print(f"\nStructure: {pdb_file}")
        print("Secondary Structure Composition:")
        for ss_type, percentage in ss_percentages.items():
            print(f"{ss_type}: {percentage:.1f}%")