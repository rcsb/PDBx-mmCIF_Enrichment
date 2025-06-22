import json
import os
import gzip
import shutil
import time
import gemmi
import numpy as np
from scipy.spatial import KDTree
import requests

# Configuration
JSON_FILE = "current_file_holdings.json"
BASE_URL = "https://files.wwpdb.org/pub"
CIF_FOLDER = "current_cif_files"
ANNOTATION_FOLDER = "annotation_files"

# Create directories if they don't exist
os.makedirs(CIF_FOLDER, exist_ok=True)
os.makedirs(ANNOTATION_FOLDER, exist_ok=True)

# Global distance parameters
CENTROID_CUTOFF = 20.0  # Increased to ensure no residues are missed
ATOMIC_CUTOFF = 3.5     # Report all atom-atom interactions below this
METAL_CUTOFF = 3.0      # Special cutoff for metal coordination
METAL_CENTROID_CUTOFF = 25.0  # Larger cutoff for metal-coordinating residues

def download_and_extract_cif(pdb_id, cif_url):
    """Download and extract .cif.gz file"""
    try:
        # Download the file
        response = requests.get(cif_url, stream=True)
        response.raise_for_status()
        
        # Save compressed file
        gz_path = os.path.join(CIF_FOLDER, f"{pdb_id}.cif.gz")
        with open(gz_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract the file
        cif_path = os.path.join(CIF_FOLDER, f"{pdb_id}.cif")
        with gzip.open(gz_path, 'rb') as gz_file:
            with open(cif_path, 'wb') as cif_file:
                shutil.copyfileobj(gz_file, cif_file)
        
        return cif_path
    except Exception as e:
        print(f"Error processing {pdb_id}: {e}")
        return None

def get_existing_cif_files():
    """Get a list of existing CIF files in the CIF_FOLDER"""
    cif_files = {}
    for filename in os.listdir(CIF_FOLDER):
        if filename.endswith('.cif'):
            pdb_id = filename.split('.')[0].upper()
            cif_files[pdb_id] = os.path.join(CIF_FOLDER, filename)
    return cif_files

def find_protein_ligand_interactions(cif_file):
    """Find protein-ligand interactions in a structure (only processes first model)"""
    try:
        structure = gemmi.read_structure(cif_file)
    except Exception as e:
        print(f"Error reading mmCIF file: {e}")
        return None, []

    # Standard amino acids and metal ions
    STANDARD_AMINO_ACIDS = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"
    }
    METAL_IONS = {"FE", "ZN", "MG", "CA", "MN", "CU", "NI", "CO"}

    def is_protein_residue(residue):
        return residue.entity_type == gemmi.EntityType.Polymer and residue.name in STANDARD_AMINO_ACIDS

    def is_metal_atom(atom):
        return atom.element.is_metal or atom.element.name in METAL_IONS

    def is_metal_ligand(residue):
        return any(is_metal_atom(atom) for atom in residue)

    def calculate_centroid(residue):
        positions = [atom.pos for atom in residue]
        if not positions:
            return None
        centroid = gemmi.Position(*np.mean([(p.x, p.y, p.z) for p in positions], axis=0))
        return centroid

    def distance(pos1, pos2):
        return pos1.dist(pos2)
    
    start_time = time.time()

    # Extract ligands and protein residues (ONLY FROM FIRST MODEL)
    ligands = []
    protein_residues = []

    # Only process the first model (model[0])
    model = structure[0]
    for chain in model:
        for residue in chain:
            try:
                if residue.entity_type == gemmi.EntityType.NonPolymer:
                    ligands.append((chain.name, residue))
                elif is_protein_residue(residue):
                    protein_residues.append((chain.name, residue))
            except Exception as e:
                print(f"Error processing residue {residue}: {e}")
                continue

    # Rest of the function remains the same...
    # Build KDTree with centroids
    protein_data = []
    valid_protein_indices = []
    ligand_centroids = [calculate_centroid(r) for _, r in ligands if calculate_centroid(r)]
    
    # Use first ligand centroid as reference if available
    ref_centroid = ligand_centroids[0] if ligand_centroids else gemmi.Position(0, 0, 0)
    
    for i, (chain, residue) in enumerate(protein_residues):
        centroid = calculate_centroid(residue)
        if centroid:
            protein_data.append((centroid.x, centroid.y, centroid.z))
            valid_protein_indices.append(i)
    
    if not protein_data:
        return structure, []
        
    protein_kdtree = KDTree(protein_data)

    # Find all interactions
    interactions = []

    for ligand_chain, ligand_residue in ligands:
        ligand_centroid = calculate_centroid(ligand_residue)
        if not ligand_centroid:
            continue
            
        # Use larger cutoff for metal-containing ligands
        current_centroid_cutoff = METAL_CENTROID_CUTOFF if is_metal_ligand(ligand_residue) else CENTROID_CUTOFF

        # Find nearby residues using centroid distances
        nearby_residues = []
        for i, (protein_chain, protein_residue) in enumerate(protein_residues):
            protein_centroid = calculate_centroid(protein_residue)
            if protein_centroid:
                # Calculate centroid distance
                dist = distance(ligand_centroid, protein_centroid)
                if dist <= current_centroid_cutoff:
                    nearby_residues.append((protein_chain, protein_residue))

        # Check all atom pairs with nearby residues
        for protein_chain, protein_residue in nearby_residues:
            for ligand_atom in ligand_residue:
                for protein_atom in protein_residue:
                    dist = distance(ligand_atom.pos, protein_atom.pos)
                    
                    # Check if either atom is a metal
                    has_metal = is_metal_atom(ligand_atom) or is_metal_atom(protein_atom)
                    
                    # Only include interaction if:
                    # 1. It's a metal interaction with distance < METAL_CUTOFF, OR
                    # 2. It's a non-metal interaction with distance < ATOMIC_CUTOFF
                    if (has_metal and dist <= METAL_CUTOFF) or (not has_metal and dist <= ATOMIC_CUTOFF):
                        interactions.append({
                            'ligand_chain': ligand_chain,
                            'ligand_residue': ligand_residue,
                            'ligand_atom': ligand_atom,
                            'protein_chain': protein_chain,
                            'protein_residue': protein_residue,
                            'protein_atom': protein_atom,
                            'distance': dist,
                            'is_metal_coordination': has_metal and dist <= METAL_CUTOFF
                        })

    hybrid_time = time.time() - start_time

    print("\nHYBRID METHOD PERFORMANCE:")
    print(f"Total interactions found: {len(interactions)}")
    print(f"Total time: {hybrid_time:.4f} seconds")
                        
    return structure, interactions

def generate_mmcif_annotation(interactions, output_file):
    """Generate mmCIF format annotation with metal coordination flag"""
    with open(output_file, "w") as f:
        f.write(
            "loop_\n"
            "_geom_contact.id\n"
            "_geom_contact.details\n"
            "_geom_contact.dist\n"
            "_geom_contact.ligand_atom_id\n"
            "_geom_contact.ligand_atom_type\n"
            "_geom_contact.ligand_atom_number\n"
            "_geom_contact.ligand_residue_number\n"
            "_geom_contact.ligand_residue_id\n"
            "_geom_contact.ligand_chain_id\n"
            "_geom_contact.protein_atom_id\n"
            "_geom_contact.protein_atom_type\n"
            "_geom_contact.protein_atom_number\n"
            "_geom_contact.protein_residue_number\n"
            "_geom_contact.protein_residue_id\n"
            "_geom_contact.protein_chain_id\n"
            "_geom_contact.metal_coordination\n"
        )

        for i, interaction in enumerate(interactions, 1):
            try:
                l = interaction['ligand_atom']
                p = interaction['protein_atom']
                line = (
                    f"{i} 'Interaction between {interaction['ligand_residue'].name} "
                    f"and {interaction['protein_residue'].name}' {interaction['distance']:.3f} "
                    f"{l.name} {l.element.name} {l.serial} "
                    f"{interaction['ligand_residue'].seqid.num} {interaction['ligand_residue'].name} "
                    f"{interaction['ligand_chain']} "
                    f"{p.name} {p.element.name} {p.serial} "
                    f"{interaction['protein_residue'].seqid.num} {interaction['protein_residue'].name} "
                    f"{interaction['protein_chain']} "
                    f"{'yes' if interaction['is_metal_coordination'] else 'no'}\n"
                )
                f.write(line)
            except Exception as e:
                print(f"Skipping interaction {i}: {e}")
    print(f"Annotation written to {output_file}")

def process_with_download(json_file):
    """Process with downloading new files"""
    # Load JSON data
    with open(json_file) as f:
        data = json.load(f)
    
    # First pass: Download all CIF files
    print("Starting download of CIF files...")
    downloaded_files = []
    for pdb_id, entry in data.items():
        if "mmcif" in entry and entry["mmcif"]:
            cif_url = BASE_URL + entry["mmcif"][0]
            print(f"Processing {pdb_id}: {cif_url}")
            cif_path = download_and_extract_cif(pdb_id, cif_url)
            if cif_path:
                downloaded_files.append((pdb_id, cif_path))
    
    # Process downloaded files
    process_files(downloaded_files)

def process_with_existing_files():
    """Process using existing CIF files"""
    existing_files = get_existing_cif_files()
    if not existing_files:
        print("No existing CIF files found in the current_cif_files folder.")
        return
    
    print(f"Found {len(existing_files)} existing CIF files:")
    for pdb_id in sorted(existing_files.keys())[:10]:  # Show first 10 as sample
        print(f"- {pdb_id}")
    if len(existing_files) > 10:
        print(f"- ... and {len(existing_files)-10} more")
    
    # Convert to list of (pdb_id, path) tuples
    file_list = [(pdb_id, path) for pdb_id, path in existing_files.items()]
    process_files(file_list)

def process_files(file_list):
    """Process a list of (pdb_id, cif_path) tuples"""
    print("\nStarting annotation processing...")
    for pdb_id, cif_path in file_list:
        print(f"\nProcessing {pdb_id}...")
        output_file = os.path.join(ANNOTATION_FOLDER, f"{pdb_id}_annotation.cif")
        
        structure, interactions = find_protein_ligand_interactions(cif_path)
        
        if structure:
            # Print summary
            metal_interactions = sum(1 for i in interactions if i['is_metal_coordination'])
            print(f"\nFound {len(interactions)} total interactions:")
            print(f"- {metal_interactions} metal coordinations (<{METAL_CUTOFF}A)")
            print(f"- {len(interactions)-metal_interactions} other interactions (<{ATOMIC_CUTOFF}A)")
            
            # Generate mmCIF output
            generate_mmcif_annotation(interactions, output_file)
            
            # Print 5 shortest interactions
            print("\nShortest interactions:")
            for i in sorted(interactions, key=lambda x: x['distance'])[:5]:
                print(
                    f"{i['ligand_atom'].name}-{i['protein_atom'].name}: "
                    f"{i['distance']:.2f}A "
                    f"{'(METAL)' if i['is_metal_coordination'] else ''}"
                )
        else:
            print(f"Failed to process {pdb_id}")

if __name__ == "__main__":
    print("Choose processing option:")
    print("1. Download new CIF files and process them")
    print("2. Process existing CIF files in current_cif_files folder")
    
    while True:
        choice = input("Enter your choice (1 or 2): ")
        if choice == '1':
            process_with_download(JSON_FILE)
            break
        elif choice == '2':
            process_with_existing_files()
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")