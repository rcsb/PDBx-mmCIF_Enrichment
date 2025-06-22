# Ligand-Binding Annotation Pipeline

A Python workflow to extract protein–ligand and metal coordination interactions from mmCIF files, generating mmCIF-compatible annotation loops critical for understanding binding mechanisms and accelerating drug discovery.

## Description

This tool processes mmCIF (`.cif`) files using a hybrid algorithm:

1. **Centroid filtering**: Calculate residue centroids and build a KD-tree for protein residues to quickly identify candidates within a user-defined centroid cutoff.
2. **Atom–atom scanning**: Perform detailed atom-level distance checks (with separate cutoffs for metal coordination and non-metal interactions) on those candidates to pinpoint binding pairs.

It then generates mmCIF-compatible annotation loops for integration into downstream archives or databases.

## Dependencies

- Python 3.9 or later
- `gemmi`
- `numpy`
- `scipy`
- `requests`

Install via pip:

```bash
pip install gemmi numpy scipy requests
```

## Setup

1. Clone or download this repository.
2. Ensure the following folders exist in the project root:
   - `current_cif_files/` — place your `.cif` files here
   - `annotation_files/`    — annotations will be written here

   ```bash
   mkdir -p current_cif_files annotation_files
   ```


## Usage

Place all desired mmCIF files into `current_cif_files/`. Then run the annotation script


## Running the Script

```bash
python annotate.py
```

Select Option 2 to scan `current_cif_files/` and process each `.cif` found.

The script will:

1. Read each `.cif` from `current_cif_files/`.
2. Identify protein–ligand and metal coordination interactions.
3. Generate an annotation file named `<PDBID>_annotation.cif` in `annotation_files/`.

## Output

Each output `.cif` contains a mmCIF loop of `_geom_contact` items, for example:

```mmCIF
loop_
_geom_contact.id
_geom_contact.details
_geom_contact.dist
... _geom_contact.metal_coordination
1 'Interaction between HEM and HIS' 2.143  FE Fe 4427 ... yes
```

That's it! Your annotations will now be ready in `annotation_files/` for further use.
