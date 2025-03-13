# -*- coding: utf-8 -*-

'''
This script calculates the Residue-Residue Contact Score (RRCS) between all residues of Protein A (chain A) and Protein B (chain B) in a GROMACS trajectory.
It saves each residue pair's RRCS as an .xvg file and compiles all results into an HDF5 (.h5) file.
Author: Zhenghan Chen
Date: 2025-03-12
'''

import MDAnalysis as mda
import itertools
import numpy as np
import timeit
import os
from tqdm import tqdm
import h5py

def main():
    # =======================
    # Define File Paths
    # =======================
    topology_file = r"C:\Users\Hanoi\Desktop\Bilis\WGCNA\pythonProject\data\Bilis_final\ABPL1_noH.pdb"
    trajectory_file = r"C:\Users\Hanoi\Desktop\Bilis\WGCNA\pythonProject\data\Bilis_final\ABPL1_noH.xtc"

    # =======================
    # Define Selections
    # =======================
    # Protein A: Includes residues classified as 'protein' or with residue name 'PL1' in chain A
    proteinA_selection = "(protein or resname PL1) and chainid A"

    # Protein B: Includes residues classified as 'protein' in chain B
    proteinB_selection = "protein and chainid B"

    # =======================
    # Define RRCS Parameters
    # =======================
    r_min = 3.23  # Å
    r_max = 4.63  # Å
    cutoff = 10.0  # Å for filtering residue pairs
    dt = 100.0  # ps, time interval to sample frames
    bt = 0.0  # ps, start time
    et = 1500000.0  # ps, end time (1500 ns)

    # =======================
    # Define Output Settings
    # =======================
    out_file = 'output_rrcs_A_vs_B'  # Output directory name
    xvg_out = True  # Whether to output .xvg files for each residue pair

    # =======================
    # Initialize Universe
    # =======================
    print("Loading universe...")
    u = mda.Universe(topology_file, trajectory_file)
    print("Universe loaded.")

    # =======================
    # Select Protein A and B Atoms
    # =======================
    print("Selecting Protein A and B...")
    proteinA = u.select_atoms(proteinA_selection)
    proteinB = u.select_atoms(proteinB_selection)

    print(f"Protein A: {len(proteinA.residues)} residues")
    print(f"Protein B: {len(proteinB.residues)} residues")

    # =======================
    # List of Residues
    # =======================
    residuesA = list(proteinA.residues)
    residuesB = list(proteinB.residues)

    # Create lists of residue IDs
    res_idsA = [res.resid for res in residuesA]
    res_idsB = [res.resid for res in residuesB]

    # =======================
    # Create All A-B Residue Pairs
    # =======================
    res_pair = list(itertools.product(res_idsA, res_idsB))
    print(f"Total residue pairs (A-B): {len(res_pair)}")

    # =======================
    # Map Residues to Relative Atom Indices
    # =======================
    print("Mapping residues to relative atom indices...")

    # For Protein A
    res_atomsA = {}
    for res in residuesA:
        res_atomsA[res.resid] = [i for i, atom in enumerate(proteinA.atoms) if atom.resid == res.resid]

    # For Protein B
    res_atomsB = {}
    for res in residuesB:
        res_atomsB[res.resid] = [i for i, atom in enumerate(proteinB.atoms) if atom.resid == res.resid]

    # Verify mapping
    for key, indices in res_atomsA.items():
        if not indices:
            print(f"Warning: No atoms found for residue {key} in Protein A.")
    for key, indices in res_atomsB.items():
        if not indices:
            print(f"Warning: No atoms found for residue {key} in Protein B.")

    # =======================
    # Initialize Data Structures
    # =======================
    contact_score = {}  # {(A_resid, B_resid): [RRCS over time]}

    for a_resid in res_idsA:
        for b_resid in res_idsB:
            contact_score[(a_resid, b_resid)] = []

    # =======================
    # Prepare Output Directory
    # =======================
    if not os.path.exists(out_file):
        os.makedirs(out_file)
        print(f"Created output directory: {out_file}")
    else:
        # Rename existing directory to avoid overwriting
        suffix = 1
        new_dir = f"{out_file}_{suffix}"
        while os.path.exists(new_dir):
            suffix += 1
            new_dir = f"{out_file}_{suffix}"
        os.rename(out_file, new_dir)
        os.makedirs(out_file)
        print(f"Existing output directory renamed to: {new_dir}")
        print(f"Created new output directory: {out_file}")

    # =======================
    # Determine Frames to Process Based on dt
    # =======================
    print("Determining frames to process based on dt...")
    frames_to_process = []
    for ts in u.trajectory:
        current_time = ts.time  # in ps
        if current_time < bt:
            continue
        if current_time > et:
            break
        # Select frames at intervals of dt
        # To select frames at approximately dt intervals, use integer division
        if (current_time - bt) % dt < (dt / 2):
            frames_to_process.append(ts.frame)
    print(f"Total frames to process: {len(frames_to_process)}")

    # =======================
    # Start Trajectory Analysis
    # =======================
    print("Starting trajectory analysis...")
    start_time = timeit.default_timer()

    # Iterate through selected frames
    for frame in tqdm(frames_to_process, desc="Processing frames"):
        u.trajectory[frame]
        current_time = u.trajectory.time  # in ps

        # Compute distance matrix between all atoms in Protein A and Protein B
        # Using MDAnalysis's distance_array function
        # dists shape: (nA_atoms, nB_atoms)
        dists = mda.lib.distances.distance_array(proteinA.positions, proteinB.positions, box=u.dimensions)

        # Iterate through all residue pairs and compute RRCS
        for a_resid, b_resid in res_pair:
            atomsA = res_atomsA[a_resid]
            atomsB = res_atomsB[b_resid]
            if not atomsA or not atomsB:
                # No atoms in one of the residues
                contact_score[(a_resid, b_resid)].append(0.0)
                continue
            # Extract distances for this residue pair using relative indices
            dists_pair = dists[np.ix_(atomsA, atomsB)]
            # Compute RRCS
            # RRCS = sum over atom pairs:
            #   if distance <= r_min: score +=1
            #   elif distance >= r_max: score +=0
            #   else: score += 1 - (distance - r_min)/(r_max - r_min)
            scores = np.where(
                dists_pair <= r_min,
                1.0,
                np.where(
                    dists_pair >= r_max,
                    0.0,
                    1.0 - (dists_pair - r_min) / (r_max - r_min)
                )
            )
            total_score = np.sum(scores)
            contact_score[(a_resid, b_resid)].append(total_score)

    end_time = timeit.default_timer()
    print(f"Trajectory analysis completed in {end_time - start_time:.2f} seconds.")

    # =======================
    # Save Results to HDF5 File
    # =======================
    print("Saving results to HDF5 file...")
    h5_filename = os.path.join(out_file, "rrcs_results_rrcs.h5")
    with h5py.File(h5_filename, 'w') as hf:
        for (a_resid, b_resid), scores in contact_score.items():
            group_name = f"{a_resid}_{b_resid}"
            grp = hf.create_group(group_name)
            # Compute times based on frames_to_process
            times = [u.trajectory[frame].time for frame in frames_to_process]
            grp.create_dataset('times', data=np.array(times))
            grp.create_dataset('rrcs', data=np.array(scores))
    print(f"HDF5 file saved as: {h5_filename}")

    # =======================
    # Save RRCS as xvg Files
    # =======================
    if xvg_out:
        print("Saving xvg files...")
        with h5py.File(h5_filename, 'r') as hf:
            for (a_resid, b_resid) in res_pair:
                group_name = f"{a_resid}_{b_resid}"
                if group_name not in hf:
                    print(f"Group {group_name} not found in HDF5 file. Skipping...")
                    continue
                grp = hf[group_name]
                times = grp['times'][:]
                scores = grp['rrcs'][:]
                # Create xvg file
                xvg_filename = os.path.join(out_file, f"{a_resid}&{b_resid}.xvg")
                with open(xvg_filename, 'w') as f:
                    # Write headers
                    f.write(f"# RRCS between residue {a_resid} and {b_resid}\n")
                    f.write(f"@    title \"RRCS between residue {a_resid} and {b_resid}\"\n")
                    f.write("@    xaxis  label \"Time (ps)\"\n")
                    f.write("@    yaxis  label \"RRCS\"\n")
                    f.write("@TYPE xy\n")
                    for t, s in zip(times, scores):
                        f.write(f"{t:.3f}\t{s:.3f}\n")
        print("xvg files saved.")

    print(f"All results saved in directory: {out_file}")

if __name__ == '__main__':
    main()
