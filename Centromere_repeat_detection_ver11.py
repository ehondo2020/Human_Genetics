#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import pysam
import scipy.stats as stats
from matplotlib.colors import FuncNorm, LinearSegmentedColormap

# Configuration parameters and file paths
DIR_FASTA36 = "./FASTA36_NC_021248_239976_279815_human_variants_Ancestor"
DIR_GENOME_FASTA = "./Human_T2Ts/Dustmasked_ref/header_replaced"
MONOMER_FASTA = "./centromere_pericentromere_satellites.fa"

OUTPUT_BED_WITH_FLANKING = "./NC_021248_239976_279815_human_variants_Ancestor_with_flanking.bed"
OUTPUT_BED_NO_FLANKING = "./NC_021248_239976_279815_human_variants_Ancestor_no_flanking.bed"
OUTPUT_BED_SOURCE = OUTPUT_BED_WITH_FLANKING

# BED generation parameters
FLANKING_BP = 50000
NUM_THREADS = 48

# Plotting parameters
SOURCE_FASTA = "./NC_021248_239976_279815_human_variants_Ancestor.fa"
OUTPUT_SUMMARY_PNG = "./NC_021248_239976_279815_EVE_Population_Consensus_Profile.png"
OUTPUT_SUPPLEMENTAL_PNG = "./NC_021248_239976_279815_EVE_Supplemental_Panels_A_B_C.png"

K_LIST = [18, 17, 16, 15]
SUPPLEMENTAL_K = 16

# Sliding window settings for main profile (bp)
WINDOW_SIZE = 500
STEP_SIZE = 100

# Sliding window settings for supplemental validation plot (bp)
SUPP_WINDOW_SIZE = 50
SUPP_STEP_SIZE = 10

# Supplemental visualization and Wilcoxon test boundaries (bp)
SUPP_5P_FLANK_BP = 20000
SUPP_5P_EVE_BP = 10000
SUPP_3P_EVE_BP = 10000
SUPP_3P_FLANK_BP = 20000
JUNCTION_TEST_BP = 2000

# Supplemental heatmap color scaling
HEATMAP_VMIN = 0.0
HEATMAP_VMAX = 1.0
HEATMAP_LOW_RANGE_MAX = 0.1
HEATMAP_LOW_RANGE_FRACTION = 0.5

HEATMAP_COLORBAR_TICKS = [
    0.000,
    0.020,
    0.100,
    0.250,
    0.500,
    1.000
]

PLOT_FLANKING_BP = 50000
MISMATCH_ALLOWANCE = 0

Y_MIN = None
Y_MAX = None

# Global dictionary for repeat k-mer counts
GLOBAL_REPEAT_KMER_COUNTS = {}


def bio_reverse_complement(seq):
    """Return biological reverse complement of a sequence in uppercase."""
    seq_upper = seq.upper()
    mapping = str.maketrans("ATCGN", "TAGCN")
    return seq_upper.translate(mapping)[::-1]


def canonical_kmer(seq):
    """Return canonical k-mer using lexicographical ordering."""
    seq_upper = seq.upper()
    rc = bio_reverse_complement(seq_upper)
    return seq_upper if seq_upper <= rc else rc


def generate_mismatch_neighbors(seq, max_mismatch):
    """Generate all k-mer neighbors within maximum allowed mismatches."""
    seq = seq.upper()

    if max_mismatch == 0:
        return {canonical_kmer(seq)}

    bases = ("A", "C", "G", "T")
    neighbors = set()

    def dfs(current, start_pos, remaining):
        if remaining == 0:
            neighbors.add(canonical_kmer("".join(current)))
            return

        for i in range(start_pos, len(current)):
            original = current[i]

            for b in bases:
                if b == original:
                    continue

                current[i] = b
                dfs(current, i + 1, remaining - 1)

            current[i] = original

    neighbors.add(canonical_kmer(seq))
    chars = list(seq)

    for m in range(1, max_mismatch + 1):
        dfs(chars, 0, m)

    return neighbors


def extract_fasta36_info(filepath):
    """Extract chromosome name and coordinates from fasta36 output."""
    chrom, coords_str = None, None
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if chrom is None and line.startswith(">>"):
                match_chrom = re.match(r'^>>(\w+)', line.strip())
                if match_chrom:
                    chrom = match_chrom.group(1)
            if coords_str is None and "banded Smith-Waterman score:" in line:
                match_coords = re.search(r'\([^)]+:([\d-]+)\)', line)
                if match_coords:
                    coords_str = match_coords.group(1)
            if chrom is not None and coords_str is not None:
                break
    return chrom, coords_str


def generate_bed_mode():
    """Generate BED files with and without flanking regions."""
    print(f"\n[+] Generating BED files: with flanking ({FLANKING_BP} bp) and without flanking (0 bp)")
    if not os.path.exists(DIR_FASTA36):
        print(f"[-] Error: Directory not found -> {DIR_FASTA36}")
        sys.exit(1)
        
    all_entries = os.listdir(DIR_FASTA36)
    fa_files = [f for f in all_entries if f.endswith(".fa")]

    files = []
    for f in all_entries:
        if f.endswith(".txt") and not f.startswith("."):
            prefix = f.split("_")[0]
            if any(prefix in fa for fa in fa_files):
                files.append(f)

    print(f" -> Processing {len(files)} files...")
    
    bed_with_flanking = []
    bed_no_flanking = []
    
    for f in files:
        filepath = os.path.join(DIR_FASTA36, f)
        chrom, coords_str = extract_fasta36_info(filepath)
        if chrom and coords_str:
            try:
                start_str, end_str = coords_str.split("-")
                start_val = int(start_str) - 1
                end_val = int(end_str)
                sample_id = f.replace(".txt", "")
                
                start_flank = max(0, start_val - FLANKING_BP)
                end_flank = end_val + FLANKING_BP
                bed_with_flanking.append(f"{chrom}\t{start_flank}\t{end_flank}\t{sample_id}")
                
                bed_no_flanking.append(f"{chrom}\t{start_val}\t{end_val}\t{sample_id}")
            except ValueError:
                pass
                
    with open(OUTPUT_BED_WITH_FLANKING, 'w') as out:
        out.write("\n".join(bed_with_flanking) + "\n")
        
    with open(OUTPUT_BED_NO_FLANKING, 'w') as out:
        out.write("\n".join(bed_no_flanking) + "\n")
        
    print(f"[✓] BED output complete:")
    print(f"  - With flanking: {OUTPUT_BED_WITH_FLANKING} ({len(bed_with_flanking)} records)")
    print(f"  - Without flanking: {OUTPUT_BED_NO_FLANKING} ({len(bed_no_flanking)} records)")


def load_single_fasta(fasta_path):
    """Load first sequence from FASTA file."""
    seq = []
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq: break
            else:
                seq.append(line.upper())
    return "".join(seq)


def load_all_monomers(monomer_fasta):
    """Load all monomer sequences from monomer FASTA file."""
    monomers = []
    current_seq = []

    with open(monomer_fasta, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith(">"):
                if current_seq:
                    monomers.append("".join(current_seq).upper())
                    current_seq = []
            else:
                current_seq.append(line)

        if current_seq:
            monomers.append("".join(current_seq).upper())

    return monomers


def scan_kmer_density(target_seq, monomer_fasta, k_list):
    """Scan target sequence for k-mer occurrences in repeat database."""
    seq_len = len(target_seq)
    kmer_pos_dict = {k: {} for k in k_list}
    is_repeat = {k: np.zeros(seq_len, dtype=int) for k in k_list}
    
    print(" -> Building canonical k-mer dictionary for target sequence...")
    for k in k_list:
        for i in range(seq_len - k + 1):

            kmer = target_seq[i:i+k].upper()

            if "N" in kmer:
                continue

            ckmer = canonical_kmer(kmer)

            if ckmer not in kmer_pos_dict[k]:
                kmer_pos_dict[k][ckmer] = []

            kmer_pos_dict[k][ckmer].append(i)

    print(f" -> Streaming repeat database (K={k_list})...")
    with open(monomer_fasta, 'r') as f:
        current_seq = []
        for line in f:
            if line.startswith(">"):
                if current_seq:
                    seq_str = "".join(current_seq).upper()
                    for k in k_list:
                        for i in range(len(seq_str) - k + 1):
                            m_kmer = seq_str[i:i+k].upper()

                            if "N" in m_kmer:
                                continue

                            ckmer = canonical_kmer(m_kmer)

                            if ckmer in kmer_pos_dict[k]:
                                for pos in kmer_pos_dict[k][ckmer]:
                                    is_repeat[k][pos:pos+k] = 1
                current_seq = []
            else:
                current_seq.append(line.strip().upper())
                
        if current_seq:
            seq_str = "".join(current_seq).upper()
            for k in k_list:
                for i in range(len(seq_str) - k + 1):
                    m_kmer = seq_str[i:i+k].upper()

                    if "N" in m_kmer:
                        continue

                    ckmer = canonical_kmer(m_kmer)

                    if ckmer in kmer_pos_dict[k]:
                        for pos in kmer_pos_dict[k][ckmer]:
                            is_repeat[k][pos:pos+k] = 1
                            
        return is_repeat


def _worker_analyze_sample(args):
    """Parallel worker calculating k-mer density for a sample target sequence."""
    idx, sample_id, target_seq, k_list, main_x_coords, main_w, supp_x_coords, supp_w = args
    
    seq_len = len(target_seq)
    
    if seq_len == 0:
        res_main = {k: np.zeros(len(main_x_coords)) for k in k_list}
        res_supp = {k: np.zeros(len(supp_x_coords)) for k in k_list}
        return idx, res_main, res_supp
        
    kmer_scores = {k: np.zeros(seq_len, dtype=np.float32) for k in k_list}
    
    for k in k_list:
        k_counts = GLOBAL_REPEAT_KMER_COUNTS.get(k, {})
        if not k_counts:
            continue

        for i in range(seq_len - k + 1):
            t_kmer = target_seq[i:i+k].upper()
            if "N" in t_kmer:
                continue

            matched_weight = None
            c_t_kmer = canonical_kmer(t_kmer)

            if c_t_kmer in k_counts:
                matched_weight = 1.0 / np.log1p(k_counts[c_t_kmer])
            elif MISMATCH_ALLOWANCE > 0:
                t_rc = bio_reverse_complement(t_kmer)
                tier_buckets = {m: [] for m in range(1, MISMATCH_ALLOWANCE + 1)}
                all_neighbors = generate_mismatch_neighbors(t_kmer, MISMATCH_ALLOWANCE)
                
                for neighbor in all_neighbors:
                    if neighbor == c_t_kmer:
                        continue
                    d1 = sum(c1 != c2 for c1, c2 in zip(t_kmer, neighbor))
                    d2 = sum(c1 != c2 for c1, c2 in zip(t_rc, neighbor))
                    dist = min(d1, d2)
                    if 1 <= dist <= MISMATCH_ALLOWANCE:
                        tier_buckets[dist].append(neighbor)
                
                for m in range(1, MISMATCH_ALLOWANCE + 1):
                    tier_weights = []
                    for neighbor in tier_buckets[m]:
                        if neighbor in k_counts:
                            tier_weights.append(1.0 / np.log1p(k_counts[neighbor]))
                    if tier_weights:
                        matched_weight = max(tier_weights)
                        break

            if matched_weight is not None:
                kmer_scores[k][i:i+k] += matched_weight
                
    res_main = {k: np.zeros(len(main_x_coords)) for k in k_list}
    res_supp = {k: np.zeros(len(supp_x_coords)) for k in k_list}

    for k in k_list:
        csum = np.concatenate(([0], np.cumsum(kmer_scores[k])))
        
        raw_main = np.zeros(len(main_x_coords))
        for w_idx, w_start in enumerate(main_x_coords):
            if w_start + main_w > seq_len: continue
            raw_main[w_idx] = (csum[w_start + main_w] - csum[w_start]) / main_w
        res_main[k] = raw_main

        raw_supp = np.zeros(len(supp_x_coords))
        for w_idx, w_start in enumerate(supp_x_coords):
            if w_start + supp_w > seq_len: continue
            raw_supp[w_idx] = (csum[w_start + supp_w] - csum[w_start]) / supp_w
        res_supp[k] = raw_supp
            
    return idx, res_main, res_supp


def fetch_target_seq_from_genome(dir_genome, sample_id, chrom, start_val, end_val, flanking_bp):
    """Fetch target sequence using pysam indexing."""
    fasta_file = None
    for f in os.listdir(dir_genome):
        if sample_id in f and (f.endswith(".fa") or f.endswith(".fasta")):
            fasta_file = f
            break
            
    if not fasta_file:
        return ""
        
    filepath = os.path.join(dir_genome, fasta_file)
    
    try:
        with pysam.FastaFile(filepath) as fa:
            if chrom not in fa.references:
                return ""
            
            chrom_len = fa.get_reference_length(chrom)
            target_start = max(0, start_val - flanking_bp)
            target_end = min(chrom_len, end_val + flanking_bp)
            
            return fa.fetch(chrom, target_start, target_end).upper()
    except Exception as e:
        print(f"    [!] Error retrieving pysam region ({sample_id}): {e}")
        return ""


def plot_profile_mode():
    """Plot population consensus k-mer profiles and supplemental panels."""
    print("\n[+] Starting population consensus profile plotting mode")

    if not os.path.exists(MONOMER_FASTA):
        print(f"[-] Error: Centromere satellite FASTA not found -> {MONOMER_FASTA}")
        sys.exit(1)
    if not os.path.exists(OUTPUT_BED_NO_FLANKING):
        print(f"[-] Error: BED file not found -> {OUTPUT_BED_NO_FLANKING}")
        sys.exit(1)

    DIR_GENOME_FASTA = "./Human_T2Ts/Dustmasked_ref/header_replaced"
    if not os.path.exists(DIR_GENOME_FASTA):
        print(f"[-] Error: Individual genome FASTA directory not found -> {DIR_GENOME_FASTA}")
        sys.exit(1)

    print(f" [i] Loading mapping coordinates for all samples from {OUTPUT_BED_NO_FLANKING}...")
    bed_records = []
    with open(OUTPUT_BED_NO_FLANKING, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("\t")
            if len(parts) == 4:
                chrom, start_str, end_str, sample_id = parts
                bed_records.append((chrom, int(start_str), int(end_str), sample_id))

    total_samples = len(bed_records)
    print(f"    -> Found {total_samples} sample records.")
    
    print(" [i] Loading monomer FASTA into memory...")
    monomer_list = load_all_monomers(MONOMER_FASTA)
    print(f"    -> Loaded {len(monomer_list)} monomer sequences.")

    if total_samples == 0:
        print("[-] Error: No data found for analysis.")
        sys.exit(1)

    print(" [i] Extracting target regions with flanking sequences from genome FASTAs...")
    tasks = []
    valid_idx = 0
    base_seq_len = None
    
    for chrom, start_val, end_val, sample_id in bed_records:
        target_seq = fetch_target_seq_from_genome(DIR_GENOME_FASTA, sample_id, chrom, start_val, end_val, FLANKING_BP)
        if not target_seq:
            print(f"    [!] Warning: FASTA or chromosome {chrom} not found for sample {sample_id}. Skipping.")
            continue
            
        if base_seq_len is None:
            base_seq_len = len(target_seq)
            
        local_main_x = list(range(0, len(target_seq) - WINDOW_SIZE + 1, STEP_SIZE))
        local_supp_x = list(range(0, len(target_seq) - SUPP_WINDOW_SIZE + 1, SUPP_STEP_SIZE))
        tasks.append((valid_idx, sample_id, target_seq, K_LIST, local_main_x, WINDOW_SIZE, local_supp_x, SUPP_WINDOW_SIZE))
        valid_idx += 1

    actual_processed_samples = len(tasks)
    print(f"    -> Running parallel scan for {actual_processed_samples} successfully extracted samples.")

    base_x_coords = list(range(0, base_seq_len - WINDOW_SIZE + 1, STEP_SIZE))
    num_windows = len(base_x_coords)

    population_data = {k: np.zeros((actual_processed_samples, num_windows)) for k in K_LIST}

    print(f" [i] Pre-building canonical k-mer frequency database from {MONOMER_FASTA}...")
    global GLOBAL_REPEAT_KMER_COUNTS
    GLOBAL_REPEAT_KMER_COUNTS = {k: {} for k in K_LIST}

    for seq_str in monomer_list:
        m_len = len(seq_str)
        for k in K_LIST:
            for i in range(m_len - k + 1):
                m_kmer = seq_str[i:i+k].upper()
                if "N" in m_kmer: continue
                ckmer = canonical_kmer(m_kmer)
                GLOBAL_REPEAT_KMER_COUNTS[k][ckmer] = (
                    GLOBAL_REPEAT_KMER_COUNTS[k].get(ckmer, 0) + 1
                )

    print(f" [i] Calculating k-mer repetition density using {NUM_THREADS} processes...")
    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(_worker_analyze_sample, tasks))
        
    num_left_w = (FLANKING_BP - WINDOW_SIZE) // STEP_SIZE + 1
    num_right_w = (FLANKING_BP - WINDOW_SIZE) // STEP_SIZE + 1
    base_mid_w = num_windows - num_left_w - num_right_w

    num_left_w_supp = (FLANKING_BP - SUPP_WINDOW_SIZE) // SUPP_STEP_SIZE + 1
    num_right_w_supp = (FLANKING_BP - SUPP_WINDOW_SIZE) // SUPP_STEP_SIZE + 1

    target_supp_k = SUPPLEMENTAL_K if SUPPLEMENTAL_K in K_LIST else K_LIST[0]

    boundary_left_profiles = []
    boundary_right_profiles = []

    off_5p_start = -SUPP_5P_FLANK_BP // SUPP_STEP_SIZE
    off_5p_end   = SUPP_5P_EVE_BP // SUPP_STEP_SIZE
    off_3p_start = -SUPP_3P_EVE_BP // SUPP_STEP_SIZE
    off_3p_end   = SUPP_3P_FLANK_BP // SUPP_STEP_SIZE

    for idx, res_main, res_supp in results:
        single_supp_res = res_supp[target_supp_k]
        local_n_supp = len(single_supp_res)
        
        j_5p = num_left_w_supp - 1
        s_5p = j_5p + off_5p_start
        e_5p = j_5p + off_5p_end
        if s_5p >= 0 and e_5p <= local_n_supp:
            boundary_left_profiles.append(single_supp_res[s_5p:e_5p])
            
        j_3p = local_n_supp - num_right_w_supp
        s_3p = j_3p + off_3p_start
        e_3p = j_3p + off_3p_end
        if s_3p >= 0 and e_3p <= local_n_supp:
            boundary_right_profiles.append(single_supp_res[s_3p:e_3p])

        local_n_main = len(res_main[target_supp_k])
        for k in K_LIST:
            single_res = res_main[k]
            left_part = single_res[:num_left_w]
            right_part = single_res[local_n_main - num_right_w:]
            mid_part = single_res[num_left_w : local_n_main - num_right_w]
            
            if len(mid_part) == base_mid_w:
                mid_interpolated = mid_part
            else:
                xp = np.linspace(0, 1, len(mid_part))
                x_new = np.linspace(0, 1, base_mid_w)
                mid_interpolated = np.interp(x_new, xp, mid_part)
            
            population_data[k][idx, :] = np.concatenate([left_part, mid_interpolated, right_part])

    mat_5p = np.vstack(boundary_left_profiles)
    mat_3p = np.vstack(boundary_right_profiles)

    n_5p_samples = mat_5p.shape[0]
    n_3p_samples = mat_3p.shape[0]

    print("5' Non-zero ratio:", np.count_nonzero(mat_5p) / mat_5p.size)
    print("5' Percentiles (50, 75, 90, 95, 99):", np.percentile(mat_5p, [50, 75, 90, 95, 99]))
    print("3' Non-zero ratio:", np.count_nonzero(mat_3p) / mat_3p.size)
    print("3' Percentiles (50, 75, 90, 95, 99):", np.percentile(mat_3p, [50, 75, 90, 95, 99]))

    print(f" [i] Generating supplemental figure (5' & 3' 4 panels, K={target_supp_k}, with Wilcoxon test)...")
    
    supp_center_offset = SUPP_WINDOW_SIZE / 2.0
    x_5p_ab = np.arange(off_5p_start, off_5p_end) * SUPP_STEP_SIZE - supp_center_offset
    x_3p_ab = np.arange(off_3p_start, off_3p_end) * SUPP_STEP_SIZE + supp_center_offset

    if JUNCTION_TEST_BP <= 0:
        raise ValueError("JUNCTION_TEST_BP must be > 0.")

    if JUNCTION_TEST_BP > min(
        SUPP_5P_FLANK_BP,
        SUPP_5P_EVE_BP,
        SUPP_3P_EVE_BP,
        SUPP_3P_FLANK_BP
    ):
        raise ValueError(
            "JUNCTION_TEST_BP must not exceed the displayed "
            "flanking/EVE range."
        )

    test_half_window = SUPP_WINDOW_SIZE / 2.0

    mask_flank_5p = (
        (x_5p_ab >= -(JUNCTION_TEST_BP - test_half_window))
        & (x_5p_ab < 0)
    )

    mask_eve_5p = (
        (x_5p_ab > 0)
        & (x_5p_ab <= (JUNCTION_TEST_BP - test_half_window))
    )

    flank_5p_vals = np.mean(
        mat_5p[:, mask_flank_5p],
        axis=1
    )

    eve_5p_vals = np.mean(
        mat_5p[:, mask_eve_5p],
        axis=1
    )

    if np.sum(mask_flank_5p) != np.sum(mask_eve_5p):
        raise RuntimeError(
            "5' junction test regions contain unequal numbers of windows."
        )

    try:
        _, p_5p = stats.wilcoxon(
            flank_5p_vals,
            eve_5p_vals,
            alternative="two-sided"
        )
        p_5p_str = (
            f"p={p_5p:.2e}"
            if p_5p >= 1e-300
            else "p < 1e-300"
        )
    except Exception:
        p_5p_str = "p=N/A"

    mask_eve_3p = (
        (x_3p_ab >= -(JUNCTION_TEST_BP - test_half_window))
        & (x_3p_ab < 0)
    )

    mask_flank_3p = (
        (x_3p_ab > 0)
        & (x_3p_ab <= (JUNCTION_TEST_BP - test_half_window))
    )

    eve_3p_vals = np.mean(
        mat_3p[:, mask_eve_3p],
        axis=1
    )

    flank_3p_vals = np.mean(
        mat_3p[:, mask_flank_3p],
        axis=1
    )

    if np.sum(mask_eve_3p) != np.sum(mask_flank_3p):
        raise RuntimeError(
            "3' junction test regions contain unequal numbers of windows."
        )

    try:
        _, p_3p = stats.wilcoxon(
            eve_3p_vals,
            flank_3p_vals,
            alternative="two-sided"
        )
        p_3p_str = (
            f"p={p_3p:.2e}"
            if p_3p >= 1e-300
            else "p < 1e-300"
        )
    except Exception:
        p_3p_str = "p=N/A"

    print(
        f"  [Wilcoxon Test] 5' Junction: "
        f"Flanking (-{JUNCTION_TEST_BP} to 0 bp) vs "
        f"EVE (0 to +{JUNCTION_TEST_BP} bp): {p_5p_str}"
    )

    print(
        f"  [Wilcoxon Test] 3' Junction: "
        f"EVE (-{JUNCTION_TEST_BP} to 0 bp) vs "
        f"Flanking (0 to +{JUNCTION_TEST_BP} bp): {p_3p_str}"
    )

    fig_supp, axes = plt.subplots(2, 2, figsize=(18, 10))

    mean_5p = np.mean(mat_5p, axis=0)
    ste_5p = np.std(mat_5p, axis=0) / np.sqrt(n_5p_samples)
    mean_3p = np.mean(mat_3p, axis=0)
    ste_3p = np.std(mat_3p, axis=0) / np.sqrt(n_3p_samples)

    y_min_supp = min(np.min(mean_5p - 1.96 * ste_5p), np.min(mean_3p - 1.96 * ste_3p))
    y_max_supp = max(np.max(mean_5p + 1.96 * ste_5p), np.max(mean_3p + 1.96 * ste_3p))
    y_margin = (y_max_supp - y_min_supp) * 0.08 if y_max_supp != y_min_supp else 0.1

    supp_ymin = Y_MIN if Y_MIN is not None else min(0.0, y_min_supp - y_margin)
    supp_ymax = Y_MAX if Y_MAX is not None else y_max_supp + y_margin
    supp_ylim = (supp_ymin, supp_ymax)

    heatmap_vmin = HEATMAP_VMIN
    heatmap_vmax = HEATMAP_VMAX

    if heatmap_vmax <= heatmap_vmin:
        raise ValueError(
            "HEATMAP_VMAX must be greater than HEATMAP_VMIN."
        )

    if not (
        heatmap_vmin
        < HEATMAP_LOW_RANGE_MAX
        < heatmap_vmax
    ):
        raise ValueError(
            "HEATMAP_LOW_RANGE_MAX must lie between "
            "HEATMAP_VMIN and HEATMAP_VMAX."
        )

    if not (
        0.0 < HEATMAP_LOW_RANGE_FRACTION < 1.0
    ):
        raise ValueError(
            "HEATMAP_LOW_RANGE_FRACTION must be between 0 and 1."
        )

    axes[0, 0].plot(x_5p_ab, mean_5p, color="crimson", linewidth=2, label=f"K={target_supp_k} Raw Mean [{p_5p_str}]")
    axes[0, 0].fill_between(x_5p_ab, mean_5p - 1.96 * ste_5p, mean_5p + 1.96 * ste_5p, color="crimson", alpha=0.2)
    axes[0, 0].axvline(0, color="black", linestyle="--", linewidth=1.5, label="EVE 5' Junction (0 bp)")
    axes[0, 0].axvspan(-SUPP_5P_FLANK_BP, 0, color="gray", alpha=0.15, label="Flanking Satellite")
    axes[0, 0].axvspan(0, SUPP_5P_EVE_BP, color="lightblue", alpha=0.3, label=f"EVE Region (+{SUPP_5P_EVE_BP}bp)")
    axes[0, 0].set_title(f"Panel A1: 5' Boundary Uninterpolated Profile (K={target_supp_k}, N={n_5p_samples})", fontsize=11, weight='bold')
    axes[0, 0].set_xlabel("Distance from 5' Junction (bp)", fontsize=9)
    axes[0, 0].set_ylabel("Inverse-frequency-weighted Satellite k-mer Score", fontsize=9)
    axes[0, 0].set_ylim(supp_ylim)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[0, 0].grid(True, linestyle="--", alpha=0.4)

    custom_bgr_cmap = LinearSegmentedColormap.from_list(
        "expanded_low_range_bgr",
        [
            "#3B82F6",
            "#00FF00",
            "#FFFF00",
            "#FF8C00",
            "#FF0000"
        ]
    )
    custom_bgr_cmap.set_over("#FF0000")

    def _forward_heatmap_norm(x):
        x = np.asarray(x, dtype=float)
        y = np.empty_like(x)

        low_mask = x <= HEATMAP_LOW_RANGE_MAX
        high_mask = ~low_mask

        y[low_mask] = (
            (x[low_mask] - HEATMAP_VMIN)
            / (HEATMAP_LOW_RANGE_MAX - HEATMAP_VMIN)
            * HEATMAP_LOW_RANGE_FRACTION
        )

        y[high_mask] = (
            HEATMAP_LOW_RANGE_FRACTION
            +
            (x[high_mask] - HEATMAP_LOW_RANGE_MAX)
            / (HEATMAP_VMAX - HEATMAP_LOW_RANGE_MAX)
            * (1.0 - HEATMAP_LOW_RANGE_FRACTION)
        )

        return y

    def _inverse_heatmap_norm(y):
        y = np.asarray(y, dtype=float)
        x = np.empty_like(y)

        low_mask = y <= HEATMAP_LOW_RANGE_FRACTION
        high_mask = ~low_mask

        x[low_mask] = (
            HEATMAP_VMIN
            +
            y[low_mask]
            / HEATMAP_LOW_RANGE_FRACTION
            * (HEATMAP_LOW_RANGE_MAX - HEATMAP_VMIN)
        )

        x[high_mask] = (
            HEATMAP_LOW_RANGE_MAX
            +
            (y[high_mask] - HEATMAP_LOW_RANGE_FRACTION)
            / (1.0 - HEATMAP_LOW_RANGE_FRACTION)
            * (HEATMAP_VMAX - HEATMAP_LOW_RANGE_MAX)
        )

        return x

    shared_heatmap_norm = FuncNorm(
        (_forward_heatmap_norm, _inverse_heatmap_norm),
        vmin=heatmap_vmin,
        vmax=heatmap_vmax
    )

    x_b1 = np.tile(x_5p_ab, n_5p_samples)
    y_b1 = np.repeat(
        np.arange(1, n_5p_samples + 1),
        len(x_5p_ab)
    )
    z_b1 = mat_5p.ravel()

    axes[1, 0].scatter(
        x_b1,
        y_b1,
        s=18,
        marker="o",
        c="white",
        edgecolors="none",
        linewidths=0,
        alpha=1.0,
        zorder=2
    )

    im_5p = axes[1, 0].scatter(
        x_b1,
        y_b1,
        s=7,
        marker="o",
        c=z_b1,
        cmap=custom_bgr_cmap,
        norm=shared_heatmap_norm,
        edgecolors="none",
        linewidths=0,
        alpha=1.0,
        zorder=3,
        rasterized=True
    )

    axes[1, 0].axvline(
        0,
        color="white",
        linestyle="--",
        linewidth=1.0,
        label="5' Junction",
        zorder=4
    )

    axes[1, 0].set_xlim(
        x_5p_ab[0],
        x_5p_ab[-1]
    )

    axes[1, 0].set_ylim(
        0.5,
        n_5p_samples + 0.5
    )

    axes[1, 0].set_title(
        f"Panel B1: 5' Boundary 2D Dot Plot "
        f"(K={target_supp_k}, N={n_5p_samples})",
        fontsize=11,
        weight='bold'
    )

    axes[1, 0].set_xlabel(
        "Distance from 5' Junction (bp)",
        fontsize=9
    )

    axes[1, 0].set_ylabel(
        "Individual Sample ID",
        fontsize=9
    )

    axes[1, 0].axvline(
        0,
        color="white",
        linestyle="--",
        linewidth=1.0,
        label="5' Junction"
    )

    axes[1, 0].set_title(
        f"Panel B1: 5' Boundary 2D Heatmap "
        f"(K={target_supp_k}, N={n_5p_samples})",
        fontsize=11,
        weight='bold'
    )

    axes[1, 0].set_xlabel(
        "Distance from 5' Junction (bp)",
        fontsize=9
    )

    axes[1, 0].set_ylabel(
        "Individual Sample ID",
        fontsize=9
    )

    axes[0, 1].plot(x_3p_ab, mean_3p, color="crimson", linewidth=2, label=f"K={target_supp_k} Raw Mean [{p_3p_str}]")
    axes[0, 1].fill_between(x_3p_ab, mean_3p - 1.96 * ste_3p, mean_3p + 1.96 * ste_3p, color="crimson", alpha=0.2)
    axes[0, 1].axvline(0, color="black", linestyle="--", linewidth=1.5, label="EVE 3' Junction (0 bp)")
    axes[0, 1].axvspan(0, SUPP_3P_FLANK_BP, color="gray", alpha=0.15, label="Flanking Satellite")
    axes[0, 1].axvspan(-SUPP_3P_EVE_BP, 0, color="lightblue", alpha=0.3, label=f"EVE Region (-{SUPP_3P_EVE_BP}bp)")
    axes[0, 1].set_title(f"Panel A2: 3' Boundary Uninterpolated Profile (K={target_supp_k}, N={n_3p_samples})", fontsize=11, weight='bold')
    axes[0, 1].set_xlabel("Distance from 3' Junction (bp)", fontsize=9)
    axes[0, 1].set_ylabel("Inverse-frequency-weighted Satellite k-mer Score", fontsize=9)
    axes[0, 1].set_ylim(supp_ylim)
    axes[0, 1].legend(loc="upper right", fontsize=8)
    axes[0, 1].grid(True, linestyle="--", alpha=0.4)

    x_b2 = np.tile(x_3p_ab, n_3p_samples)
    y_b2 = np.repeat(
        np.arange(1, n_3p_samples + 1),
        len(x_3p_ab)
    )
    z_b2 = mat_3p.ravel()

    axes[1, 1].scatter(
        x_b2,
        y_b2,
        s=18,
        marker="o",
        c="white",
        edgecolors="none",
        linewidths=0,
        alpha=1.0,
        zorder=2
    )

    im_3p = axes[1, 1].scatter(
        x_b2,
        y_b2,
        s=7,
        marker="o",
        c=z_b2,
        cmap=custom_bgr_cmap,
        norm=shared_heatmap_norm,
        edgecolors="none",
        linewidths=0,
        alpha=1.0,
        zorder=3,
        rasterized=True
    )

    axes[1, 1].axvline(
        0,
        color="white",
        linestyle="--",
        linewidth=1.0,
        label="3' Junction",
        zorder=4
    )

    axes[1, 1].set_xlim(
        x_3p_ab[0],
        x_3p_ab[-1]
    )

    axes[1, 1].set_ylim(
        0.5,
        n_3p_samples + 0.5
    )

    axes[1, 1].set_title(
        f"Panel B2: 3' Boundary 2D Dot Plot "
        f"(K={target_supp_k}, N={n_3p_samples})",
        fontsize=11,
        weight='bold'
    )

    axes[1, 1].set_xlabel(
        "Distance from 3' Junction (bp)",
        fontsize=9
    )

    axes[1, 1].set_ylabel(
        "Individual Sample ID",
        fontsize=9
    )

    axes[1, 1].axvline(
        0,
        color="white",
        linestyle="--",
        linewidth=1.0,
        label="3' Junction"
    )

    axes[1, 1].set_title(
        f"Panel B2: 3' Boundary 2D Heatmap "
        f"(K={target_supp_k}, N={n_3p_samples})",
        fontsize=11,
        weight='bold'
    )

    axes[1, 1].set_xlabel(
        "Distance from 3' Junction (bp)",
        fontsize=9
    )

    axes[1, 1].set_ylabel(
        "Individual Sample ID",
        fontsize=9
    )

    cbar_b1 = fig_supp.colorbar(
        im_5p,
        ax=axes[1, 0],
        pad=0.02,
        fraction=0.046
    )
    cbar_b1.set_label(
        "Inverse-frequency-weighted Satellite k-mer Score",
        fontsize=9
    )
    cbar_b1.set_ticks(HEATMAP_COLORBAR_TICKS)
    cbar_b1.ax.tick_params(labelsize=8)

    cbar_b2 = fig_supp.colorbar(
        im_3p,
        ax=axes[1, 1],
        pad=0.02,
        fraction=0.046
    )
    cbar_b2.set_label(
        "Inverse-frequency-weighted Satellite k-mer Score",
        fontsize=9
    )
    cbar_b2.set_ticks(HEATMAP_COLORBAR_TICKS)
    cbar_b2.ax.tick_params(labelsize=8)

    fig_supp.tight_layout()
    fig_supp.savefig(OUTPUT_SUPPLEMENTAL_PNG, dpi=300)
    plt.close(fig_supp)
    print(f"  [✓] Saved supplemental figure: {OUTPUT_SUPPLEMENTAL_PNG}")

    print(" [i] Calculating population summary statistics and generating main figure...")
    plt.figure(figsize=(12, 6.5))

    DISTINCT_30_COLORS = [
        "crimson", "firebrick", "darkorange", "gold", "forestgreen", 
        "royalblue", "mediumblue", "purple", "darkorchid", "deeppink",
        "teal", "olive", "chocolate", "saddlebrown", "cadetblue",
        "darkcyan", "indianred", "yellowgreen", "dodgerblue", "magenta",
        "darkkhaki", "peru", "slateblue", "mediumvioletred", "seagreen",
        "darkgoldenrod", "lightseagreen", "coppertone", "midnightblue", "black"
    ]

    print("\n[Statistical Test: Flanking vs EVE Region (Wilcoxon Signed-Rank Test)]")
    for rank, k in enumerate(K_LIST):
        matrix = population_data[k]
        
        flank_means = (np.mean(matrix[:, :num_left_w], axis=1) + np.mean(matrix[:, -num_right_w:], axis=1)) / 2.0
        eve_means = np.mean(matrix[:, num_left_w : matrix.shape[1] - num_right_w], axis=1)
        
        try:
            stat, p_val = stats.wilcoxon(flank_means, eve_means)
            p_str = f"p={p_val:.2e}" if p_val >= 1e-300 else "p < 1e-300"
        except Exception:
            p_val = 1.0
            p_str = "p=N/A"
            
        role_str = "Primary" if k == SUPPLEMENTAL_K else "Sensitivity"
        print(f"  K={k} [{role_str}]: Flank_Mean={np.mean(flank_means):.2f}, EVE_Mean={np.mean(eve_means):.2f} -> {p_str}")

        mean_profile = np.mean(matrix, axis=0)
        std_err = np.std(matrix, axis=0) / np.sqrt(actual_processed_samples)
        
        color_for_k = DISTINCT_30_COLORS[rank % len(DISTINCT_30_COLORS)]
        
        base_label = f"K={k} ({role_str})"
        label_for_k = f"{base_label} [{p_str}]"
        
        plt.plot(base_x_coords, mean_profile, color=color_for_k, label=label_for_k, linewidth=2.5, alpha=0.9)
        plt.fill_between(base_x_coords, mean_profile - 1.96 * std_err, mean_profile + 1.96 * std_err, 
                         color=color_for_k, alpha=0.15)

    plt.axvspan(0, PLOT_FLANKING_BP, color='gray', alpha=0.15, label='Flanking Satellite (Consensus)')
    plt.axvspan(base_seq_len - PLOT_FLANKING_BP, base_seq_len, color='gray', alpha=0.15)
    
    all_means = [np.mean(population_data[k], axis=0) for k in K_LIST]
    all_stds = [np.std(population_data[k], axis=0) / np.sqrt(actual_processed_samples) for k in K_LIST]
    data_min = min([np.min(m - 1.96 * s) for m, s in zip(all_means, all_stds)])
    data_max = max([np.max(m + 1.96 * s) for m, s in zip(all_means, all_stds)])
    margin = (data_max - data_min) * 0.1 if data_max != data_min else 0.1

    actual_ymin = Y_MIN if Y_MIN is not None else min(0.0, data_min - margin)
    actual_ymax = Y_MAX if Y_MAX is not None else data_max + margin * 2.0

    text_y_pos = actual_ymin + (actual_ymax - actual_ymin) * 0.03
    plt.text(base_seq_len / 2, text_y_pos, f'EVE Insert Region\n(Population Cohort: N={actual_processed_samples})', 
             fontsize=11, color='dimgray', ha='center', va='bottom', alpha=0.6, weight='bold')

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=9, framealpha=0.9, borderaxespad=0)

    plt.title(f"Population-level Inverse-frequency-weighted Satellite k-mer Profile across EVE Boundary", 
              fontsize=15, weight='bold', pad=50)
    plt.xlabel("Standardized Genomic Position along EVE-Flanking Locus (bp)", fontsize=11)
    plt.ylabel("Inverse-frequency-weighted Satellite k-mer Score", fontsize=11)
    
    plt.ylim(actual_ymin, actual_ymax)
    plt.xlim(0, base_seq_len)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    plt.savefig(OUTPUT_SUMMARY_PNG, dpi=300)
    plt.close()
    print(f"\n[✓] Analysis complete. Saved profile figure: {OUTPUT_SUMMARY_PNG}")


if __name__ == "__main__":
    generate_bed_mode()
    plot_profile_mode()