#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from Bio import SeqIO
from itertools import product
from scipy.spatial.distance import cosine
import pandas as pd
from scipy.stats import spearmanr
import datetime
from matplotlib.lines import Line2D

# =========================================================
# 1. k-mer Calculation
# =========================================================

def generate_kmers(k=5):
    return [''.join(p) for p in product('ACGT', repeat=k)]

def kmer_freq(seq, k=5):
    seq = seq.upper()
    kmers = generate_kmers(k)
    d = dict.fromkeys(kmers, 0)

    total = len(seq) - k + 1
    if total <= 0:
        return np.zeros(len(kmers))

    for i in range(total):
        kmer = seq[i:i+k]
        if kmer in d:
            d[kmer] += 1

    vec = np.array(list(d.values()), dtype=float)
    s = vec.sum()
    if s > 0:
        vec = vec / s
    return vec

# =========================================================
# 2. Entropy Calculation
# =========================================================

def entropy(vec, k=5):
    p = vec / np.sum(vec)
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return H / np.log(4**k)

# =========================================================
# 3. ECI Calculation
# =========================================================

def ECI(vec, viral_vec, k=5):
    H = entropy(vec, k)
    D = cosine(vec, viral_vec)
    return (1 - H) * (1 - D), H, D

# =========================================================
# 4. FASTA Parsing
# =========================================================

def load_fasta(path):
    data = {}
    for r in SeqIO.parse(path, "fasta"):
        data[r.id] = str(r.seq)
    return data

# =========================================================
# 5. GC Content Calculation
# =========================================================

def calculate_gc(seq):
    seq = seq.upper()
    gc = seq.count('G') + seq.count('C')
    return gc / len(seq)

# =========================================================
# 6. MAIN Pipeline
# =========================================================

def main():

    human_file = "AF063866_6_39989_human_variants_Ancestor.fa"
    viral_file = "AF063866_6_39989_for_species_comparison.fa"

    print("[1] Loading FASTA sequence files")
    human = load_fasta(human_file)
    viral = load_fasta(viral_file)

    print("[2] Computing k-mer frequency vectors")

    human_vecs = []
    labels = []

    for sid, seq in human.items():
        human_vecs.append(kmer_freq(seq))
        labels.append(sid)

    human_vecs = np.array(human_vecs)

    viral_vec = np.mean([kmer_freq(s) for s in viral.values()], axis=0)

    print("[3] Computing ECI values")

    ECIs = []
    Hs = []
    Ds = []

    for v in human_vecs:
        eci, h, d = ECI(v, viral_vec)
        ECIs.append(eci)
        Hs.append(h)
        Ds.append(d)

    ECIs = np.array(ECIs)
    Hs = np.array(Hs)
    Ds = np.array(Ds)
    
    gc_contents = [calculate_gc(seq) for seq in human.values()]

    chromosomes = []
    for sid in labels:
        parts = sid.split('_')
        chr_name = "Unknown"
        for p in parts:
            if p.startswith("chr"):
                chr_name = p
                break
        chromosomes.append(chr_name)

    df = pd.DataFrame({
        "ECI": ECIs,
        "Entropy": Hs,
        "ViralDist": Ds,
        "GC": gc_contents,
        "ID": labels,
        "Chromosome": chromosomes
    })
    
    corr, p = spearmanr(df["GC"], df["ECI"])
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    msg = f"[{timestamp}] | Spearman rho: {corr:.4f}, p-value: {p:.3e}"

    print(msg)
    with open("analysis_stats.txt", "a") as f:
        f.write(msg + "\n")

    # =========================================================
    # FIGURE 1: ECI distribution
    # =========================================================

    plt.figure(figsize=(8,5))
    plt.hist(ECIs, bins=30, color="black", alpha=0.7)
    plt.xlabel("ECI")
    plt.ylabel("Count")
    plt.title("EVE Constraint Index distribution")
    plt.tight_layout()
    plt.savefig("FIG1_ECI_distribution.png", dpi=300)
    plt.close()

    # =========================================================
    # FIGURE 2: entropy vs viral divergence (Separate Files)
    # =========================================================

    TOP_LAYER_CHROM = "chr3"

    def chr_sort_key(chrom_str):
        name = chrom_str.replace("chr", "")
        if name.isdigit():
            return (0, int(name))
        elif name in ["X", "Y", "Z", "W"]:
            return (1, name)
        else:
            return (2, name)

    fig2_xlim = (0.645, 0.725)
    fig2_ylim = (0.330, 0.445)
    fig2_clim = (0.1825, 0.2050)

    # ---------------------------------------------------------
    # 2-1. ECI Color Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(6, 6))
    
    df_normal = df[df["Chromosome"] != TOP_LAYER_CHROM]
    plt.scatter(df_normal["Entropy"], df_normal["ViralDist"], c=df_normal["ECI"], cmap="viridis", s=40, vmin=min(df["ECI"]), vmax=max(df["ECI"]))
    
    df_top = df[df["Chromosome"] == TOP_LAYER_CHROM]
    if not df_top.empty:
        sc1 = plt.scatter(df_top["Entropy"], df_top["ViralDist"], c=df_top["ECI"], cmap="viridis", s=45, edgecolor="black", linewidth=0.5, vmin=min(df["ECI"]), vmax=max(df["ECI"]))
    else:
        sc1 = plt.scatter(df["Entropy"], df["ViralDist"], c=df["ECI"], cmap="viridis", s=40)
        
    plt.colorbar(sc1, label="ECI")
    plt.xlabel("Normalized Entropy")
    plt.ylabel("Viral Divergence")
    plt.title(f"Entropy vs Viral divergence landscape (ECI)\n[{TOP_LAYER_CHROM} on Top]")
    
    plt.xlim(fig2_xlim)
    
    plt.tight_layout()
    plt.savefig("FIG2_entropy_viral_space_ECI.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 2-2. Chromosome Color Plot
    # ---------------------------------------------------------
    plt.figure(figsize=(7.5, 6))
    unique_chrs = sorted(df["Chromosome"].unique(), key=chr_sort_key)
    colors = plt.cm.get_cmap("tab20", len(unique_chrs))
    
    handles_dict = {}
    
    for i, chrom in enumerate(unique_chrs):
        if chrom == TOP_LAYER_CHROM:
            continue
        subset = df[df["Chromosome"] == chrom]
        sc = plt.scatter(subset["Entropy"], subset["ViralDist"], 
                         color=colors(i), s=50, alpha=0.7)
        handles_dict[chrom] = sc

    if TOP_LAYER_CHROM in unique_chrs:
        i_top = unique_chrs.index(TOP_LAYER_CHROM)
        subset_top = df[df["Chromosome"] == TOP_LAYER_CHROM]
        sc_top = plt.scatter(subset_top["Entropy"], subset_top["ViralDist"], 
                             color=colors(i_top), s=50, alpha=0.7)
        handles_dict[TOP_LAYER_CHROM] = sc_top

    plt.xlabel("Normalized Entropy")
    plt.ylabel("Viral Divergence")
    plt.title(f"Entropy vs Viral divergence landscape (Chromosome)")
    
    plt.legend([handles_dict[c] for c in unique_chrs if c in handles_dict], unique_chrs,
               title="Chromosome", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(fig2_xlim)

    plt.tight_layout()
    plt.savefig("FIG2_entropy_viral_space_Chromosome.png", dpi=300)
    plt.close()

    # =========================================================
    # FIGURE 3: ECI Ranking
    # =========================================================

    sorted_df = df.sort_values("ECI")

    plt.figure(figsize=(10,4))
    plt.plot(sorted_df["ECI"].values, marker="o", markersize=3)
    plt.xlabel("Individuals (sorted)")
    plt.ylabel("ECI")
    plt.title("Individual ECI ranking")
    plt.tight_layout()
    plt.savefig("FIG3_ECI_ranking.png", dpi=300)
    plt.close()

    # =========================================================
    # FIGURE 4: Outliers Detection
    # =========================================================

    z = (ECIs - np.mean(ECIs)) / np.std(ECIs)
    outlier_idx = np.where(np.abs(z) > 2.5)[0]

    outlier_ids = df.iloc[outlier_idx]["ID"].tolist()
    with open("outliers.txt", "w") as f:
        for oid in outlier_ids:
            f.write(f"{oid}\n")

    fig4_ylim = (0.181, 0.204)

    plt.figure(figsize=(8,5))
    plt.scatter(range(len(ECIs)), ECIs, c="gray", alpha=0.6)
    plt.scatter(outlier_idx, ECIs[outlier_idx], c="red", label="outliers")
    plt.xlabel("Individuals")
    plt.ylabel("ECI")
    plt.title("ECI outlier detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig("FIG4_ECI_outliers.png", dpi=300)
    plt.close()
    
    # =========================================================
    # FIGURE 5: GC Bias Assessment
    # =========================================================
    plt.figure(figsize=(6,4))

    plt.scatter(df["GC"], df["ECI"], c="gray", alpha=0.5, label="All samples")

    if len(df) > 1:
        slope, intercept = np.polyfit(df["GC"], df["ECI"], 1)
        x_vals = np.array([df["GC"].min(), df["GC"].max()])
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, color="red", linestyle="--",
                 linewidth=1.5, label="Trend line")

    plt.xlabel("GC Content")
    plt.ylabel("ECI")
    plt.title("GC Content vs ECI")

    # ---------------------------------------------------------
    # Legend 1: Data Contents
    # ---------------------------------------------------------
    ax = plt.gca()

    legend1 = ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        framealpha=0.9
    )

    ax.add_artist(legend1)

    plt.draw()

    renderer = plt.gcf().canvas.get_renderer()
    bbox = legend1.get_window_extent(renderer=renderer)
    bbox_axes = bbox.transformed(ax.transAxes.inverted())

    legend1_height = bbox_axes.height

    # ---------------------------------------------------------
    # Legend 2: Statistical Summary
    # ---------------------------------------------------------
    dummy = Line2D([], [], linestyle="", color="none")

    legend2 = ax.legend(
        [dummy],
        [f"Spearman $\\rho$: {corr:.4f}\np-value: {p:.3e}"],
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0 - legend1_height - 0.02),
        handlelength=0,
        handletextpad=0,
        framealpha=0.9
    )

    plt.tight_layout()
    plt.savefig("Check_GC_Bias.png", dpi=300)
    plt.close()
    
if __name__ == "__main__":
    main()
