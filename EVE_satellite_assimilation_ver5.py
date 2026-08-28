import os
import sys
import numpy as np
import pandas as pd
import itertools
from itertools import product
from scipy.spatial.distance import jensenshannon, cosine
from scipy.stats import mannwhitneyu, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Percentile thresholds to handle outliers in Figure 1 color scale
VD_VMIN = 1
VD_VMAX = 100 - VD_VMIN

ECI_VMIN = 1
ECI_VMAX = 100 -VD_VMAX

# Publication-ready plotting style settings
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def read_fasta(fasta_path):
    """Parse FASTA file and retrieve pure sequence excluding gaps (-) and Ns."""
    sequences = {}
    with open(fasta_path, 'r') as f:
        header = None
        seq_parts = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header:
                    sequences[header] = "".join(seq_parts).replace('-', '').replace('N', '').upper()
                header = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header:
            sequences[header] = "".join(seq_parts).replace('-', '').replace('N', '').upper()
    return sequences

def get_kmer_profile(sequence, k, kmer_to_index):
    """Generate k-mer probability distribution vector (P) from sequence."""
    counts = np.zeros(len(kmer_to_index), dtype=np.float64)
    # Apply Laplace smoothing to avoid log(0)
    counts += 0.01 
    
    total_kmers = len(sequence) - k + 1
    if total_kmers <= 0:
        return counts / np.sum(counts)
        
    for i in range(total_kmers):
        kmer = sequence[i:i+k]
        if kmer in kmer_to_index:
            counts[kmer_to_index[kmer]] += 1.0
            
    return counts / np.sum(counts)

def calculate_gc_content(sequence):
    """Calculate GC content (%) of sequence."""
    if len(sequence) == 0:
        return 0.0
    gc_count = sequence.count('G') + sequence.count('C')
    return (gc_count / len(sequence)) * 100

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

def entropy(vec, k=5):
    p = vec / np.sum(vec)
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return H / np.log(4**k)

def ECI(vec, viral_vec, k=5):
    H = entropy(vec, k)
    D = cosine(vec, viral_vec)
    return (1 - H) * (1 - D), H, D

def main():
    # ==== File Paths ====
    af_fasta = "./AF063866_6_39989_human_variants_Ancestor.fa"
    nc_fasta = "./NC_021248_239976_279815_human_variants_Ancestor.fa"
    sat_fasta = "./centromere_pericentromere_satellites.fa"
    viral_fasta = "./AF063866_6_39989_for_species_comparison.fa"
    k_size = 5

    for f in [af_fasta, nc_fasta, sat_fasta, viral_fasta]:
        if not os.path.exists(f):
            print(f"[ERROR] Required file not found: {f}")
            sys.exit(1)

    print(f"--- Initialization ---")
    print(f"Generating {k_size}-mer dictionary...")
    bases = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in itertools.product(bases, repeat=k_size)]
    kmer_to_index = {kmer: i for i, kmer in enumerate(all_kmers)}

    # 1. Construct Host Satellite Background and Viral Vector
    print(f"Loading Host Satellite Background: {sat_fasta}")
    sat_seqs = read_fasta(sat_fasta)
    combined_sat_seq = "".join(sat_seqs.values())
    q_background = get_kmer_profile(combined_sat_seq, k_size, kmer_to_index)

    print(f"Loading Viral Reference Sequence: {viral_fasta}")
    viral_seqs = read_fasta(viral_fasta)
    viral_vec = np.mean([kmer_freq(s, k=k_size) for s in viral_seqs.values()], axis=0)

    # 2. Calculate Metrics (JSD, GC Content, Viral Divergence, ECI)
    results = []
    
    regions = {
        "AF063866:6-39989": af_fasta,
        "NC021248:239976-279815": nc_fasta
    }

    for region_name, fasta_file in regions.items():
        print(f"Processing {region_name}...")
        seqs = read_fasta(fasta_file)
        
        for hap_id, seq in seqs.items():
            if "ancestor" in hap_id.lower() or "common" in hap_id.lower():
                continue
                
            p_target = get_kmer_profile(seq, k_size, kmer_to_index)
            jsd_value = (jensenshannon(p_target, q_background, base=2)) ** 2
            gc_content = calculate_gc_content(seq)
            
            vec_kmer = kmer_freq(seq, k=k_size)
            eci_val, h_val, d_val = ECI(vec_kmer, viral_vec, k=k_size)
            
            results.append({
                "Haplotype": hap_id,
                "Region": region_name,
                "JSD_to_Satellite": jsd_value,
                "GC_Content": gc_content,
                "Viral_Divergence": d_val,
                "ECI": eci_val
            })

    df = pd.DataFrame(results)
    
    csv_out = "Table_EVE_Satellite_Assimilation.csv"
    df.to_csv(csv_out, index=False)
    print(f"-> Saved numerical matrix to {csv_out}")

    # ==== Statistical Analysis and Plotting ====
    print("Calculating statistics and rendering plots...")
    
    # Mann-Whitney U Test
    af_jsd = df[df["Region"] == "AF063866:6-39989"]["JSD_to_Satellite"]
    nc_jsd = df[df["Region"] == "NC021248:239976-279815"]["JSD_to_Satellite"]
    stat, p_val_mw = mannwhitneyu(af_jsd, nc_jsd, alternative='two-sided')

    y_min = df["JSD_to_Satellite"].min()
    y_max = df["JSD_to_Satellite"].max()
    y_range = y_max - y_min
    p_text = f"p = {p_val_mw:.2e}" if p_val_mw < 0.001 else f"p = {p_val_mw:.4f}"

    # Spearman Correlation
    reg_af = "AF063866:6-39989"
    reg_nc = "NC021248:239976-279815"

    df_af = df[df["Region"] == reg_af]
    df_nc = df[df["Region"] == reg_nc]

    def fmt_p(p):
        return f"p = {p:.2e}" if p < 0.001 else f"p = {p:.4f}"

    rho_vd_af, p_vd_af = spearmanr(df_af["JSD_to_Satellite"], df_af["Viral_Divergence"])
    rho_vd_nc, p_vd_nc = spearmanr(df_nc["JSD_to_Satellite"], df_nc["Viral_Divergence"])

    rho_eci_af, p_eci_af = spearmanr(df_af["JSD_to_Satellite"], df_af["ECI"])
    rho_eci_nc, p_eci_nc = spearmanr(df_nc["JSD_to_Satellite"], df_nc["ECI"])

    region_order = [reg_af, reg_nc]
    region_map = {reg: i for i, reg in enumerate(region_order)}
    
    # Random seed for scatter plot jittering
    np.random.seed(42)
    x_positions = df["Region"].map(region_map) + np.random.uniform(-0.12, 0.12, size=len(df))

    # =========================================================
    # FIGURE 1: EVE Assimilation Level by Region (Viral Divergence & ECI)
    # =========================================================
    fig1, axes1 = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

    vd_vmin = np.percentile(df["Viral_Divergence"], VD_VMIN)
    vd_vmax = np.percentile(df["Viral_Divergence"], VD_VMAX)

    eci_vmin = np.percentile(df["ECI"], ECI_VMIN)
    eci_vmax = np.percentile(df["ECI"], ECI_VMAX)

    # --- Left Panel: Viral Divergence ---
    ax_left = axes1[0]
    sns.boxplot(data=df, x="Region", y="JSD_to_Satellite", order=region_order,
                color="#e0e0e0", width=0.4, ax=ax_left, fliersize=0,
                boxprops=dict(alpha=0.5), zorder=1)
    
    sc_left = ax_left.scatter(x_positions, df["JSD_to_Satellite"], c=df["Viral_Divergence"],
                              cmap="cividis", vmin=vd_vmin, vmax=vd_vmax, alpha=0.8, s=35, edgecolors='none',
                              zorder=2)
    
    cbar_left = fig1.colorbar(sc_left, ax=ax_left, fraction=0.046, pad=0.04, extend='both')
    cbar_left.set_label("Viral Divergence", fontsize=10, fontweight='bold')

    ax_left.set_ylim(y_min - y_range * 0.10, y_max + y_range * 0.35)
    y_bar = y_max + (y_range * 0.08)
    ax_left.plot([0, 0, 1, 1], [y_bar, y_bar+(y_range*0.04), y_bar+(y_range*0.04), y_bar], lw=1.2, c='black')
    ax_left.text(0.5, y_bar + (y_range * 0.08), p_text, ha='center', va='bottom', color='black', fontsize=10, fontweight='bold')
    
    text_vd = (
        f"AF: Spearman $\\rho$ = {rho_vd_af:.3f} ({fmt_p(p_vd_af)})\n"
        f"NC: Spearman $\\rho$ = {rho_vd_nc:.3f} ({fmt_p(p_vd_nc)})"
    )
    ax_left.text(0.03, 0.03, text_vd, transform=ax_left.transAxes, ha='left', va='bottom', fontsize=8.5, fontweight='bold')

    ax_left.set_ylabel("JSD (Distance to Host Satellites)", fontsize=11, fontweight='bold')
    ax_left.set_xlabel("EVE Domain", fontsize=11, fontweight='bold')
    ax_left.set_title("Assimilation Level (colored by Viral Divergence)", fontsize=11, fontweight='bold')

    # --- Right Panel: ECI ---
    ax_right = axes1[1]
    sns.boxplot(data=df, x="Region", y="JSD_to_Satellite", order=region_order,
                color="#e0e0e0", width=0.4, ax=ax_right, fliersize=0,
                boxprops=dict(alpha=0.5), zorder=1)
    
    sc_right = ax_right.scatter(x_positions, df["JSD_to_Satellite"], c=df["ECI"],
                               cmap="viridis", vmin=eci_vmin, vmax=eci_vmax, alpha=0.8, s=35, edgecolors='none',
                               zorder=2)
    
    cbar_right = fig1.colorbar(sc_right, ax=ax_right, fraction=0.046, pad=0.04, extend='both')
    cbar_right.set_label("ECI", fontsize=10, fontweight='bold')

    ax_right.set_ylim(y_min - y_range * 0.10, y_max + y_range * 0.35)
    ax_right.plot([0, 0, 1, 1], [y_bar, y_bar+(y_range*0.04), y_bar+(y_range*0.04), y_bar], lw=1.2, c='black')
    ax_right.text(0.5, y_bar + (y_range * 0.08), p_text, ha='center', va='bottom', color='black', fontsize=10, fontweight='bold')
    
    text_eci = (
        f"AF: Spearman $\\rho$ = {rho_eci_af:.3f} ({fmt_p(p_eci_af)})\n"
        f"NC: Spearman $\\rho$ = {rho_eci_nc:.3f} ({fmt_p(p_eci_nc)})"
    )
    ax_right.text(0.03, 0.03, text_eci, transform=ax_right.transAxes, ha='left', va='bottom', fontsize=8.5, fontweight='bold')

    ax_right.set_ylabel("JSD (Distance to Host Satellites)", fontsize=11, fontweight='bold')
    ax_right.set_xlabel("EVE Domain", fontsize=11, fontweight='bold')
    ax_right.set_title("Assimilation Level (colored by ECI)", fontsize=11, fontweight='bold')

    sns.despine(fig=fig1, trim=True)
    fig1.tight_layout()
    
    fig1_pdf = "Figure1_EVE_Assimilation_by_Region.pdf"
    fig1_png = "Figure1_EVE_Assimilation_by_Region.png"
    fig1.savefig(fig1_pdf, format='pdf', bbox_inches='tight')
    fig1.savefig(fig1_png, format='png', bbox_inches='tight')
    plt.close(fig1)

    # =========================================================
    # FIGURE 2: Assimilation vs. GC Content
    # =========================================================
    fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=300)
    colors = {"NC021248:239976-279815": "#e41a1c", "AF063866:6-39989": "#377eb8"}

    sns.scatterplot(data=df, x="GC_Content", y="JSD_to_Satellite", hue="Region", palette=colors, alpha=0.6, edgecolor=None, ax=ax2)
    
    for idx, (reg_name, color) in enumerate(colors.items()):
        sub_df = df[df["Region"] == reg_name]
        rho, p_val_sp = spearmanr(sub_df["GC_Content"], sub_df["JSD_to_Satellite"])
        sig_star = "***" if p_val_sp < 0.001 else "**" if p_val_sp < 0.01 else "*" if p_val_sp < 0.05 else "ns (p>0.05)"
        
        ax2.text(0.95, 0.90 - (idx*0.08), f"{reg_name.split(':')[0]}: $\\rho$={rho:.2f} {sig_star}", 
                 transform=ax2.transAxes, ha='right', va='top', fontsize=9, color=color, fontweight='bold')

    ax2.set_xlabel("EVE GC Content (%)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("JSD (Distance to Host Satellites)", fontsize=11, fontweight='bold')
    ax2.set_title("Assimilation vs. GC Content", fontsize=12, fontweight='bold')
    ax2.legend(title="Region", frameon=True, fontsize=8)

    sns.despine(ax=ax2, trim=True)
    fig2.tight_layout()
    
    fig2_pdf = "Figure2_Assimilation_vs_GC_Content.pdf"
    fig2_png = "Figure2_Assimilation_vs_GC_Content.png"
    fig2.savefig(fig2_pdf, format='pdf', bbox_inches='tight')
    fig2.savefig(fig2_png, format='png', bbox_inches='tight')
    plt.close(fig2)

    print("\n[SUCCESS] Assimilation analysis completed.")
    print(f"  - Generated Data Matrix: {csv_out}")
    print(f"  - Generated Figure 1: {fig1_pdf} / {fig1_png}")
    print(f"  - Generated Figure 2: {fig2_pdf} / {fig2_png}")

if __name__ == "__main__":
    main()
