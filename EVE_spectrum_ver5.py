import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend setup for HPC/headless environments
import matplotlib.pyplot as plt
import seaborn as sns

# Publication-ready style settings
sns.set_theme(style="ticks", context="paper")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

def read_fasta(fasta_path):
    """Parse FASTA file into a dictionary."""
    sequences = {}
    with open(fasta_path, 'r') as f:
        header = None
        seq_parts = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header:
                    sequences[header] = "".join(seq_parts)
                header = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header:
            sequences[header] = "".join(seq_parts)
    return sequences

def analyze_true_alignment(alignment_fa, region_name, output_prefix):
    print(f"\n--- Processing {region_name} ---")
    alignment = read_fasta(alignment_fa)
    
    # 1. Identify ancestral sequence
    ancestor_id = None
    for idx in alignment.keys():
        if "ancestor" in idx.lower() or "common" in idx.lower():
            ancestor_id = idx
            break
            
    if not ancestor_id:
        print(f"[ERROR] Could not automatically identify ancestor header (Ancestor) in alignment.")
        print(f"Header list in alignment:")
        for hid in alignment.keys(): print(f"  >{hid}")
        sys.exit(1)
        
    print(f"-> Identified Ancestor row: >{ancestor_id}")
    
    ancestor_seq = alignment[ancestor_id].upper()
    haplotypes = {k: v.upper() for k, v in alignment.items() if k != ancestor_id}
    alignment_length = len(ancestor_seq)
    
    # 2. Count substitutions at strictly homologous sites (Tracking Ti/Tv)
    data_rows = []
    total_at_to_gc = 0
    total_gc_to_at = 0
    
    for hap_id, hap_seq in haplotypes.items():
        at_to_gc = 0
        gc_to_at = 0
        at_to_gc_ti = 0  # Transition: A->G, T->C
        at_to_gc_tv = 0  # Transversion: A->C, T->G
        
        for i in range(alignment_length):
            anc_base = ancestor_seq[i]
            hap_base = hap_seq[i]
            
            # Skip site if either base is invalid (gap/N)
            if _is_invalid(anc_base) or _is_invalid(hap_base):
                continue
                
            if _is_mutated(anc_base, hap_base):
                if anc_base in ['A', 'T'] and hap_base in ['G', 'C']:
                    at_to_gc += 1
                    if (anc_base == 'A' and hap_base == 'G') or (anc_base == 'T' and hap_base == 'C'):
                        at_to_gc_ti += 1
                    elif (anc_base == 'A' and hap_base == 'C') or (anc_base == 'T' and hap_base == 'G'):
                        at_to_gc_tv += 1
                elif anc_base in ['G', 'C'] and hap_base in ['A', 'T']:
                    gc_to_at += 1
                    
        total_at_to_gc += at_to_gc
        total_gc_to_at += gc_to_at
        
        data_rows.append({'Haplotype': hap_id, 'Region': region_name, 'Direction': 'A/T → G/C', 'Count': at_to_gc, 'Ti': at_to_gc_ti, 'Tv': at_to_gc_tv})
        data_rows.append({'Haplotype': hap_id, 'Region': region_name, 'Direction': 'G/C → A/T', 'Count': gc_to_at, 'Ti': 0, 'Tv': 0})
        
    macro_b = total_at_to_gc / total_gc_to_at if total_gc_to_at > 0 else 0
    print(f"-> Region Macro Bias Ratio (B): {macro_b:.4f}")

    # 3. Compute pairwise distance matrix and generate NEXUS output (Vectorized with NumPy)
    print("-> Generating pairwise distance matrix for SplitsTree (Vectorized)...")
    all_taxa = list(alignment.keys())
    num_taxa = len(all_taxa)
    
    # Convert alignment sequences to 1-byte NumPy arrays
    np_seqs = {t: np.frombuffer(alignment[t].upper().encode('ascii'), dtype='S1') for t in all_taxa}
    
    distance_matrix = {t: {} for t in all_taxa}
    for i in range(num_taxa):
        t1 = all_taxa[i]
        distance_matrix[t1][t1] = 0.0
        for j in range(i + 1, num_taxa):
            t2 = all_taxa[j]
            s1 = np_seqs[t1]
            s2 = np_seqs[t2]
            
            # Mask array for non-gap valid sites
            valid_mask = (s1 != b'-') & (s2 != b'-')
            valid_sites = np.sum(valid_mask)
            
            if valid_sites > 0:
                # Count mismatched sites among valid positions
                diff = np.sum(valid_mask & (s1 != s2))
                p_dist = float(diff) / valid_sites
            else:
                p_dist = 0.0
                
            distance_matrix[t1][t2] = p_dist
            distance_matrix[t2][t1] = p_dist

    # ==========================================
    # Generate MDS (Multidimensional Scaling) evolutionary space plot from distance matrix
    # ==========================================
    print("-> Computing MDS (Multidimensional Scaling) for evolutionary space...")
    from sklearn.manifold import MDS
    import re

    # Natural sorting key for chromosome names
    def chr_sort_key(chrom_str):
        name = chrom_str.replace("chr", "")
        if name.isdigit():
            return (0, int(name))
        elif name in ["X", "Y", "Z", "W"]:
            return (1, name)
        else:
            return (2, name)

    # Convert distance matrix to a symmetric NumPy array
    dist_array = np.zeros((num_taxa, num_taxa))
    for i, t1 in enumerate(all_taxa):
        for j, t2 in enumerate(all_taxa):
            dist_array[i, j] = distance_matrix[t1][t2]
            
    # Perform 2D MDS reduction using precomputed distance matrix
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
    mds_coords = mds.fit_transform(dist_array)

    # Extract chromosome names from FASTA headers
    chromosomes = []
    for t in all_taxa:
        if t == ancestor_id:
            chromosomes.append("Ancestor")
        else:
            # Extract chromosome name matching pattern
            match = re.search(r'(chr[0-9a-zA-Z]+)', t)
            if match:
                chromosomes.append(match.group(1))
            else:
                chromosomes.append("Unknown")
    
    # Construct DataFrame for plotting
    mds_df = pd.DataFrame({
        'X': mds_coords[:, 0],
        'Y': mds_coords[:, 1],
        'Taxa': all_taxa,
        'Is_Ancestor': [1 if t == ancestor_id else 0 for t in all_taxa],
        'Chromosome': chromosomes
    })
    
    # Initialize MDS plot figure
    fig_mds, ax_mds = plt.subplots(figsize=(7.5, 6), dpi=300)
    
    # Extract and sort unique chromosome identifiers
    unique_chrs = [c for c in mds_df['Chromosome'].unique() if c not in ["Ancestor", "Unknown"]]
    unique_chrs = sorted(unique_chrs, key=chr_sort_key)
    
    # Dynamically map color palette to chromosomes
    colors = plt.cm.get_cmap("tab20", max(1, len(unique_chrs)))
    handles_dict = {}
        
    # 1. Plot haplotypes color-coded by chromosome
    for i, chrom in enumerate(unique_chrs):
        subset = mds_df[(mds_df['Chromosome'] == chrom) & (mds_df['Is_Ancestor'] == 0)]
        if not subset.empty:
            sc = ax_mds.scatter(subset['X'], subset['Y'], 
                                color=colors(i), alpha=0.6, s=25, edgecolors='none')
            handles_dict[chrom] = sc

    # Fallback for unclassified chromosome headers
    subset_unknown = mds_df[(mds_df['Chromosome'] == "Unknown") & (mds_df['Is_Ancestor'] == 0)]
    if not subset_unknown.empty:
        sc_unk = ax_mds.scatter(subset_unknown['X'], subset_unknown['Y'], 
                                color='gray', alpha=0.5, s=25, edgecolors='none')
        handles_dict["Unknown"] = sc_unk
        unique_chrs.append("Unknown")
    
    # 2. Plot Common Ancestor as a highlighted marker
    anc_mask = mds_df['Is_Ancestor'] == 1
    if np.any(anc_mask):
        sc_anc = ax_mds.scatter(mds_df.loc[anc_mask, 'X'], mds_df.loc[anc_mask, 'Y'], 
                       c='red', marker='*', s=200, edgecolors='white', linewidths=1.5, zorder=5)
        handles_dict["Common Ancestor"] = sc_anc
        
    # Customize plot labels and design
    ax_mds.set_xlabel("MDS Dimension 1", fontsize=11, fontweight='bold')
    ax_mds.set_ylabel("MDS Dimension 2", fontsize=11, fontweight='bold')
    ax_mds.set_title(f"Evolutionary Space Map ({region_name})", fontsize=12, fontweight='bold', pad=15)
    
    # Create legend outside plotting area
    legend_keys = unique_chrs + ["Common Ancestor"] if np.any(anc_mask) else unique_chrs
    leg = ax_mds.legend([handles_dict[k] for k in legend_keys if k in handles_dict], legend_keys,
                        title="Chromosome", bbox_to_anchor=(1.05, 1), loc='upper left', 
                        frameon=True, facecolor='white', edgecolor='gray', fontsize=9)
    leg.get_frame().set_linewidth(0.5)
    
    sns.despine(trim=True)
    plt.tight_layout()
    
    # Save figure files
    plt.savefig(f"{output_prefix}_evolutionary_MDS.png", format='png', bbox_inches='tight')
    plt.savefig(f"{output_prefix}_evolutionary_MDS.pdf", format='pdf', bbox_inches='tight')
    plt.close()
    print(f"-> Saved Evolutionary MDS Map: {output_prefix}_evolutionary_MDS.png")
    # ==========================================

    # Write NEXUS format file
    nexus_path = f"{output_prefix}_splits_network.nex"
    with open(nexus_path, "w") as nex:
        nex.write("#NEXUS\n\n")
        
        # TAXA block
        nex.write("BEGIN TAXA;\n")
        nex.write(f"  DIMENSIONS NTAX={num_taxa};\n")
        nex.write("  TAXLABELS\n")
        for t in all_taxa:
            nex.write(f"    '{t}'\n")
        nex.write("  ;\nEND;\n\n")
        
        # DISTANCES block
        nex.write("BEGIN DISTANCES;\n")
        nex.write(f"  DIMENSIONS NTAX={num_taxa};\n")
        nex.write("  FORMAT TRIANGLE=BOTH DIAGONAL LABELS;\n")
        nex.write("  MATRIX\n")

        for t1 in all_taxa:

            row = []

            for t2 in all_taxa:
                row.append(f"{distance_matrix[t1][t2]:.6f}")

            nex.write(
                f"  '{t1}' " +
                " ".join(row) +
                "\n"
            )

        nex.write("  ;\n")
        nex.write("END;\n")
        
    print(f"-> Saved SplitsTree NEXUS: {nexus_path}")
    return data_rows

def _is_invalid(base):
    return base in ['-', 'N', 'X']

def _is_mutated(anc, hap):
    return anc != hap

if __name__ == "__main__":
    # Input alignment fasta files
    af_alignment = "./human_variants_AF_Ancestor_with_Ancestor_mafft_genafpair_full.fa" 
    nc_alignment = "./human_variants_NC_Ancestor_with_Ancestor_mafft_genafpair_full.fa"
    
    if not (os.path.exists(af_alignment) and os.path.exists(nc_alignment)):
        print(f"[ERROR] Alignment file not found. Please check file paths.")
        sys.exit(1)
        
    print("Starting Rigorous EVE Evolutionary Analysis Pipeline...")
    # Process each target genomic domain
    af_rows = analyze_true_alignment(af_alignment, "AF063866:6-39989", "AF_EVE")
    nc_rows = analyze_true_alignment(nc_alignment, "NC021248:239976-279815", "NC_EVE")
    
    df = pd.DataFrame(af_rows + nc_rows)
    
    # Calculate statistics and render optimized boxplots
    print("\nCalculating statistics & rendering optimized boxplots for each region...")
    from scipy.stats import wilcoxon
    
    colors = {'A/T → G/C': '#d95f02', 'G/C → A/T': '#7570b3'}
    regions = ["AF063866:6-39989", "NC021248:239976-279815"]
    region_prefixes = {
        "AF063866:6-39989": "AF_EVE",
        "NC021248:239976-279815": "NC_EVE"
    }
    
    # Iterate through regions to generate individual plots
    for reg in regions:
        reg_df = df[df['Region'] == reg]
        
        # Initialize canvas figure
        fig, ax = plt.subplots(figsize=(5.5, 6), dpi=300)
        
        # Main boxplot and stripplot visualization
        sns.boxplot(data=reg_df, x='Region', y='Count', hue='Direction', palette=colors, fliersize=0, width=0.45, ax=ax)
        sns.stripplot(data=reg_df, x='Region', y='Count', hue='Direction', palette=colors, dodge=True, alpha=0.2, size=2.5, jitter=0.2, ax=ax, legend=False)
        
        # Prepare data for paired testing
        at_gc_df = reg_df[reg_df['Direction'] == 'A/T → G/C'].sort_values('Haplotype')
        at_gc = at_gc_df['Count'].values
        gc_at = reg_df[reg_df['Direction'] == 'G/C → A/T'].sort_values('Haplotype')['Count'].values
        
        # Perform Wilcoxon signed-rank test
        stat, p_val = wilcoxon(at_gc, gc_at)
        
        # Calculate upper whisker bound to determine display scale
        combined_data = np.concatenate([at_gc, gc_at])
        q3 = np.percentile(combined_data, 75)
        q1 = np.percentile(combined_data, 25)
        iqr = q3 - q1
        whisker_top = q3 + 1.5 * iqr
        
        # Determine non-outlier maximum value
        non_outliers = combined_data[combined_data <= whisker_top]
        display_max = np.max(non_outliers) if len(non_outliers) > 0 else q3
        
        # Set dynamic vertical margins and bracket heights
        y_bar = display_max * 1.05
        h = display_max * 0.03
        
        # Draw significance brackets
        x_left = 0 - 0.11
        x_right = 0 + 0.11
        ax.plot([x_left, x_left, x_right, x_right], [y_bar, y_bar+h, y_bar+h, y_bar], lw=1.2, c='black')
        
        # Format p-value string
        if p_val < 0.001:
            p_text = f"p = {p_val:.4e}"
        else:
            p_text = f"p = {p_val:.4f}"
            
        # Annotate p-value text above bracket
        text_y = y_bar + h + (display_max * 0.02)
        ax.text((x_left + x_right) / 2, text_y, p_text, ha='center', va='bottom', color='black', fontsize=9, fontweight='bold')

        # Ti vs Tv breakdown statistics and annotation
        ti_list = at_gc_df['Ti'].values
        tv_list = at_gc_df['Tv'].values
        mean_ti = np.mean(ti_list)
        mean_tv = np.mean(tv_list)
        mean_all_at_gc = mean_ti + mean_tv
        
        ti_pct = (mean_ti / mean_all_at_gc) * 100 if mean_all_at_gc > 0 else 0.0
        tv_pct = (mean_tv / mean_all_at_gc) * 100 if mean_all_at_gc > 0 else 0.0
        
        try:
            _, p_ti_tv = wilcoxon(ti_list, tv_list)
        except Exception:
            p_ti_tv = 1.0
            
        p_ti_tv_text = f"p = {p_ti_tv:.4e}" if p_ti_tv < 0.001 else f"p = {p_ti_tv:.4f}"
        
        info_box_text = (
            "gBGC Ti/Tv Breakdown (A/T → G/C)\n"
            f"• Transitions (A→G, T→C): {mean_ti:.1f} ({ti_pct:.1f}%)\n"
            f"• Transversions (A→C, T→G): {mean_tv:.1f} ({tv_pct:.1f}%)\n"
            f"• Wilcoxon Signed-Rank {p_ti_tv_text}"
        )
        
        # Place summary information text box
        ax.text(0.05, 0.08, info_box_text, transform=ax.transAxes, fontsize=8.5, fontweight='bold',
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='lightgray', alpha=0.9, lw=0.8))

        # Adjust Y-axis scale dynamically
        ax.set_ylim(- (display_max * 0.05), y_bar + h + (display_max * 0.15))
        
        # Set axes labels and legend
        ax.set_xlabel("Genomic Domain", fontsize=12, fontweight='bold', labelpad=10)
        ax.set_ylabel("Number of Mutations per Haplotype", fontsize=12, fontweight='bold', labelpad=10)
        ax.set_title(f"Substitution Spectrum & gBGC Bias\n({reg})", fontsize=13, fontweight='bold', pad=20)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[0:2], labels[0:2], title="Mutation Direction", frameon=True, facecolor='white', edgecolor='none', loc='upper right')
        sns.despine(trim=True)
        plt.tight_layout()
        
        # Save PDF and PNG outputs
        prefix = region_prefixes.get(reg, "EVE_region")
        fig_pdf = f"Figure_{prefix}_Substitution_Spectrum_Bias.pdf"
        fig_png = f"Figure_{prefix}_Substitution_Spectrum_Bias.png"
        plt.savefig(fig_pdf, format='pdf', bbox_inches='tight')
        plt.savefig(fig_png, format='png', bbox_inches='tight')
        plt.close()
        print(f"  - Generated Plot: {fig_pdf} & {fig_png}")
        
    print(f"\n[SUCCESS] Pipeline completed successfully.")
    print(f"  - Generated Plots:")
    print(f"    1. Figure_AF_EVE_Substitution_Spectrum_Bias.pdf / .png")
    print(f"    2. Figure_NC_EVE_Substitution_Spectrum_Bias.pdf / .png")
    print(f"  - Generated NEXUS: AF_region_splits_network.nex & NC_region_splits_network.nex")