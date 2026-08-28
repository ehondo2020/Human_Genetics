import os
import re
import glob
import subprocess
import tempfile
import concurrent.futures
from multiprocessing import cpu_count
import shutil
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from pycirclize import Circos
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ==========================================
# 1. Directory and Prefix Settings
# ==========================================
bed_dir = "./Homo_entomopox_bed"
fasta_dir = "./Entomopox_Verkko_T2T_only_ref"

# Output directories for trained database and score matrices
train_db_dir = "./train_db_temp"
os.makedirs(train_db_dir, exist_ok=True)

matrix_dir = "./Last_species_matrices"
os.makedirs(matrix_dir, exist_ok=True)

prefixes = ["NC_021248_239976_279815_","AF063866_6_39989_"]

# Helper function to dynamically construct ancestor FASTA paths
def get_ancestor_fasta_path(prefix: str) -> str:
    return f"./Ancestor_link/Common_Ancestor_{prefix.rstrip('_')}.fasta"

# LAST alignment configuration
align_len_value = 100
identity_threshold = 0.75

# Lastal and last-split configuration
ALLOWANCE_NUMBER = 100
AMBIGUOUS_PARAM = 0.1
m_option = f"-m{ALLOWANCE_NUMBER}"

# Parallel processing settings
MAX_WORKERS = 16
LAST_THREADS = 8
PARALLEL_WORKERS = 2
p_option = f"-P{LAST_THREADS}"

# Minimum alignment length threshold for dot plots
DOT_PLOT_THRESHOLD = 100

# -D option setting
D = 1e9
d_option = f"-D{D}"

# Target species order
TARGET_ORDER = [
    "Common_Ancestor",
    "Homo_sapiens",
    "JG3.0.0",
    "GCF_009914755.1",
    "Pan_paniscus",
    "Pan_troglodytes",
    "Gorilla_gorilla"
]

# Link overlay order
LINK_OVERLAY_ORDER = [
    "Pan_paniscus",
    "Homo_sapiens",
    "GCF_009914755.1",
    "JG3.0.0",
    "Pan_troglodytes",
    "Gorilla_gorilla"
]

# Reference link pairs and display colors
LINK_PAIRS = [
    ("Common_Ancestor", "Homo_sapiens", "dodgerblue"),
    ("Common_Ancestor", "GCF_009914755.1", "forestgreen"),
    ("Common_Ancestor", "JG3.0.0", "purple"),
    ("Common_Ancestor", "Pan_paniscus", "magenta"),
    ("Common_Ancestor", "Pan_troglodytes", "teal"),
    ("Common_Ancestor", "Gorilla_gorilla", "indigo")
]

# Reference species selection
REFERENCE_IDS = ["Common_Ancestor"]

# ==========================================
# Dot Plot Design Parameters
# ==========================================
DP_FONTSIZE_LABEL = 14
DP_FONTSIZE_XLABEL = 14
DP_FONTSIZE_TITLE = 14

DP_FONTWEIGHT_LABEL = 'normal'
DP_FONTWEIGHT_XLABEL = 'normal'
DP_FONTWEIGHT_TITLE = 'normal'

DP_LINEWIDTH = 2.5
DP_ALPHA = 1.0

DP_TITLE_X = 0.5
DP_TITLE_Y = 1.0
DP_DPI = 300

DP_YLABEL_PAD = None
DP_XLABEL_PAD = None

def get_sort_index(record: SeqRecord) -> int:
    header_text = record.id + " " + record.description
    for i, keyword in enumerate(TARGET_ORDER):
        if keyword in header_text:
            return i
    return len(TARGET_ORDER)

# ==========================================
# 2. Sequence Extraction Worker
# ==========================================
def extract_sequence_worker(args):
    bed_path, prefix, fasta_dir, flank_size = args
    filename = os.path.basename(bed_path)
    assembly_id = filename.replace(prefix, "").replace(".bed", "")
    
    with open(bed_path, 'r') as f:
        line = f.readline().strip()
        if not line:
            return None
        parts = line.split('\t')
        chrom = parts[0]
        start = int(parts[1])
        end = int(parts[2])
        center = (start + end) // 2

    target_fasta_files = glob.glob(os.path.join(fasta_dir, f"*{assembly_id}*.fasta"))
    if not target_fasta_files:
        return None
    
    target_fasta = target_fasta_files[0]
    
    try:
        fasta_dict = SeqIO.index(target_fasta, "fasta")
        if chrom in fasta_dict:
            full_seq = fasta_dict[chrom].seq.upper()
            ext_start = max(0, center - flank_size)
            ext_end = center + flank_size
            sub_seq = full_seq[ext_start:ext_end]
            
            rel_bed_start = max(0, start - ext_start)
            rel_bed_end = max(0, end - ext_start)
            rel_bed_end = min(len(sub_seq), rel_bed_end)
            
            return SeqRecord(
                sub_seq, 
                id=assembly_id, 
                description=f"extracted_from={chrom}:{ext_start+1}-{ext_end} rel_coords={rel_bed_start}:{rel_bed_end}"
            )
    except Exception as e:
        print(f"Error processing {assembly_id}: {e}")
    return None

# ==========================================
# 3. Alignment Worker
# ==========================================
def run_last_worker(args):
    idx, q_rec, db_name, matrix_path, prefix, user_color = args
    
    temp_dir = f"./debug_maf_worker_{prefix}{idx}_pid_{os.getpid()}"
    os.makedirs(temp_dir, exist_ok=True)
    
    query_fa = os.path.join(temp_dir, f"query_{idx}.fa")
    maf_raw = os.path.join(temp_dir, f"raw_{idx}.maf")
    split_maf = os.path.join(temp_dir, f"split_{idx}.maf")
    
    try:
        SeqIO.write(q_rec, query_fa, "fasta")

        cmd_lastal = ["lastal", p_option, m_option, d_option, "-p", matrix_path, db_name, query_fa]
        with open(maf_raw, "w") as f_out:
            subprocess.run(cmd_lastal, stdout=f_out, stderr=subprocess.PIPE, text=True, check=True)

        cmd_split = ["last-split", "-m", str(AMBIGUOUS_PARAM), maf_raw]
        with open(split_maf, "w") as f_out:
            subprocess.run(cmd_split, stdout=f_out, stderr=subprocess.PIPE, text=True, check=True)

        if not os.path.exists(split_maf) or os.path.getsize(split_maf) == 0:
            return []

        links = []
        with open(split_maf, "r") as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("a"):
                try:
                    s1_line = lines[i+1].strip()
                    s2_line = lines[i+2].strip()
                    if s1_line.startswith("s ") and s2_line.startswith("s "):
                        s1 = s1_line.split()
                        s2 = s2_line.split()
                        
                        t_id = s1[1]
                        t_start = int(s1[2])
                        t_aln_len = int(s1[3])
                        t_seq = s1[6]
                        
                        q_id = s2[1]
                        q_start = int(s2[2])
                        q_aln_len = int(s2[3])
                        q_strand = s2[4]
                        q_seq_len = int(s2[5])
                        q_seq = s2[6]
                        
                        matches = sum(1 for a, b in zip(t_seq, q_seq) if a.upper() == b.upper() and a != '-')
                        alen = len(t_seq)
                        
                        # Filter alignments exceeding the 'N' threshold (10%)
                        n_count_t = t_seq.upper().count('N')
                        n_count_q = q_seq.upper().count('N')
                        max_n_count = max(n_count_t, n_count_q)
                        n_ratio = max_n_count / alen if alen > 0 else 0
                        
                        if n_ratio > 0.10:
                            i += 3
                            continue

                        # Calculate identity based exclusively on canonical bases (A, C, G, T)
                        valid_pairs = [
                            (a.upper(), b.upper())
                            for a, b in zip(t_seq, q_seq)
                            if (
                                a.upper() in ("A", "T", "G", "C")
                                and
                                b.upper() in ("A", "T", "G", "C")
                            )
                        ]

                        pure_matches = sum(
                            1
                            for a, b in valid_pairs
                            if a == b
                        )

                        alen_valid = len(valid_pairs)

                        identity = (
                            pure_matches / alen_valid
                            if alen_valid > 0
                            else 0.0
                        )

                        t_start_pos = t_start
                        t_end_pos = t_start + t_aln_len

                        if q_strand == "-":
                            q_start_pos = q_seq_len - (q_start + q_aln_len)
                            q_end_pos = q_seq_len - q_start
                            strand = "-"
                        else:
                            q_start_pos = q_start
                            q_end_pos = q_start + q_aln_len
                            strand = "+"

                        if (
                            alen_valid >= align_len_value
                            and
                            identity >= identity_threshold
                        ):
                            links.append({
                                "q_id": q_id,
                                "q_start": q_start_pos,
                                "q_end": q_end_pos,
                                "t_id": t_id,
                                "t_start": t_start_pos,
                                "t_end": t_end_pos,
                                "strand": strand,
                                "identity": identity,
                                "user_color": user_color
                            })

                except Exception:
                    pass
                i += 3
            else:
                i += 1
        return links
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] lastal/last-split execution failed: {e.stderr}")
        return []
    except Exception as e:
        print(f"[ERROR] Worker exception: {e}")
        return []

# ==========================================
# Main Pipeline
# ==========================================
if __name__ == "__main__":
    prefix_ancestor_data = {}
    print("--- Starting dynamic indexing of ancestral sequences ---")
    for prefix in prefixes:
        anc_path = get_ancestor_fasta_path(prefix)
        if os.path.exists(anc_path):
            anc_rec = list(SeqIO.parse(anc_path, "fasta"))[0]
            prefix_ancestor_data[prefix] = {
                "path": anc_path,
                "record": anc_rec,
                "flank_size": len(anc_rec.seq) // 2
            }
            print(f"  [{prefix}] -> {anc_path} ({len(anc_rec.seq)} bp)")
        else:
            print(f"  [WARN] Ancestral FASTA file not found: {anc_path}")

    print(f"\n--- 0.0 Starting multi-FASTA generation (Parallel: {MAX_WORKERS} workers) ---")
    for prefix in prefixes:
        if prefix not in prefix_ancestor_data:
            print(f"[SKIP] {prefix}: Skipped due to missing ancestral data.")
            continue
            
        out_fasta = f"./{prefix}synteny.fa"
        
        if os.path.exists(out_fasta):
            print(f"[{out_fasta}] already exists. Skipping multi-FASTA creation.")
            continue

        bed_files = glob.glob(os.path.join(bed_dir, f"{prefix}*.bed"))
        if not bed_files:
            print(f"Warning: No BED files found starting with {prefix}.")
            continue

        current_flank_size = prefix_ancestor_data[prefix]["flank_size"]
        tasks = [(bp, prefix, fasta_dir, current_flank_size) for bp in bed_files]
        
        extracted_records = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(extract_sequence_worker, tasks))
            extracted_records = [r for r in results if r is not None]

        import copy
        anc_record = copy.deepcopy(prefix_ancestor_data[prefix]["record"])
        anc_record.id = "Common_Ancestor"
        anc_record.description = f"rel_coords=0:{len(anc_record.seq)}"
        extracted_records.append(anc_record)

        if extracted_records:
            SeqIO.write(extracted_records, out_fasta, "fasta")
            print(f"Created [{out_fasta}] (Sequences: {len(extracted_records)})")
            
    print(f"\n--- 0.1 Training species-specific scoring matrices ---")
    
    species_matrices = {}
    
    for prefix in prefixes:
        if prefix not in prefix_ancestor_data:
            continue
            
        species_matrices[prefix] = {}
        prefix_matrix_dir = os.path.join(matrix_dir, prefix.rstrip("_"))
        os.makedirs(prefix_matrix_dir, exist_ok=True)
        
        ref_dbs = {}
        for ref_id in REFERENCE_IDS:
            db_name = os.path.join(train_db_dir, f"ref_full_db_{prefix}_{ref_id}")
            if ref_id == "Common_Ancestor":
                ref_path = prefix_ancestor_data[prefix]["path"]
            else:
                ref_genome_search = glob.glob(os.path.join(fasta_dir, f"*{ref_id}*.fasta"))
                if not ref_genome_search:
                    continue
                ref_path = ref_genome_search[0]
                
            if not os.path.exists(f"{db_name}.prj"):
                print(f"Building reference database [{prefix}]: {ref_id}")
                subprocess.run(["lastdb", p_option, "-uMAM8", db_name, ref_path], check=True)
            else:
                print(f"[SKIP] Reference DB {ref_id} ({prefix}) already exists. Reusing.")
            ref_dbs[ref_id] = db_name
            
        for ref_id in REFERENCE_IDS:
            if ref_id not in ref_dbs: continue
            db_name = ref_dbs[ref_id]
            
            for species_id in TARGET_ORDER:
                if species_id in REFERENCE_IDS:
                    continue

                matrix_path = os.path.join(prefix_matrix_dir, f"{species_id}_vs_{ref_id}.mat")

                if species_id not in species_matrices[prefix]:
                    species_matrices[prefix][species_id] = {}
                species_matrices[prefix][species_id][ref_id] = matrix_path

                if os.path.exists(matrix_path) and os.path.getsize(matrix_path) > 0:
                    continue

                all_query_fasta = f"./{prefix}synteny.fa"

                if not os.path.exists(all_query_fasta):
                    print(f"[WARN] Training skipped: {all_query_fasta} not found.")
                    continue

                species_query_fasta = os.path.join(
                    train_db_dir,
                    f"{prefix}_{species_id}_training.fa"
                )

                records = []
                for rec in SeqIO.parse(all_query_fasta, "fasta"):
                    if species_id in rec.id:
                        records.append(rec)

                if len(records) == 0:
                    print(f"[WARN] Training sequences for {species_id} not found in {prefix}.")
                    continue

                SeqIO.write(records, species_query_fasta, "fasta")

                print(
                    f"Training [{prefix}]: {species_id} vs {ref_id} "
                    f"({len(records)} sequences)"
                )

                with open(matrix_path, "w") as f_mat:
                    subprocess.run(
                        f"last-train "
                        f"{p_option} "
                        f"--revsym "
                        f"{d_option} "
                        f"--postmask=0 "
                        f"--sample-number=5000 "
                        f"--verbose "
                        f"{db_name} "
                        f"{species_query_fasta}",
                        shell=True,
                        stdout=f_mat,
                        check=True
                    )
                
    print(f"\n--- 2. Starting alignment and visualization ---")
    for prefix in prefixes:
        fasta_file = f"./{prefix}synteny.fa"
        if not os.path.exists(fasta_file):
            continue

        records = list(SeqIO.parse(fasta_file, "fasta"))
        if len(records) < 2:
            continue

        record_dict = {rec.id: rec for rec in records}
        align_tasks = []
        task_id = 0
        
        for target_key, query_key, user_color in LINK_PAIRS:
            target_recs = [r for rid, r in record_dict.items() if rid == target_key]
            query_recs  = [r for rid, r in record_dict.items() if rid == query_key]
            
            if not target_recs:
                print(f"[DEBUG] {prefix}: Target '{target_key}' not found in FASTA.")
                continue
            if not query_recs:
                print(f"[DEBUG] {prefix}: Query '{query_key}' not found in FASTA.")
                continue
            
            for target_rec in target_recs:
                lastdb_dir = f"./lastdb_{prefix}_{target_key}"
                os.makedirs(lastdb_dir, exist_ok=True)
                db_name = os.path.join(lastdb_dir, f"ref_{target_key}_db")
                ref_fa_path = os.path.join(lastdb_dir, f"ref_{target_key}.fa")
                
                if not os.path.exists(ref_fa_path):
                    SeqIO.write(target_rec, ref_fa_path, "fasta")
                
                if not os.path.exists(f"{db_name}.prj"):
                    subprocess.run(["lastdb", "-uMAM8", db_name, ref_fa_path], check=True)

                for query_rec in query_recs:
                    q_sp = query_key
                    t_sp = target_key
                    mat_path = species_matrices.get(prefix, {}).get(q_sp, {}).get(t_sp, "")
                    
                    if not mat_path:
                        for sp_id in species_matrices.get(prefix, {}):
                            if q_sp in sp_id:
                                mat_path = species_matrices[prefix][sp_id].get(t_sp, "")
                                if mat_path: break

                    if os.path.exists(mat_path):
                        align_tasks.append((task_id, query_rec, db_name, mat_path, prefix, user_color))
                        task_id += 1
                    else:
                        print(f"[DEBUG] {prefix}: Scoring matrix for {q_sp} vs {t_sp} ({mat_path}) does not exist.")

        if not align_tasks:
            continue

        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(run_last_worker, align_tasks))

        records.sort(key=get_sort_index)
        ordered_records = records
        circos = Circos({r.id: len(r.seq) for r in ordered_records}, space=3)

        has_results = False
        all_links = []
        
        for links_data in results:
            for l in links_data:
                has_results = True
                
                target_sp_id = l["q_id"] if l["q_id"] != "Common_Ancestor" else l["t_id"]
                
                assigned_color = l["user_color"]
                for _, sp_name, sp_color in LINK_PAIRS:
                    if sp_name in target_sp_id:
                        assigned_color = sp_color
                        break
                
                l["final_color"] = assigned_color
                all_links.append(l)

        if has_results:
            def get_link_sort_key(l):
                sp_id = l["q_id"] if l["q_id"] != "Common_Ancestor" else l["t_id"]
                try:
                    sp_idx = next(i for i, name in enumerate(LINK_OVERLAY_ORDER) if name in sp_id)
                except StopIteration:
                    sp_idx = -1
                
                return (l["strand"] == "-", sp_idx)

            all_links.sort(key=get_link_sort_key)
            
            for l in all_links:
                if l["strand"] == "-":
                    color = (0.85, 0.15, 0.15)
                else:
                    color = l["final_color"]
                
                circos.link((l["q_id"], l["q_start"], l["q_end"]),
                            (l["t_id"], l["t_start"], l["t_end"]),
                            color=color, alpha=1.0, lw=0.1, ec=color, direction=0, rasterized=True)
                
        if has_results:
            for rec in ordered_records:
                sector = circos.get_sector(rec.id)
                track = sector.add_track((95, 100))
                track.axis(fc="lightgrey")
                track.text(rec.id, r=115, size=12)
                match = re.search(r"rel_coords=(\d+):(\d+)", rec.description)
                if match:
                    track.rect(int(match.group(1)), int(match.group(2)), fc="orange", ec="none")

            fig = circos.plotfig()
            fig.savefig(f"./{prefix}Common_Ancestor_synteny.png", dpi=300)
            plt.close(fig)
            print(f"Success: Generated synteny plot for {prefix}.")
            
            print(f"[INFO] Generating multi-panel dotplots for {prefix}...")
            
            target_records = [r for r in ordered_records if r.id != "Common_Ancestor"]
            num_targets = len(target_records)
            
            if num_targets > 0:
                fig_dot, axes = plt.subplots(num_targets, 1, figsize=(7, 4 * num_targets), sharex=True)
                if num_targets == 1:
                    axes = [axes]
                
                anc_len = len(anc_record.seq)
                
                for idx, t_rec in enumerate(target_records):
                    ax = axes[idx]
                    t_id = t_rec.id
                    t_len = len(t_rec.seq)
                    
                    ax.set_facecolor("#fcfcfc")
                    ax.grid(True, which='both', color='#e0e0e0', linestyle='--', linewidth=0.5)
                    
                    species_links = [l for l in all_links if l["q_id"] == t_id]
                    
                    for l in species_links:
                        if (l["q_end"] - l["q_start"]) < DOT_PLOT_THRESHOLD: 
                            continue
                        
                        x_coords = [l["t_start"], l["t_end"]]
                        
                        if l["strand"] == "-":
                            y_coords = [l["q_end"], l["q_start"]]
                            color = (0.85, 0.15, 0.15)
                        else:
                            y_coords = [l["q_start"], l["q_end"]]
                            color = l.get("final_color", "#4b5563")
                        
                        ax.plot(x_coords, y_coords, color=color, linewidth=DP_LINEWIDTH, alpha=DP_ALPHA)
                    
                    ax.set_ylabel(f"{t_id}\n(bp)", fontsize=DP_FONTSIZE_LABEL, fontweight=DP_FONTWEIGHT_LABEL, labelpad=DP_YLABEL_PAD)
                    ax.set_xlim(0, anc_len)
                    ax.set_ylim(0, t_len)
                    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
                
                axes[-1].set_xlabel("Common Ancestor Coordinate (bp)", fontsize=DP_FONTSIZE_XLABEL, fontweight=DP_FONTWEIGHT_XLABEL, labelpad=DP_XLABEL_PAD)
                
                plt.suptitle(f"{prefix.rstrip('_')}", 
                             fontsize=DP_FONTSIZE_TITLE, fontweight=DP_FONTWEIGHT_TITLE, x=DP_TITLE_X, y=DP_TITLE_Y)
                plt.tight_layout()
                
                dotplot_out = f"./{prefix}dotplot_panels.png"
                fig_dot.savefig(dotplot_out, dpi=DP_DPI, bbox_inches="tight")
                plt.close(fig_dot)
                print(f"[SUCCESS] Multi-panel dotplot saved to {dotplot_out}")
            
        else:
            print(f"Notice: No valid alignments found for {prefix}.")