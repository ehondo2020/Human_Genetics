import os
import sys
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

matrix_dir = "./Last_species_matrices"
os.makedirs(matrix_dir, exist_ok=True)

train_db_dir = "./train_db_temp"
os.makedirs("./train_db_temp", exist_ok=True)

prefixes = ["NC_021248_239976_279815_","AF063866_6_39989_"]

# Sequence extraction flanks (Upstream / Downstream)
PREFIX_FLANKS = {
    "NC_021248_239976_279815_": {
        "UPSTREAM_FLANK": 200000,
        "DOWNSTREAM_FLANK": 2500000,
    },
    "AF063866_6_39989_": {
        "UPSTREAM_FLANK": 1500000,
        "DOWNSTREAM_FLANK": 1500000,
    },
}

# Alignment filtering parameters
align_len_value = 100
identity_threshold = 0.7

# Lastal parameters
ALLOWANCE_NUMBER = 20
m_option = f"-m{ALLOWANCE_NUMBER}"
MIN_SCORE_GAP_ALIGN = 20
e_option = f"-e{MIN_SCORE_GAP_ALIGN}"
MAX_SCORE_DROP_GAP_ALIGN_X = 100
x_option = f"-x{MAX_SCORE_DROP_GAP_ALIGN_X}"
MAX_SCORE_DROP_GAP_ALIGN_Z = 100
z_option = f"-z{MAX_SCORE_DROP_GAP_ALIGN_Z}"

# last-split parameters
AMBIGUOUS_PARAM = 1.0
ambigous_option = f"-m{AMBIGUOUS_PARAM}"

# Sliding window size for alignment density calculations
WINDOW_SIZE=10000

# Parallel processing settings
MAX_WORKERS = 16
LAST_THREADS = 8
PARALLEL_WORKERS = 2
p_option = f"-P{LAST_THREADS}"

# Minimum alignment length to render links
MIN_DRAW_LEN = 200

# Minimum alignment length threshold for dot plots
DOT_PLOT_THRESHOLD = 0

# LAST parameters
D = 1e9
d_option = f"-D{D}"

C = 2
c_option = f"-C{C}"

# Target species order per prefix
PREFIX_TARGET_ORDERS = {
    "NC_021248_239976_279815_": [
        "Homo_sapiens",
        "JG3.0.0",
        "GCF_009914755.1",
        "GCF_Homo_ctrl",
        "GCF_JG_ctrl"
    ],
    "AF063866_6_39989_": [
        "Homo_sapiens",
        "JG3.0.0",
        "GCF_009914755.1"
    ]
}

# Link pairs: (Target, Query, Color)
LINK_PAIRS = [
    ("GCF_009914755.1", "Homo_sapiens", "dodgerblue"),
    ("GCF_009914755.1", "JG3.0.0", "forestgreen"),
    ("Homo_sapiens", "JG3.0.0", "purple"),
    ("GCF_009914755.1", "GCF_Homo_ctrl", "lightgrey"),
    ("GCF_009914755.1", "GCF_JG_ctrl", "darkgrey")
]

# Extract unique reference IDs
REFERENCE_IDS = list(set([pair[0] for pair in LINK_PAIRS]))

# Query chunking parameters
QUERY_CHUNK_NUM = 100
OVERLAP_SIZE = 20000

# Dot plot layout parameters
DP_FONTSIZE_LABEL = 10
DP_FONTSIZE_XLABEL = 11
DP_FONTSIZE_TITLE = 12

DP_FONTWEIGHT_LABEL = 'bold'
DP_FONTWEIGHT_XLABEL = 'bold'
DP_FONTWEIGHT_TITLE = 'bold'

DP_LINEWIDTH = 1.5
DP_ALPHA = 0.8

DP_TITLE_Y = 0.98
DP_DPI = 300

DP_YLABEL_PAD = None
DP_XLABEL_PAD = None

DP_X_REFERENCE_ID = "GCF_009914755.1"

def get_sort_index(record: SeqRecord, target_order) -> int:
    header_text = record.id + " " + record.description
    for i, keyword in enumerate(target_order):
        if keyword in header_text:
            return i
    return len(target_order)

# ==========================================
# 2. Sequence Extraction Worker Function
# ==========================================
def extract_sequence_worker(args):
    bed_path, prefix, fasta_dir, up_flank, down_flank = args
    filename = os.path.basename(bed_path)
    
    assembly_id = filename.replace(prefix, "").replace(".bed", "")

    if assembly_id == "GCF_Homo_ctrl":
        search_id = "GCF_009914755.1"
    elif assembly_id == "GCF_JG_ctrl":
        search_id = "GCF_009914755.1"
    else:
        search_id = assembly_id

    try:
        with open(bed_path, 'r') as f:
            line = f.readline().strip()
            if not line:
                return None
            parts = line.split('\t')
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
    except Exception:
        return None

    target_fasta_files = glob.glob(os.path.join(fasta_dir, f"*{search_id}*.fasta"))
    if not target_fasta_files:
        print(f"[ERROR] FASTA file for ID '{search_id}' not found in {fasta_dir}")
        return None

    target_fasta = target_fasta_files[0]

    try:
        fasta_dict = SeqIO.index(target_fasta, "fasta")
        if chrom in fasta_dict:
            full_seq = fasta_dict[chrom].seq.upper()
            ext_start = max(0, start - up_flank)
            ext_end = min(len(full_seq), end + down_flank)
            sub_seq = full_seq[ext_start:ext_end]

            rel_bed_start = start - ext_start
            rel_bed_end = end - ext_start

            return SeqRecord(
                sub_seq, 
                id=assembly_id, 
                description=f"extracted_from={chrom}:{ext_start+1}-{ext_end} rel_coords={rel_bed_start}:{rel_bed_end}"
            )
    except Exception as e:
        print(f"[ERROR] Exception during sequence extraction for {assembly_id}: {e}")

    return None

# ==========================================
# 3. Alignment Worker Function (Chunk-aware)
# ==========================================
def run_last_worker(args):
    idx, q_rec, db_name, matrix_path, prefix, user_color, chunk_offset = args
    
    maf_storage_dir = "./lastal_raw_maf_temp"
    os.makedirs(maf_storage_dir, exist_ok=True)
    
    ref_sp_id = os.path.basename(db_name).replace("ref_", "").replace("_db", "")
    
    stored_maf_raw = os.path.join(maf_storage_dir, f"{prefix}{q_rec.id}_vs_{ref_sp_id}.maf")
    
    temp_dir = f"./debug_maf_worker_{prefix}{q_rec.id}_{idx}_pid_{os.getpid()}"
    os.makedirs(temp_dir, exist_ok=True)
    
    query_fa = os.path.join(temp_dir, f"query_{idx}.fa")
    split_maf = os.path.join(temp_dir, f"split_{idx}.maf")
    
    try:
        SeqIO.write(q_rec, query_fa, "fasta")

        if os.path.exists(stored_maf_raw) and os.path.getsize(stored_maf_raw) > 0:
            pass
        else:
            cmd_lastal = ["lastal", p_option, m_option, d_option, e_option, x_option, z_option, "-v", "-p", matrix_path, db_name, query_fa]
            with open(stored_maf_raw, "w") as f_out:
                subprocess.run(cmd_lastal, stdout=f_out, stderr=sys.stderr, text=True, check=True)

        cmd_split = ["last-split", ambigous_option, stored_maf_raw]
        
        result_split = subprocess.run(cmd_split, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        
        lines = result_split.stdout.splitlines()

        if not lines:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return []

        links = []
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
                        
                        q_id_raw = s2[1]
                        q_id_original = q_id_raw.split("_chunk_")[0]
                        
                        q_start_local = int(s2[2])
                        q_aln_len = int(s2[3])
                        q_strand = s2[4]
                        q_seq_len_local = int(s2[5])
                        q_seq = s2[6]
                        
                        alen = len(t_seq)
                        
                        n_count_t = t_seq.upper().count('N')
                        n_count_q = q_seq.upper().count('N')
                        max_n_count = max(n_count_t, n_count_q)
                        n_ratio = max_n_count / alen if alen > 0 else 0
                        if n_ratio > 0.10:
                            i += 3
                            continue

                        valid_pairs = [(a.upper(), b.upper()) for a, b in zip(t_seq, q_seq) if a != '-' and b != '-']
                        pure_matches = sum(1 for a, b in valid_pairs if a == b)
                        alen_valid = len(valid_pairs)
                        identity = pure_matches / alen_valid if alen_valid > 0 else 0
                        
                        t_start_pos = t_start
                        t_end_pos = t_start + t_aln_len
                        
                        if q_strand == "-":
                            q_start_local_pos = q_seq_len_local - (q_start_local + q_aln_len)
                            q_end_local_pos = q_seq_len_local - q_start_local
                            strand = "-"
                        else:
                            q_start_local_pos = q_start_local
                            q_end_local_pos = q_start_local + q_aln_len
                            strand = "+"
                            
                        q_start_global = q_start_local_pos + chunk_offset
                        q_end_global = q_end_local_pos + chunk_offset
                            
                        if alen >= align_len_value and identity >= identity_threshold:
                            links.append({
                                "q_id": q_id_original, "q_start": q_start_global, "q_end": q_end_global,
                                "t_id": t_id, "t_start": t_start_pos, "t_end": t_end_pos,
                                "strand": strand, "identity": identity,
                                "user_color": user_color
                            })
                except Exception:
                    pass
                i += 3
            else:
                i += 1

        shutil.rmtree(temp_dir, ignore_errors=True)
        return links
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] execution failed in chunk {idx}: {e.stderr}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return []
    except Exception as e:
        print(f"[ERROR] Worker exception in chunk {idx}: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return []

def plot_alignment_density(prefix, ordered_records, all_links, window_size=WINDOW_SIZE):
    """
    Calculate alignment density using a sliding window and highlight EVE region.
    """
    import matplotlib.pyplot as plt
    
    ref_records = [r for r in ordered_records if r.id == DP_X_REFERENCE_ID]
    if len(ref_records) == 0:
        return
    ref_record = ref_records[0]
    anc_len = len(ref_record.seq)

    eve_start_x, eve_end_x = 0, 0
    match_ref_eve = re.search(r"rel_coords=(\d+):(\d+)", ref_record.description)
    if match_ref_eve:
        eve_start_x = int(match_ref_eve.group(1))
        eve_end_x = int(match_ref_eve.group(2))

    target_records = [r for r in ordered_records if r.id != DP_X_REFERENCE_ID and r.id != "Common_Ancestor"]
    num_targets = len(target_records)
    if num_targets == 0:
        return

    fig, axes = plt.subplots(num_targets, 1, figsize=(14, 4 * num_targets), sharex=True)
    if num_targets == 1: axes = [axes]
    
    for i, t_rec in enumerate(target_records):
        t_id = t_rec.id
        ax = axes[i]
        
        bins = [0] * (anc_len // window_size + 1)
        
        for l in all_links:
            if l["t_id"] == DP_X_REFERENCE_ID and l["q_id"] == t_id:
                s, e = l["t_start"], l["t_end"]
                start_idx = s // window_size
                end_idx = min(len(bins) - 1, e // window_size)
                for idx in range(start_idx, end_idx + 1):
                    bins[idx] += 1
        
        x_coords = [idx * window_size for idx in range(len(bins))]
        ax.fill_between(x_coords, bins, color="dodgerblue", alpha=0.4, label=f"Alignment Density (vs {t_id})")
        ax.plot(x_coords, bins, color="dodgerblue", lw=1.5)
        
        if eve_end_x > eve_start_x:
            ax.axvspan(eve_start_x, eve_end_x, color="orange", alpha=0.3, label="EVE Region")
            ax.axvline(eve_start_x, color="darkorange", ls="--", lw=1)
            ax.axvline(eve_end_x, color="darkorange", ls="--", lw=1)

        ax.set_ylabel(f"{t_id} (Links)", fontsize=11, fontweight='bold')
        ax.set_xlim(0, anc_len)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel(f"{DP_X_REFERENCE_ID} Position (bp)", fontsize=12)
    plt.tight_layout()
    output_path = f"./{prefix}alignment_density.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Success: Saved alignment density plot to {output_path}")

# ==========================================
# Main Process
# ==========================================
if __name__ == "__main__":
    print(f"--- 0.0 Generating multi-FASTA files (Parallel: {MAX_WORKERS} workers) ---")

    for prefix in prefixes:
        out_fasta = f"./{prefix}synteny.fa"
        
        if os.path.exists(out_fasta):
            print(f"[{out_fasta}] already exists. Skipping creation.")
            continue

        bed_files = glob.glob(os.path.join(bed_dir, f"{prefix}*.bed"))
        if not bed_files:
            print(f"Warning: No BED files found starting with {prefix}.")
            continue

        flanks = PREFIX_FLANKS[prefix]

        tasks = [
            (
                bp,
                prefix,
                fasta_dir,
                flanks["UPSTREAM_FLANK"],
                flanks["DOWNSTREAM_FLANK"],
            )
            for bp in bed_files
        ]
        
        extracted_records = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(extract_sequence_worker, tasks))
            extracted_records = [r for r in results if r is not None]

        if extracted_records:
            SeqIO.write(extracted_records, out_fasta, "fasta")
            print(f"Created [{out_fasta}] (Sequences: {len(extracted_records)})")
            
    print(f"--- 0.1 Training region/species-specific score matrices ---")

    species_matrices = {}

    for prefix in prefixes:
        print(f"\n[Region: {prefix}] Starting matrix training...")
        species_matrices[prefix] = {}
        
        prefix_matrix_dir = os.path.join(matrix_dir, prefix)
        os.makedirs(prefix_matrix_dir, exist_ok=True)
        
        def build_db_worker_local(ref_id, current_prefix):
            train_db_dir = "./train_db_temp"
            db_name = os.path.join(train_db_dir, f"ref_ext_db_{current_prefix}_{ref_id}")
            
            source_fasta = f"./{current_prefix}synteny.fa"
            ref_path = os.path.join(train_db_dir, f"{current_prefix}_{ref_id}_ext.fa")
            
            if not os.path.exists(ref_path):
                if not os.path.exists(source_fasta):
                    return None, None
                records = list(SeqIO.parse(source_fasta, "fasta"))
                target_recs = [r for r in records if ref_id in r.id]
                if not target_recs:
                    return None, None
                
                valid_recs = []
                for r in target_recs:
                    n_ratio = r.seq.upper().count('N') / len(r.seq) if len(r.seq) > 0 else 0
                    if n_ratio <= 0.10:
                        valid_recs.append(r)
                if not valid_recs:
                    print(f"  [WARNING] Reference sequence {ref_id} skipped due to high 'N' content.")
                    return None, None
                    
                SeqIO.write(valid_recs, ref_path, "fasta")
                
            if not os.path.exists(f"{db_name}.prj"):
                subprocess.run(["lastdb", p_option, "-uMAM8", db_name, ref_path], check=True)
            return ref_id, db_name

        ref_dbs = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            future_to_ref = {executor.submit(build_db_worker_local, rid, prefix): rid for rid in REFERENCE_IDS}
            for future in concurrent.futures.as_completed(future_to_ref):
                rid, db_path = future.result()
                if rid: ref_dbs[rid] = db_path

        for ref_id, species_id, _ in LINK_PAIRS:
            if species_id in ["GCF_Homo_ctrl", "GCF_JG_ctrl"]:
                continue
                
            if ref_id not in ref_dbs: continue
            db_name = ref_dbs[ref_id]
            
            matrix_path = os.path.join(prefix_matrix_dir, f"{species_id}_vs_{ref_id}.mat")
            
            if species_id not in species_matrices[prefix]:
                species_matrices[prefix][species_id] = {}
            species_matrices[prefix][species_id][ref_id] = matrix_path
            
            if os.path.exists(matrix_path) and os.path.getsize(matrix_path) > 0:
                continue
                
            source_fasta = f"./{prefix}synteny.fa"
            query_ext_fasta = os.path.join(train_db_dir, f"{prefix}_{species_id}_vs_{ref_id}_ext.fa")
            
            if not os.path.exists(query_ext_fasta):
                if not os.path.exists(source_fasta): continue
                records = list(SeqIO.parse(source_fasta, "fasta"))
                query_recs = [r for r in records if species_id in r.id]
                if not query_recs: continue
                
                valid_query_recs = []
                for r in query_recs:
                    n_ratio = r.seq.upper().count('N') / len(r.seq) if len(r.seq) > 0 else 0
                    if n_ratio <= 0.10:
                        valid_query_recs.append(r)
                if not valid_query_recs:
                    print(f"  [WARNING] Query sequence {species_id} skipped due to high 'N' content.")
                    continue
                
                SeqIO.write(valid_query_recs, query_ext_fasta, "fasta")
            
            print(f"  Training: {species_id} vs {ref_id} (Region: {prefix})...")
            with open(matrix_path, "w") as f_mat:
                subprocess.run(f"last-train {p_option} --revsym --matsym {d_option} {c_option} --postmask=0 --sample-number=5000 --verbose {db_name} {query_ext_fasta}", 
                               shell=True, stdout=f_mat, check=True)
                
    print(f"\n--- 1. Starting sequence extraction ---")
    for prefix in prefixes:
        out_fasta = f"./{prefix}synteny.fa"
        if os.path.exists(out_fasta): continue
        bed_files = glob.glob(os.path.join(bed_dir, f"{prefix}*.bed"))
                
        flanks = PREFIX_FLANKS[prefix]

        tasks = [
            (
                bp,
                prefix,
                fasta_dir,
                flanks["UPSTREAM_FLANK"],
                flanks["DOWNSTREAM_FLANK"],
            )
            for bp in bed_files
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(executor.map(extract_sequence_worker, tasks))
            extracted_records = [r for r in results if r is not None]
        if extracted_records:
            SeqIO.write(extracted_records, out_fasta, "fasta")

    print(f"\n--- 2. Starting alignment and visualization ---")
    for prefix in prefixes:
        fasta_file = f"./{prefix}synteny.fa"
        if not os.path.exists(fasta_file):
            print(f"[{fasta_file}] not found. Skipping.")
            continue

        all_records = list(SeqIO.parse(fasta_file, "fasta"))
        target_order = PREFIX_TARGET_ORDERS[prefix]

        records = [rec for rec in all_records if get_sort_index(rec, target_order) < len(target_order)]

        if len(records) < 2:
            print(f"[{prefix}] Insufficient sequences for comparison (requires at least 2).")
            continue

        record_dict = {rec.id: rec for rec in records}
        align_tasks = []
        task_id = 0
        
        for target_key, query_key, user_color in LINK_PAIRS:
            target_recs = [r for rid, r in record_dict.items() if target_key in rid]
            query_recs = [r for rid, r in record_dict.items() if query_key in rid]
            
            if not target_recs:
                print(f"[DEBUG] {prefix}: Target '{target_key}' not found in FASTA (check BED files).")
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
                        for p_key in species_matrices:
                            if prefix in p_key:
                                mat_path = species_matrices[p_key].get(q_sp, {}).get(t_sp, "")
                                if mat_path: break
                                
                    if q_sp == "GCF_Homo_ctrl":
                        mat_path = species_matrices.get(prefix, {}).get("Homo_sapiens", {}).get(t_sp, "")
                    elif q_sp == "GCF_JG_ctrl":
                        mat_path = species_matrices.get(prefix, {}).get("JG3.0.0", {}).get(t_sp, "")

                    if os.path.exists(mat_path):
                        seq_len = len(query_rec.seq)
                        chunk_size = seq_len // QUERY_CHUNK_NUM
                        chunk_records = []
                        
                        for i in range(QUERY_CHUNK_NUM):
                            start_pos = i * chunk_size
                            end_pos = (i + 1) * chunk_size if i < QUERY_CHUNK_NUM - 1 else seq_len
                            
                            ext_end_pos = min(seq_len, end_pos + OVERLAP_SIZE)
                            
                            chunk_seq = query_rec.seq[start_pos:ext_end_pos]
                            
                            chunk_id = f"{query_rec.id}_chunk_{i}"
                            chunk_rec = SeqRecord(chunk_seq, id=chunk_id, description="")
                            chunk_records.append(chunk_rec)
                            
                            align_tasks.append((task_id, chunk_rec, db_name, mat_path, prefix, user_color, start_pos))
                            task_id += 1
                            
                        overlap_out_fa = f"./{prefix}synteny_overlap_chunked_{q_sp}.fa"
                        if not os.path.exists(overlap_out_fa):
                            SeqIO.write(chunk_records, overlap_out_fa, "fasta")
                            
                    else:
                        print(f"[DEBUG] {prefix}: Scoring matrix for {q_sp} vs {t_sp} ({mat_path}) missing. Skipping alignment.")

        if not align_tasks:
            print(f"[DEBUG] {prefix}: No executable alignment tasks found.")
            continue

        print(f"[INFO] Starting alignment for all {len(align_tasks)} tasks (Parallel: {MAX_WORKERS})...")
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {executor.submit(run_last_worker, task): task for task in align_tasks}
            
            completed_count = 0
            for future in concurrent.futures.as_completed(future_to_task):
                completed_count += 1
                res = future.result()
                results.append(res)
                if completed_count % 10 == 0 or completed_count == len(align_tasks):
                    print(f"  ▶ Progress: {completed_count}/{len(align_tasks)} tasks completed ({(completed_count/len(align_tasks))*100:.1f}%)")
         
        records.sort(key=lambda r: get_sort_index(r, target_order))
        ordered_records = records
        circos = Circos({r.id: len(r.seq) for r in ordered_records}, space=3)

        has_results = False
        all_links = []

        for links_data in results:
            for l in links_data:
                has_results = True
                
                assigned_color = l["user_color"]
                target_sp_id = l["q_id"] + "_" + l["t_id"]
                for _, sp_name, sp_color in LINK_PAIRS:
                    if sp_name in target_sp_id and sp_color == l["user_color"]:
                        assigned_color = sp_color
                        break
                
                l["final_color"] = assigned_color
                all_links.append(l)

        if has_results:
            all_links.sort(key=lambda x: x["strand"] == "-")

            for l in all_links:
                if (l["q_end"] - l["q_start"]) < MIN_DRAW_LEN:
                    continue
                
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
            fig.savefig(f"./{prefix}synteny_pycirclize.png", dpi=300)
            plt.close(fig)
            print(f"Success: Output synteny plot for {prefix}.")
            plot_alignment_density(prefix, ordered_records, all_links, window_size=WINDOW_SIZE)
            
            print(f"[INFO] Generating multi-panel dotplots for {prefix}...")
            
            ref_records = [r for r in ordered_records if r.id == DP_X_REFERENCE_ID]
            
            if len(ref_records) > 0:
                ref_record = ref_records[0]
                anc_len = len(ref_record.seq)
                
                eve_start_x, eve_end_x = 0, 0
                match_ref_eve = re.search(r"rel_coords=(\d+):(\d+)", ref_record.description)
                if match_ref_eve:
                    eve_start_x = int(match_ref_eve.group(1))
                    eve_end_x = int(match_ref_eve.group(2))

                target_records = [r for r in ordered_records if r.id != DP_X_REFERENCE_ID and r.id != "Common_Ancestor"]
                num_targets = len(target_records)
                
                if num_targets > 0:
                    fig_dot, axes = plt.subplots(num_targets, 1, figsize=(7, 4 * num_targets), sharex=True)
                    if num_targets == 1:
                        axes = [axes]
                    
                    for idx, t_rec in enumerate(target_records):
                        ax = axes[idx]
                        t_id = t_rec.id
                        t_len = len(t_rec.seq)
                        
                        ax.set_facecolor("#fcfcfc")
                        ax.grid(True, which='both', color='#e0e0e0', linestyle='--', linewidth=0.5)
                        
                        t_eve_start, t_eve_end = 0, 0
                        match_t_eve = re.search(r"rel_coords=(\d+):(\d+)", t_rec.description)
                        if match_t_eve:
                            t_eve_start = int(match_t_eve.group(1))
                            t_eve_end = int(match_t_eve.group(2))

                        if (eve_end_x > eve_start_x) and (t_eve_end > t_eve_start):
                            rect_w = eve_end_x - eve_start_x
                            rect_h = t_eve_end - t_eve_start
                            rect = plt.Rectangle((eve_start_x, t_eve_start), rect_w, rect_h, 
                                                 facecolor="orange", edgecolor="darkorange", 
                                                 alpha=0.3, zorder=1, label="EVE Region")
                            ax.add_patch(rect)
                        
                        species_links = [l for l in all_links if l["t_id"] == DP_X_REFERENCE_ID and l["q_id"] == t_id]
                        
                        for l in species_links:
                            if (l["q_end"] - l["q_start"]) < DOT_PLOT_THRESHOLD: 
                                continue
                            
                            if "identity" in l and l["identity"] < identity_threshold:
                                continue
                            
                            x_coords = [l["t_start"], l["t_end"]]
                            
                            if l["strand"] == "-":
                                y_coords = [l["q_end"], l["q_start"]]
                                color = (0.85, 0.15, 0.15)
                            else:
                                y_coords = [l["start"], l["q_end"]] if "start" in l else [l["q_start"], l["q_end"]]
                                color = l.get("final_color", "#4b5563")
                            
                            ax.plot(x_coords, y_coords, color=color, linewidth=DP_LINEWIDTH, alpha=DP_ALPHA)
                        
                        ax.set_ylabel(f"{t_id} Position (bp)", fontsize=DP_FONTSIZE_LABEL, fontweight=DP_FONTWEIGHT_LABEL, labelpad=DP_YLABEL_PAD)
                        ax.set_xlim(0, anc_len)
                        ax.set_ylim(0, t_len)
                        ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
                        if eve_end_x > eve_start_x:
                            ax.legend(loc="upper right", fontsize=8)
                    
                    axes[-1].set_xlabel(f"{DP_X_REFERENCE_ID} Position (bp)", fontsize=DP_FONTSIZE_XLABEL, fontweight=DP_FONTWEIGHT_XLABEL, labelpad=DP_XLABEL_PAD)
                    
                    plt.suptitle(f"Dotplot Alignment: {DP_X_REFERENCE_ID} vs Primate Species\n({prefix.rstrip('_')})", 
                                 fontsize=DP_FONTSIZE_TITLE, fontweight=DP_FONTWEIGHT_TITLE, y=DP_TITLE_Y)
                    plt.tight_layout()
                    
                    dotplot_out = f"./{prefix}dotplot_panels.png"
                    fig_dot.savefig(dotplot_out, dpi=DP_DPI, bbox_inches="tight")
                    plt.close(fig_dot)
                    print(f"[SUCCESS] Multi-panel dotplot saved to {dotplot_out}")
            
        else:
            print(f"Notice: No valid alignment found for {prefix} (check filtering parameters).")