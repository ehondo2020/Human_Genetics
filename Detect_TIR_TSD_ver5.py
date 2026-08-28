import os
import glob
import subprocess
import sys
from Bio import Align
from Bio.Seq import Seq
import math
import concurrent.futures
import numpy as np
from numba import njit
import itertools
from pyfaidx import Fasta

# ==========================================
# Parameter Settings
# ==========================================
BASE_DIR = "./Entomopox_Verkko_T2T_only_ref"
FLANK_SIZE = 5000        # Search range in bp on both sides of EPVE
MIN_TIR_LEN = 25         # Minimum alignment length for TIR
MIN_TIR_IDEN = 0.80      # Minimum identity threshold for TIR (80%)
MAX_TSD_MISMATCH_RATE = 0.05  # Maximum mismatch rate for TSD (5%)
E_VALUE_THRESHOLD = 0.06     # E-value threshold for significance
TSD_WINDOW = 50          # Window size to search for TSD adjacent to TIR
MIN_TSD_LEN = 8          # Minimum length for TSD
MAX_TSD_LEN = 15         # Maximum length for TSD

# --- Alignment and Scoring Parameters ---
ALIGN_MATCH_SCORE = 1
ALIGN_MISMATCH_SCORE = -2
ALIGN_OPEN_GAP_SCORE = -5
ALIGN_EXTEND_GAP_SCORE = -1

# Output file definitions
OUT_TIR = "Strict_Evidence_Summary_TIR.txt"
OUT_DR = "Strict_Evidence_Summary_DR.txt"
DEBUG_LOG = "Detailed_Debug_Log.txt"

THREAD_BUFFER = 2

def run_cmd(cmd):
    """Execute command and return output or None if failed"""
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        return None
    return result.stdout

@njit(fastmath=True)
def find_tsd_numba(l_arr, r_arr, min_len, max_len, max_mismatch_rate):
    len_l = len(l_arr)
    len_r = len(r_arr)
    
    for length in range(max_len, min_len - 1, -1):
        allowed_mismatches = int(length * max_mismatch_rate)
        
        for i in range(len_l - length + 1):
            sub_l = l_arr[i:i+length]
            
            # Sequence complexity check
            is_complex = False
            first_val = sub_l[0]
            for k in range(1, length):
                if sub_l[k] != first_val:
                    is_complex = True
                    break
            if not is_complex:
                continue
                
            for j in range(len_r - length + 1):
                sub_r = r_arr[j:j+length]
                
                mismatches = 0
                for k in range(length):
                    if sub_l[k] != sub_r[k]:
                        mismatches += 1
                        if mismatches > allowed_mismatches:
                            break
                
                if mismatches <= allowed_mismatches:
                    return i, length
    return -1, -1

def find_tsd(left_window, right_window, min_len, max_len):
    """Search for TSD between two windows allowing mismatches"""
    l_arr = np.frombuffer(left_window.upper().encode('ascii'), dtype=np.uint8)
    r_arr = np.frombuffer(right_window.upper().encode('ascii'), dtype=np.uint8)
    
    idx, length = find_tsd_numba(l_arr, r_arr, min_len, max_len, MAX_TSD_MISMATCH_RATE)
    if idx != -1:
        return left_window[idx:idx+length]
    return None

def calculate_e_value(score, m, n):
    """Calculate E-value based on Karlin-Altschul statistics"""
    if score <= 0: return 999
    lambd = 1.1
    K = 0.1
    try:
        e_value = K * m * n * math.exp(-lambd * score)
    except OverflowError:
        e_value = 0
    return e_value

def process_bed_file(bed_file):
    """Worker function to process each BED file independently"""
    summary_lines = []
    debug_lines = []
    
    aligner = Align.PairwiseAligner()
    aligner.mode = 'local'
    aligner.match_score = ALIGN_MATCH_SCORE
    aligner.mismatch_score = ALIGN_MISMATCH_SCORE
    aligner.open_gap_score = ALIGN_OPEN_GAP_SCORE
    aligner.extend_gap_score = ALIGN_EXTEND_GAP_SCORE
    
    try:
        filename = os.path.basename(bed_file)
        name_without_ext = filename.rsplit('.', 1)[0]
        
        if "GCF_" in name_without_ext:
            genome_id = "GCF_" + name_without_ext.split("GCF_")[1]
        else:
            parts = name_without_ext.split('_')
            last_digit_idx = -1
            for i, p in enumerate(parts):
                if p.isdigit():
                    last_digit_idx = i
            
            if last_digit_idx != -1 and last_digit_idx < len(parts) - 1:
                genome_id = "_".join(parts[last_digit_idx + 1:])
            else:
                genome_id = name_without_ext
        
        fasta_list = [f for f in os.listdir(BASE_DIR) if f.endswith(".fasta") and genome_id in f]
        if not fasta_list:
            debug_lines.append(f"[DEBUG] File: {filename} | SKIP: Corresponding FASTA not found (Search ID: {genome_id})\n")
            return summary_lines, debug_lines
        
        genome_fasta = os.path.join(BASE_DIR, fasta_list[0])
        
        from Bio import SeqIO
        try:
            genome_dict = Fasta(genome_fasta)
        except Exception as e:
            debug_lines.append(f"[DEBUG] File: {filename} | ERROR: FASTA load failed ({e})\n")
            return summary_lines, debug_lines

        with open(bed_file, 'r') as f_in:
            for line in f_in:
                if not line.strip(): continue
                cols = line.strip().split('\t')
                chrom, start, end = cols[0], int(cols[1]), int(cols[2])

                if chrom not in genome_dict:
                    debug_lines.append(f"[DEBUG] File: {filename} | Region: {chrom}:{start}-{end} | SKIP: {chrom} not found\n")
                    continue

                up_start = max(0, start - FLANK_SIZE)
                down_end = end + FLANK_SIZE

                left_seq = str(genome_dict[chrom][up_start:start]).upper()
                right_seq_raw = str(genome_dict[chrom][end:down_end]).upper()

                if not left_seq or not right_seq_raw:
                    continue
                
                vals = [left_seq, right_seq_raw]
                right_seq_rc = str(Seq(right_seq_raw).reverse_complement())

                search_modes = [
                    ("TIR", right_seq_rc, True),
                    ("DR", right_seq_raw, False)
                ]

                all_region_evidences = []

                for search_type, target_seq, is_rc in search_modes:
                    candidates = []
                    try:
                        alignments = aligner.align(left_seq, target_seq)
                        best_score_in_loop = None

                        for aln in itertools.islice(alignments, 50):
                            if best_score_in_loop is None:
                                best_score_in_loop = aln.score

                            l_start, l_end = aln.aligned[0][0][0], aln.aligned[0][-1][1]
                            t_start, t_end = aln.aligned[1][0][0], aln.aligned[1][-1][1]
                            cur_len = l_end - l_start
                            
                            if cur_len < MIN_TIR_LEN: continue
                            cur_eval = calculate_e_value(aln.score, len(left_seq), len(target_seq))
                            if cur_eval > E_VALUE_THRESHOLD: continue
                            
                            cur_iden = str(aln).count('|') / cur_len
                            if cur_iden < MIN_TIR_IDEN: continue
                            
                            gap_l = len(left_seq) - l_end
                            gap_r = len(target_seq) - t_end
                            total_gap = gap_l + gap_r
                            
                            score_threshold = cur_len * (MIN_TIR_IDEN * ALIGN_MATCH_SCORE + (1 - MIN_TIR_IDEN) * ALIGN_MISMATCH_SCORE)
                            if cur_len < 1000 and aln.score < score_threshold:
                                continue

                            candidates.append({
                                "aln": aln,
                                "total_gap": total_gap,
                                "identity": cur_iden,
                                "len": cur_len,
                                "score": aln.score,
                                "e_value": cur_eval,
                                "coords": (l_start, l_end, t_start, t_end)
                            })
                            if aln.score < (best_score_in_loop * 0.4): break
                                
                    except (OverflowError, Exception):
                        continue

                    if not candidates:
                        continue

                    for cand in candidates:
                        lf_m_s, lf_m_e = cand["coords"][0], cand["coords"][1]
                        t_m_s, t_m_e = cand["coords"][2], cand["coords"][3]
                        
                        if is_rc:
                            rf_m_s, rf_m_e = len(target_seq) - t_m_e, len(target_seq) - t_m_s
                            right_repeat_seq = str(Seq(vals[1][rf_m_s:rf_m_e]).reverse_complement())
                        else:
                            rf_m_s, rf_m_e = t_m_s, t_m_e
                            right_repeat_seq = vals[1][rf_m_s:rf_m_e]

                        tsd_l_win = left_seq[max(0, lf_m_s - TSD_WINDOW):lf_m_s]
                        tsd_r_win = vals[1][rf_m_e:rf_m_e + TSD_WINDOW]
                        tsd_match = find_tsd(tsd_l_win, tsd_r_win, MIN_TSD_LEN, MAX_TSD_LEN)

                        if tsd_match:
                            left_repeat_seq = left_seq[lf_m_s:lf_m_e]
                            l_tsd_offset = tsd_l_win.upper().rfind(tsd_match.upper())
                            l_tsd_start = up_start + max(0, lf_m_s - TSD_WINDOW) + l_tsd_offset
                            l_tsd_end = l_tsd_start + len(tsd_match)
                            actual_left_tsd = tsd_l_win[l_tsd_offset : l_tsd_offset + len(tsd_match)]

                            r_tsd_offset = tsd_r_win.upper().find(tsd_match.upper())
                            r_tsd_start = end + rf_m_e + r_tsd_offset
                            r_tsd_end = r_tsd_start + len(tsd_match)
                            actual_right_tsd = tsd_r_win[r_tsd_offset : r_tsd_offset + len(tsd_match)]

                            gap_tsd_l = (up_start + lf_m_s) - l_tsd_end
                            gap_l_eve = start - (up_start + lf_m_e)
                            gap_eve_r = (end + rf_m_s) - end
                            gap_r_tsd = r_tsd_start - (end + rf_m_e)

                            res_txt = (
                                f"========================================\n"
                                f"[Evidence Detected] File: {filename}\n"
                                f"  [EPVE Region] {chrom}:{start}-{end}\n"
                                f"  [Structural Type] {search_type} ({'Inverted' if is_rc else 'Direct'})\n"
                                f"  [Terminal Repeat] Length: {cand['len']}bp, Identity: {cand['identity']*100:.1f}%, Score: {cand['score']:.1f}, E-value: {cand['e_value']:.2e}\n"
                                f"    Left {search_type} : {chrom}:{up_start+lf_m_s}-{up_start+lf_m_e}\n"
                                f"      [Sequence] {left_repeat_seq}\n"
                                f"    Right {search_type}: {chrom}:{end+rf_m_s}-{end+rf_m_e}\n"
                                f"      [Sequence] {right_repeat_seq}\n"
                                f"  [TSD] Length: {len(tsd_match)}bp\n"
                                f"    Left TSD : {chrom}:{l_tsd_start}-{l_tsd_end} [Sequence] {actual_left_tsd}\n"
                                f"    Right TSD: {chrom}:{r_tsd_start}-{r_tsd_end} [Sequence] {actual_right_tsd}\n"
                                f"  [Distance Information] TSD-bases-{search_type}-bases-EVE-bases-{search_type}-bases-TSD\n"
                                f"                         TSD-'{gap_tsd_l}'-{search_type}-'{gap_l_eve}'-EVE-'{gap_eve_r}'-{search_type}-'{gap_r_tsd}'-TSD\n"
                            )
                            
                            all_region_evidences.append({
                                "text": res_txt,
                                "tsd_len": len(tsd_match),
                                "gap_tsd_tir": gap_tsd_l + gap_r_tsd,
                                "identity": cand["identity"],
                                "gap_tir_eve": gap_l_eve + gap_eve_r
                            })

                found_at_least_one = False
                if all_region_evidences:
                    all_region_evidences.sort(key=lambda x: (
                        -x["tsd_len"], 
                        x["gap_tsd_tir"], 
                        -x["identity"], 
                        x["gap_tir_eve"]
                    ))
                    summary_lines.append(all_region_evidences[0]["text"])
                    found_at_least_one = True

                if not found_at_least_one:
                    debug_lines.append(f"[DEBUG] File: {filename} | Region: {chrom}:{start}-{end} | SKIP: Insufficient valid TIR/DR or TSD\n")

    except Exception as e:
        debug_lines.append(f"[DEBUG] File: {bed_file} | ERROR: Unexpected exception ({e})\n")

    return summary_lines, debug_lines

def main():
    print(f"--- TIR/TSD Search and Extraction Program (Parallel Mode) ---")
    
    search_path = os.path.join(BASE_DIR, "FASTA_36_results_*/Final_Fasta36_results/*.bed")
    all_bed_files = glob.glob(search_path, recursive=True)
    bed_files = [f for f in all_bed_files if not f.endswith("_chromosome.bed")]

    if not bed_files:
        print(f"Target BED files not found. Search path: {search_path}")
        return

    max_workers = max(1, os.cpu_count() - THREAD_BUFFER)
    print(f"Processing {len(bed_files)} files using {max_workers} processes in parallel...")

    with open(OUT_TIR, "w") as f_tir, open(OUT_DR, "w") as f_dr, open(DEBUG_LOG, "w") as f_debug:
        header = (
            "### Entomopox-like Element TIR/TSD Strict Evidence Summary ###\n"
            f"Parameters: TIR_IDEN>={MIN_TIR_IDEN}, TIR_LEN>={MIN_TIR_LEN}, TSD_LEN={MIN_TSD_LEN}-{MAX_TSD_LEN}\n\n"
        )
        f_tir.write(header)
        f_dr.write(header)
        
        f_debug.write("### TIR/TSD Detailed Debug Log ###\n")
        f_debug.write("Records reasons for region rejection (SKIP reasons).\n\n")

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_bed_file, bed_file): bed_file for bed_file in bed_files}
            
            for future in concurrent.futures.as_completed(futures):
                summary_lines, debug_lines = future.result()
                
                for line in summary_lines:
                    print(line, end="")
                    
                    if "[Structural Type] TIR" in line:
                        f_tir.write(line)
                    elif "[Structural Type] DR" in line:
                        f_dr.write(line)
                    else:
                        f_tir.write(line)
                        f_dr.write(line)
                
                for line in debug_lines:
                    f_debug.write(line)
                
                f_tir.flush()
                f_dr.flush()
                f_debug.flush()
                
    print(f"--- Processing complete: Generated {OUT_TIR} and {OUT_DR} ---")

if __name__ == "__main__":
    main()
