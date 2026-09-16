#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess
import time
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# ==============================================================================
# Configuration
# ==============================================================================
DETAILS_FILE = "hic_coverage_details.txt"
AF_BED_FILE = "AF_063866_6_39989_EVEs_with_flanking_with_original_ID.bed"
NC_BED_FILE = "NC_021248_239976_279815_EVEs_with_flanking_with_original_ID.bed"

# HPC Server (DDBJ/NIG) Connection Settings
SPARCON_HOST = "*@server_name.ddbj.nig.ac.jp"
SPARCON_T2T_DIR = "./Human_T2Ts"  # Relative path to T2T directory on HPC server

OUTPUT_DIR = "./EVE_filtered_bams"
TMP_BED_DIR = "./HiC_tmp_files"

# Resource Allocation Settings
CHROMAP_THREADS = 16        # Number of threads for chromap alignment
VIEW_THREADS = 4            # Number of threads for samtools view BED filtering
SORT_THREADS = 8            # Number of threads for samtools sort & compression
SORT_MEM = "1G"             # Memory per thread for samtools sort (-m)
THREADS_PER_SAMPLE = CHROMAP_THREADS + VIEW_THREADS + SORT_THREADS  # Total threads per sample (28)

def adjust_linux_limits():
    """Automatically increase file descriptor limit (ulimit -n) to hard limit."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f"[ENV] Increased file descriptor limit (ulimit -n) from {soft} to {hard}.")
    except Exception as e:
        print(f"[WARNING] Failed to adjust ulimit automatically: {e}")

def check_requirements():
    """Verify required dependencies and SSH connectivity to HPC server."""
    adjust_linux_limits()
    for cmd in ["aws", "chromap", "samtools", "python3", "ssh"]:
        if shutil.which(cmd) is None:
            print(f"Error: Required command '{cmd}' not found. Please check your PATH.")
            sys.exit(1)

    print(f"[SSH CHECK] Testing connection to HPC server ({SPARCON_HOST})...")
    res = subprocess.run(["ssh", SPARCON_HOST, "echo OK"], capture_output=True, text=True)
    if res.returncode != 0 or "OK" not in res.stdout:
        print(f"Error: SSH connection to HPC server ({SPARCON_HOST}) failed.")
        sys.exit(1)
    print(f"[SSH CHECK] SSH connection to HPC server verified.")

def parse_and_group_target_samples(details_path):
    """
    Extract target samples from details file and group tasks by sample ID.
    Prevents conflicts even if the same sample exists in both AF and NC regions.
    """
    sample_tasks = defaultdict(dict)
    current_region = None
    in_target_block = False
    af_found, nc_found = False, False

    if not os.path.exists(details_path):
        print(f"Error: File not found: {details_path}")
        sys.exit(1)

    with open(details_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()

            if line_str.startswith("---"):
                if not af_found and "[AF Region]" in line_str and "both haplotypes present" in line_str.lower():
                    current_region = "AF"
                    in_target_block = af_found = True
                elif not nc_found and "[NC Region]" in line_str and "both haplotypes present" in line_str.lower():
                    current_region = "NC"
                    in_target_block = nc_found = True
                else:
                    in_target_block = False
                continue

            if not line_str:
                in_target_block = False
                continue

            if in_target_block:
                cols = [c.strip() for c in line_str.split("\t")]
                if len(cols) >= 5 and cols[0] != "sample":
                    sample_id, status = cols[0], cols[4]
                    if status.lower() == "high":
                        if current_region == "AF":
                            sample_tasks[sample_id]["AF"] = AF_BED_FILE
                        elif current_region == "NC":
                            sample_tasks[sample_id]["NC"] = NC_BED_FILE

    return sample_tasks

def get_s3_hic_fastq_pairs_strict(sample_id):
    """Strictly pair R1 and R2 FASTQ files from S3 bucket."""
    s3_path = f"s3://human-pangenomics/working/HPRC/{sample_id}/"
    cmd = ["aws", "s3", "ls", s3_path, "--recursive", "--no-sign-request"]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error (S3 ls {sample_id}): {e}")
        return []

    all_hic_files = []
    for line in res.stdout.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        cols = line_str.split()
        if len(cols) >= 4:
            file_path = cols[3]
            file_lower = file_path.lower()
            if "hic" in file_lower and (file_lower.endswith(".fastq.gz") or file_lower.endswith(".fq.gz")):
                all_hic_files.append(f"s3://human-pangenomics/{file_path}")

    r1_files = [f for f in all_hic_files if "_1." in f or "_r1_" in f.lower() or "_r1." in f.lower() or ".1.fastq" in f.lower()]

    pairs = []
    for r1 in sorted(r1_files):
        candidates = []
        if "_1." in r1: candidates.append(r1.replace("_1.", "_2."))
        if "_r1_" in r1: candidates.append(r1.replace("_r1_", "_r2_"))
        if "_R1_" in r1: candidates.append(r1.replace("_R1_", "_R2_"))
        if "_r1." in r1: candidates.append(r1.replace("_r1.", "_r2."))
        if "_R1." in r1: candidates.append(r1.replace("_R1.", "_R2."))
        if ".1.fastq" in r1: candidates.append(r1.replace(".1.fastq", ".2.fastq"))

        matched_r2 = None
        for cand in candidates:
            if cand in all_hic_files:
                matched_r2 = cand
                break

        if matched_r2:
            pairs.append((r1, matched_r2))
        else:
            print(f"Warning ({sample_id}): Corresponding R2 not found for R1: {r1}")

    return pairs

def create_sample_eve_bed(sample_id, region_tag, bed_file_path):
    """Extract sample-specific BED regions (reformatting col 5 seq ID to col 1, cols 2-3 to coordinates)."""
    os.makedirs(TMP_BED_DIR, exist_ok=True)
    sample_bed = os.path.join(TMP_BED_DIR, f"{sample_id}_{region_tag}_eve.bed")

    if not os.path.exists(bed_file_path):
        print(f"Error: BED file not found: {bed_file_path}")
        return None

    matched_lines = []
    with open(bed_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            cols = line_str.split()
            if len(cols) >= 5:
                line_sample_id = cols[3].split(".")[0]
                if line_sample_id == sample_id:
                    seq_id = cols[4]   # Column 5: Sequence ID in FASTA header (e.g., haplotype2-0000113)
                    start = cols[1]    # Column 2: Start position
                    end = cols[2]      # Column 3: End position
                    matched_lines.append(f"{seq_id}\t{start}\t{end}")

    if not matched_lines:
        print(f"Warning: Region definition for {sample_id} ({region_tag}) not found in {bed_file_path}.")
        return None

    with open(sample_bed, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(matched_lines) + "\n")

    return sample_bed

def prepare_diploid_reference(sample_id, t2t_dir):
    """Retrieve or reuse diploid reference FASTA and Chromap index from HPC server."""
    os.makedirs(TMP_BED_DIR, exist_ok=True)
    combined_fa = os.path.join(TMP_BED_DIR, f"{sample_id}.diploid_ref.fasta")
    idx_file = f"{combined_fa}.index"

    # 1. Stream FASTA from HPC server if not present locally
    if not os.path.exists(combined_fa) or os.path.getsize(combined_fa) == 0:
        print(f"[{sample_id}] Streaming diploid reference FASTA from HPC server...")
        sp_hap1 = f"{t2t_dir}/{sample_id}.assembly.haplotype1.fasta"
        sp_hap2 = f"{t2t_dir}/{sample_id}.assembly.haplotype2.fasta"
        
        ssh_cmd = f"ssh {SPARCON_HOST} 'cat {sp_hap1} {sp_hap2}' > '{combined_fa}'"
        res = subprocess.run(ssh_cmd, shell=True)
        if res.returncode != 0 or not os.path.exists(combined_fa) or os.path.getsize(combined_fa) == 0:
            print(f"Error: Failed to stream FASTA for {sample_id}.")
            if os.path.exists(combined_fa):
                os.remove(combined_fa)
            return None

    # 2. Retrieve existing Chromap index from HPC server if not present locally
    if not os.path.exists(idx_file):
        print(f"[{sample_id}] Fetching existing Chromap index from HPC server...")
        sp_idx = f"{t2t_dir}/{sample_id}.diploid_ref.fasta.index" # Remote index path on HPC
        
        scp_cmd = f"scp {SPARCON_HOST}:'{sp_idx}' '{idx_file}'"
        res = subprocess.run(scp_cmd, shell=True)
        
        # Fallback: generate Chromap index locally if remote index retrieval fails
        if res.returncode != 0 or not os.path.exists(idx_file):
            print(f"[{sample_id}] Warning: Remote index retrieval failed. Generating Chromap index locally...")
            subprocess.run(["chromap", "-i", "-r", combined_fa, "-o", idx_file], check=True)
        else:
            print(f"[{sample_id}] Successfully retrieved index from HPC server.")
    else:
        print(f"[{sample_id}] Reusing existing local Chromap index.")

    return combined_fa

def download_s3_file_with_retry(s3_url, local_path, max_retries=3):
    """Download file from S3 bucket with retry mechanism."""
    for attempt in range(1, max_retries + 1):
        if os.path.exists(local_path):
            os.remove(local_path)
        cmd = ["aws", "s3", "cp", s3_url, local_path, "--no-sign-request"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return True
        print(f"  [Retry {attempt}/{max_retries}] S3 download failed: {s3_url}")
        if os.path.exists(local_path):
            os.remove(local_path)
    return False

def process_single_sample_job(sample_id, regions_dict):
    """
    Process a single sample: stream FASTQ pairs, map with Chromap,
    filter interactions against target EVE BED regions, and merge output BAMs.
    """
    print(f"\n=== Processing Sample: {sample_id} (Target regions: {list(regions_dict.keys())}) ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Check for incomplete target regions
    pending_regions = {}
    for region_tag, bed_file in regions_dict.items():
        out_bam = os.path.join(OUTPUT_DIR, f"{sample_id}_{region_tag}_eve_interactions.bam")
        if not os.path.exists(out_bam):
            pending_regions[region_tag] = bed_file
        else:
            print(f"[{sample_id}] Skipping completed output: {out_bam}")

    if not pending_regions:
        print(f"[{sample_id}] All BAM files completed for this sample.")
        return

    ref_fa = None
    try:
        # 2. Prepare reference FASTA and Chromap index
        ref_fa = prepare_diploid_reference(sample_id, SPARCON_T2T_DIR)
        if not ref_fa:
            return

        # 3. Fetch S3 FASTQ pairs
        fastq_pairs = get_s3_hic_fastq_pairs_strict(sample_id)
        if not fastq_pairs:
            print(f"Warning: No FASTQ pairs found for {sample_id}.")
            return

        print(f"[{sample_id}] Detected FASTQ pairs (Total {len(fastq_pairs)} pairs):")
        for p_idx, (r1, r2) in enumerate(fastq_pairs, start=1):
            print(f"  - Pair {p_idx}: R1={os.path.basename(r1)} | R2={os.path.basename(r2)}")

        # 4. Process pending regions (AF / NC)
        for region_tag, bed_file in pending_regions.items():
            sample_eve_bed = create_sample_eve_bed(sample_id, region_tag, bed_file)
            if not sample_eve_bed:
                continue

            out_bam = os.path.join(OUTPUT_DIR, f"{sample_id}_{region_tag}_eve_interactions.bam")
            tmp_bam = f"{out_bam}.tmp"
            tmp_bai = f"{tmp_bam}.bai"
            out_bai = f"{out_bam}.bai"

            work_dir = os.path.join(TMP_BED_DIR, f"work_{sample_id}_{region_tag}")
            os.makedirs(work_dir, exist_ok=True)

            pair_bams = []
            success = True

            print(f"[{sample_id}] Processing region {region_tag} by read pair...")

            try:
                for idx, (r1_s3, r2_s3) in enumerate(fastq_pairs, start=1):
                    pair_bam = os.path.join(work_dir, f"pair_{idx}.bam")

                    # Reuse existing intermediate BAM if present
                    if os.path.exists(pair_bam) and os.path.getsize(pair_bam) > 0:
                        print(f"[{sample_id}] Intermediate BAM for pair {idx}/{len(fastq_pairs)} already exists, skipping.")
                        pair_bams.append(pair_bam)
                        continue

                    print(f"[{sample_id}] Streaming mapping for pair {idx}/{len(fastq_pairs)} from S3...")
                    print(f"  R1: {r1_s3}")
                    print(f"  R2: {r2_s3}")

                    abs_ref_fa = os.path.abspath(ref_fa)
                    abs_sample_eve_bed = os.path.abspath(sample_eve_bed)
                    abs_work_dir = os.path.abspath(work_dir)
                    abs_pair_bam = os.path.abspath(pair_bam)

                    align_cmd = f"""
                    set -eo pipefail
                    cd '{abs_work_dir}'
                    chromap --preset hic -t {CHROMAP_THREADS} -r '{abs_ref_fa}' -x '{abs_ref_fa}.index' \\
                        -1 <(aws s3 cp '{r1_s3}' - --no-sign-request) \\
                        -2 <(aws s3 cp '{r2_s3}' - --no-sign-request) --SAM -o /dev/stdout | \\
                    awk -v bed_file='{abs_sample_eve_bed}' '
                        BEGIN {{
                            nr = 0;
                            while ((getline < bed_file) > 0) {{
                                if ($0 !~ /^#/ && NF >= 3) {{
                                    b_chr[nr] = $1; b_s[nr] = $2 + 0; b_e[nr] = $3 + 0; nr++;
                                }}
                            }}
                            close(bed_file);
                        }}
                        /^@/ {{ print; next }}
                        {{
                            rchr = $3; rpos = $4 + 0;
                            mchr = ($7 == "=") ? $3 : $7; mpos = $8 + 0;
                            hit = 0;
                            for (i = 0; i < nr; i++) {{
                                if (rchr == b_chr[i] && rpos >= b_s[i] && rpos <= b_e[i]) {{ hit = 1; break; }}
                                if (mchr == b_chr[i] && mpos >= b_s[i] && mpos <= b_e[i]) {{ hit = 1; break; }}
                            }}
                            if (hit) print $0;
                        }}
                    ' | \\
                    samtools view -u -@ {VIEW_THREADS} - | \\
                    samtools sort -@ {SORT_THREADS} -m {SORT_MEM} -l 1 -T 'sort_tmp' -o '{abs_pair_bam}' -
                    """
                    
                    subprocess.run(align_cmd, shell=True, executable='/bin/bash', check=True)
                    pair_bams.append(pair_bam)
                    print(f"[{sample_id}] Mapping completed for pair {idx}/{len(fastq_pairs)}.")

                if not success or not pair_bams:
                    raise RuntimeError(f"[{sample_id}] Failed to process pair for region {region_tag}.")

                # 5. Merge intermediate BAMs and index
                print(f"[{sample_id}] Merging {len(pair_bams)} intermediate BAM files for {region_tag}...")
                if len(pair_bams) == 1:
                    shutil.move(pair_bams[0], tmp_bam)
                else:
                    bam_list_str = " ".join([f"'{b}'" for b in pair_bams])
                    merge_cmd = f"samtools merge -@ {THREADS_PER_SAMPLE} -f '{tmp_bam}' {bam_list_str}"
                    subprocess.run(merge_cmd, shell=True, check=True)

                subprocess.run(f"samtools index '{tmp_bam}'", shell=True, check=True)

                # Atomic rename
                os.rename(tmp_bam, out_bam)
                if os.path.exists(tmp_bai):
                    os.rename(tmp_bai, out_bai)

                print(f"Completed (Extracted BAM): {out_bam}")

            except Exception as e:
                print(f"Error ({sample_id} - {region_tag}): {e}")
                if os.path.exists(tmp_bam):
                    os.remove(tmp_bam)
                if os.path.exists(tmp_bai):
                    os.remove(tmp_bai)
            finally:
                # Clean up temporary working directory
                if os.path.exists(work_dir):
                    shutil.rmtree(work_dir, ignore_errors=True)

    finally:
        # 6. Clean up temporary reference FASTA and Chromap index
        if ref_fa:
            idx_file = f"{ref_fa}.index"

            if os.path.exists(ref_fa):
                try:
                    os.remove(ref_fa)
                    print(f"[{sample_id}] Removed temporary FASTA: {ref_fa}")
                except Exception as e:
                    print(f"[WARNING] [{sample_id}] Failed to remove FASTA: {e}")

            if os.path.exists(idx_file):
                try:
                    os.remove(idx_file)
                    print(f"[{sample_id}] Removed Chromap index: {idx_file}")
                except Exception as e:
                    print(f"[WARNING] [{sample_id}] Failed to remove Chromap index: {e}")

        print(f"[{sample_id}] Reference FASTA / index cleanup completed.")

MAX_CONCURRENT_JOBS = 10  # Maximum number of concurrent jobs in queue

def submit_slurm_job(sample_id, task_id):
    script_file_path = f'job_hic_eve_{sample_id}.sh'
    
    with open(script_file_path, 'w', encoding='utf-8') as script_file:
        script_file.write('#!/bin/bash\n')
        script_file.write(f'#SBATCH --job-name=hic_eve_{sample_id}\n')
        script_file.write(f'#SBATCH --output=job_hic_eve_{sample_id}.out\n')
        script_file.write(f'#SBATCH --error=job_hic_eve_{sample_id}.err\n')
        script_file.write('#SBATCH --cpus-per-task=8\n')
        script_file.write('#SBATCH --mem=64G\n')
        script_file.write('#SBATCH --time=0-18:00:00\n')
        script_file.write('\n')
        script_file.write('set -x\n\n')
        script_file.write('ulimit -n $(ulimit -Hn)\n\n')
        
        script = f'python3 Stream_hic_EVE_interactions_ver2.py {task_id}\n'
        script_file.write(script)

    cmd = f'sbatch {script_file_path}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"[{sample_id}] sbatch output: {result.stdout.strip()}")

    match = re.search(r'Submitted batch job (\d+)', result.stdout)
    if match:
        return int(match.group(1))
    else:
        print(f"[{sample_id}] Error: Job submission failed.")
        return None

def filter_active_jobs(active_job_ids):
    """Filter and return currently running or pending job IDs."""
    if not active_job_ids:
        return []
    
    still_active = []
    # Query squeue in a single call to minimize load
    squeue_cmd = f"squeue -j {','.join(map(str, active_job_ids))}"
    result = subprocess.run(squeue_cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        for job_id in active_job_ids:
            if str(job_id) in result.stdout:
                still_active.append(job_id)
    return still_active

def wait_for_all_jobs(active_job_ids):
    """Wait for all submitted Slurm jobs to complete."""
    print(f"\n--- All jobs submitted. Monitoring remaining {len(active_job_ids)} jobs ---")
    while active_job_ids:
        active_job_ids = filter_active_jobs(active_job_ids)
        if active_job_ids:
            time.sleep(10)
    print("--- All jobs completed successfully ---\n")

def main():
    check_requirements()

    print("1. Parsing target sample definitions...")
    sample_tasks = parse_and_group_target_samples(DETAILS_FILE)

    sorted_sample_ids = sorted(sample_tasks.keys())
    total_samples = len(sorted_sample_ids)
    print(f"   Total unique target samples: {total_samples}")

    if total_samples == 0:
        print("No matching high-coverage samples found.")
        return

    print("\n2. Starting sequential sample processing...")
    
    for idx, target_sample in enumerate(sorted_sample_ids, start=1):
        print(f"\n==================================================")
        print(f"   Progress: [{idx}/{total_samples}] Processing sample '{target_sample}'")
        print(f"==================================================")
        
        # Process sequentially per sample
        process_single_sample_job(target_sample, sample_tasks[target_sample])

    print("\nAll sample processing completed successfully.")

if __name__ == "__main__":
    main()
