from pathlib import Path

# Directory setup
input_dir = Path("./Extracted_HiC_EVE_only_results")
output_dir = input_dir / "EVE_counterparts"
output_dir.mkdir(parents=True, exist_ok=True)

# Output paths for combined files
af_combined_file = output_dir / "EVE_counterparts_AF.fa"
nc_combined_file = output_dir / "EVE_counterparts_NC.fa"

with open(af_combined_file, "w", encoding="utf-8") as f_af, open(nc_combined_file, "w", encoding="utf-8") as f_nc:
    
    for bed_file in sorted(input_dir.glob("*_Other.bed")):
        filename = bed_file.name
        parts = filename.split("_")
        
        # Determine category (AF/NC) from filename
        category = None
        if len(parts) >= 2:
            target_str = parts[1]
            if target_str == "AF":
                category = "AF"
            elif target_str == "NC":
                category = "NC"
            else:
                print(f"Info: Target string '{target_str}' in {filename} does not match 'AF' or 'NC'.")

        if len(parts) >= 3:
            prefix = "_".join(parts[:3])
        else:
            prefix = bed_file.stem
            
        output_file = output_dir / f"{prefix}_counterparts.fa"
        
        # Process BED file entries and write FASTA records
        with open(bed_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                cols = line.split("\t")
                
                if len(cols) >= 16:
                    # Extract header fields (cols 6, 9, 10, 13, 14) and sequence (col 16)
                    header_cols = [cols[5], cols[8], cols[9], cols[12], cols[13]]
                    header = ">" + "_".join(header_cols)
                    sequence = cols[15]
                    
                    fasta_entry = f"{header}\n{sequence}\n"
                    
                    f_out.write(fasta_entry)
                    
                    if category == "AF":
                        f_af.write(fasta_entry)
                    elif category == "NC":
                        f_nc.write(fasta_entry)
                else:
                    print(f"Warning: Skipped line {line_num} in {filename} due to insufficient columns.")

print("All individual FASTA files and combined files (AF / NC) have been processed successfully.")