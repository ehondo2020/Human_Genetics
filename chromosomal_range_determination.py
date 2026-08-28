import os
import re
from collections import defaultdict

def process_genome_positions(input_file, output_file, fasta_file):
    chrom_ranges = defaultdict(list)
    chrom_headers = defaultdict(list)
    chrom_original_ranges = defaultdict(list)
    chrom_total_lengths = defaultdict(int)

    with open(input_file, 'r') as infile:
        current_header = ""
        for line in infile:
            line = line.strip()

            if line.startswith(">"):
                current_header = line
            
            elif re.match(r'chr\w+:\d+-\d+', line):
                chrom, positions = line.split(':')
                start, end = map(int, positions.split('-'))

                start, end = min(start, end), max(start, end)

                chrom_ranges[chrom].append((start, end))
                chrom_headers[chrom].append(current_header)
                chrom_original_ranges[chrom].append((start, end)) 

    
    def chromosome_key(chrom):
        match = re.search(r'\d+', chrom)
        if match:
            return (int(match.group()), "") 
        elif chrom.lower() == 'chrx':
            return (1000, 'X') 
        elif chrom.lower() == 'chry':
            return (1001, 'Y')  
        else:
            return (9999, chrom)  

    sorted_chromosomes = sorted(chrom_ranges.keys(), key=chromosome_key)

    with open(output_file, 'w') as outfile:
        total_length = 0

        for chrom in sorted_chromosomes:
            merged_ranges = merge_ranges(chrom_ranges[chrom])

            chrom_length = 0
            for start, end in merged_ranges:
                length = end - start + 1
                chrom_length += length
                total_length += length

                outfile.write(f"{chrom}:{start}-{end} (Length: {format_number(length)} bases)\n")
                
                unique_headers = set(chrom_headers[chrom])
                for header in unique_headers:
                    outfile.write(f"{header}\n")
                
                outfile.write("Original ranges used for this range:\n")
                for orig_start, orig_end in chrom_original_ranges[chrom]:
                    if orig_start >= start and orig_end <= end:
                        outfile.write(f"  {chrom}:{orig_start}-{orig_end}\n")
                
                outfile.write("\n") 

            outfile.write(f"Total length for {chrom}: {format_number(chrom_length)} bases\n\n")

        outfile.write(f"Overall total length: {format_number(total_length)} bases\n")
        
        fasta_length = calculate_fasta_length(fasta_file)
        outfile.write(f"Total length of {fasta_file}: {format_number(fasta_length)} bases\n")

        percentage = (total_length / fasta_length) * 100 if fasta_length > 0 else 0
        outfile.write(f"Percentage of total genome covered: {format_number(total_length)} / {format_number(fasta_length)} ({percentage:.2f}%)\n")

def merge_ranges(ranges):
    sorted_ranges = sorted(ranges)
    merged = []

    current_start, current_end = sorted_ranges[0]

    for start, end in sorted_ranges[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))
    return merged

def format_number(number):
    return "{:,}".format(number)

def calculate_fasta_length(fasta_file):
    total_length = 0
    with open(fasta_file, 'r') as f:
        for line in f:
            if not line.startswith('>'):
                total_length += len(line.strip())
    return total_length

def process_directory(input_dir, output_dir, fasta_file):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, filename.replace('.txt', '_rev.txt'))
            print(f"Processing {input_file} and writing to {output_file}")
            process_genome_positions(input_file, output_file, fasta_file)

input_dir = './input_dir' #specify input_dir made by human_genomic_position_extract.py
output_dir = './output_dir'
fasta_file = './T2T-CHM13v2.0_genomic.fa' #Use hT2T genome data downloaded from NCBI
process_directory(input_dir, output_dir, fasta_file)
