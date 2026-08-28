import re

def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        longest_block = []
        for line in infile:
            if line.startswith("Longest range "):
                longest_block.append(line)
            elif line.startswith("\t"):
                longest_block.append(line)
            else:
                if longest_block:
                    process_longest_block(longest_block, outfile)
                    longest_block = []
                outfile.write(line)

        if longest_block:
            process_longest_block(longest_block, outfile)

def process_longest_block(block, outfile):
    min_start, max_end = float('inf'), 0
    chr_info = None

    for line in block:
        match = re.search(r"(chr\w+)\s+(\d+)\s+(\d+)", line)
        if match:
            chr_info, start, end = match.groups()
            start, end = int(start), int(end)
            min_start = min(min_start, start)
            max_end = max(max_end, end)

    total_range_length = max_end - min_start

    if chr_info:
        new_first_line = re.sub(r"(chr\w+)\s+\d+\s+\d+", f"{chr_info} {min_start} {max_end}", block[0])
        new_first_line = new_first_line.strip() + f"  Total range length: {total_range_length}\n"
        outfile.write(new_first_line)
    
    for line in block[1:]:
        outfile.write(line)

input_file = './output.txt' #made from make_single_range_1.py
output_file = './suppl_table.txt'
process_file(input_file, output_file)