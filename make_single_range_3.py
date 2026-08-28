import re

input_file = './suppl_table2.txt'
output_file = './suppl_table1.txt'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        if line.startswith("Longest range "):
            range_match = re.search(r'(chr\w+ \d+ \d+)', line)
            file_match = re.search(r'File: (.+?)\)', line)
            
            if range_match and file_match:
                file_name = file_match.group(1).replace(" ", "_") + "_L"
                output_line = f"{range_match.group(1)} {file_name}\n"
                outfile.write(output_line)

        elif line.startswith("Unique range "):
            range_match = re.search(r'(chr\w+ \d+ \d+)', line)
            file_match = re.search(r'File: (.+?)\)', line)
            
            if range_match and file_match:
                file_name = file_match.group(1).replace(" ", "_") + "_U"
                output_line = f"{range_match.group(1)} {file_name}\n"
                outfile.write(output_line)