import os

bed_directory = './Genome_position_list_bed' #Prepared bed files are under this directory
output_file = './output.txt'

chromosome_data = {}

for filename in os.listdir(bed_directory):
    if filename.endswith('.bed'):
        parts = filename.split('_')
        if len(parts) >= 4:
            string_a = parts[3]

            with open(os.path.join(bed_directory, filename), 'r') as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        chromosome = parts[0]
                        start = int(parts[1])
                        end = int(parts[2])
                        length = end - start

                        if chromosome not in chromosome_data:
                            chromosome_data[chromosome] = []
                        chromosome_data[chromosome].append((string_a, start, end, length, line.strip()))

with open(output_file, 'w') as output:
    for chromosome, ranges in chromosome_data.items():
        non_overlapping_groups = []
        current_group = []

        ranges.sort(key=lambda x: x[1])  

        for string_a, start, end, length, line in ranges:
            if not current_group:
                current_group.append((string_a, start, end, length, line))
            else:
                last_start = current_group[-1][2]
                if start <= last_start:
                    current_group.append((string_a, start, end, length, line))
                else:
                    non_overlapping_groups.append(current_group)
                    current_group = [(string_a, start, end, length, line)]

        if current_group:
            non_overlapping_groups.append(current_group)

        for group in non_overlapping_groups:
            if len(group) == 1:
                string_a, start, end, length, line = group[0]
                output.write(f"Unique range for {chromosome}: {line} (File: {string_a}) - Length: {length} bp\n\n")
            else:
                group.sort(key=lambda x: x[3], reverse=True)
                longest_range = group[0]
                longest_string_a, longest_start, longest_end, longest_length, longest_line = longest_range

                output.write(f"Longest range in group: {longest_line} (File: {longest_string_a}) - Length: {longest_length} bp\n")

                output.write("\tShorter ranges in this group:\n")
                for string_a, start, end, length, line in group:
                    if line != longest_line:
                        overlap_length = min(longest_end, end) - max(longest_start, start)
                        overlap_percentage = (overlap_length / length) * 100 if length > 0 else 0
                        output.write(f"\t\t{string_a}: {line} - Length: {length} bp, Overlap: {overlap_length} bp ({overlap_percentage:.2f}%)\n")

                output.write("\n")