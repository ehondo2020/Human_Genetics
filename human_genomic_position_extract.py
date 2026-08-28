import os
import re

def process_txt_file(viral_name):
    input_file = "./input_file" #output_file from fasta36_data_extract
    output_dir = "./output_dir" #Specify output directory
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = f"{output_dir}/Genome_position_list_{viral_name}.txt"

    viral_name_lower = viral_name.lower()

    with open(input_file, 'r') as file, open(output_file, 'w') as genome_position_list:
        lines = file.readlines()
        
        dataset_a = []
        inside_dataset_a = False
        for i, line in enumerate(lines):
            if line.startswith("Query= "):
                if inside_dataset_a and viral_name_lower in ''.join(dataset_a).lower():
                    process_dataset_a(dataset_a, viral_name_lower, genome_position_list)
                dataset_a = [line]
                inside_dataset_a = True
            else:
                dataset_a.append(line)
        
        if inside_dataset_a and viral_name_lower in ''.join(dataset_a).lower():
            process_dataset_a(dataset_a, viral_name_lower, genome_position_list)


def process_dataset_a(dataset_a, viral_name, output_file):
    query_line = [line for line in dataset_a if line.startswith("Query= ")][0]
    query_info = query_line.split("|")[0].replace("Query= ", "").strip()
    
    output_file.write(">" + query_info + "\n")
    
    for line in dataset_a:
        if line.startswith(">"):
            match = re.search(r"chromosome (\d+|X|Y)", line, re.IGNORECASE)
            if match:
                chromosome = f"chr{match.group(1)}"
                process_dataset_b(dataset_a, chromosome, output_file)


def process_dataset_b(dataset_a, chromosome, output_file):
    dataset_b = []
    inside_dataset_b = False
    for line in dataset_a:
        if line.startswith(" Score = "):
            if inside_dataset_b:
                process_dataset_b_lines(dataset_b, chromosome, output_file)
            dataset_b = [line]
            inside_dataset_b = True
        elif line.startswith("Query= "):
            if inside_dataset_b:
                process_dataset_b_lines(dataset_b, chromosome, output_file)
            inside_dataset_b = False
        else:
            dataset_b.append(line)

    if inside_dataset_b:
        process_dataset_b_lines(dataset_b, chromosome, output_file)


def process_dataset_b_lines(dataset_b, chromosome, output_file):
    sbjct_lines = [line for line in dataset_b if line.startswith("Sbjct ")]
    if not sbjct_lines:
        return

    first_sbjct_line = sbjct_lines[0] 
    last_sbjct_line = sbjct_lines[-1] 

    sbjct_start = first_sbjct_line.split()[1]

    sbjct_end = last_sbjct_line.split()[-1]

    result = f"{chromosome}:{sbjct_start}-{sbjct_end}"
    output_file.write(result + "\n")


with open("./viral_name_list.txt", 'r') as viral_file: #keyword file (viral name list from input_file by fasta36_data_extract.py)
    viral_names = [line.strip() for line in viral_file]

for viral_name in viral_names:
    process_txt_file(viral_name)
