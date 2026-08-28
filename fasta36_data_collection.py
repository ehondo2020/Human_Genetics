import os
import re


input_dir = './Input_dir' #Specify directory name
output_file_data = './output_file.txt' #Specify output file name
output_file_query = './output_info.txt' #Specify output info name

with open(output_file_data, 'w') as out_data, open(output_file_query, 'w') as out_query:
    pass

query_pattern = re.compile(r'^Query= ')

identities_pattern = re.compile(r'Identities = (\d+)/(\d+) \((\d+)%\), Gaps = (\d+)/(\d+) \((\d+)%\)') #重要！！！この行は形式が合っていないとマッチしないので、特にアミノ酸配列デーがある場合には、Positives = (\d+)/(\d+) \((\d+)%\)をIdentitiesとgapの間に挿入すること。

for file_name in os.listdir(input_dir):
    if file_name.endswith('.txt'):
        file_path = os.path.join(input_dir, file_name)

        with open(file_path, 'r') as f:
            dataset = []
            write_dataset = False
            identities_values = None  

            for line in f:
                
                if query_pattern.match(line):
                    if dataset and write_dataset:
                        
                        with open(output_file_data, 'a') as out_data:
                            out_data.writelines(dataset)
                        
                        with open(output_file_query, 'a') as out_query:
                            out_query.write(f"{dataset[0].strip()}    Length: {identities_values[0]}bps    Homology: {identities_values[1]}％\n")
                    
                    dataset = [line]
                    write_dataset = False  
                    identities_values = None  
                else:
                    dataset.append(line)

                match = identities_pattern.search(line)
                if match:
                    Identities = list(map(int, match.groups()))  
                    if Identities[1] >= 5000 and Identities[2] >= 57: #Specify threshold
                        write_dataset = True  
                        identities_values = (Identities[1], Identities[2])  

            if dataset and write_dataset:
                with open(output_file_data, 'a') as out_data:
                    out_data.writelines(dataset)
                with open(output_file_query, 'a') as out_query:
                    out_query.write(f"{dataset[0].strip()}    Length: {identities_values[0]}bps    Homology: {identities_values[1]}％\n")

