import numpy as np
import sys, os,re,platform,math
def meiler_feature(fastas):
    meiler = {
    'A': [1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23],
    'C': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
    'D': [1.60, 0.11, 2.78, -0.77, 2.95, 0.25, 0.20],
    'E': [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
    'F': [2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38],
    'G': [0.00, 0.00, 0.00, 0.00, 6.07, 0.13, 0.15],
    'H': [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30],
    'I': [4.19, 0.19, 4.00, 1.80, 6.04, 0.30, 0.45],
    'K': [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
    'L': [2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31],
    'M': [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
    'N': [1.60, 0.13, 2.95, -0.60, 6.52, 0.21, 0.22],
    'P': [2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34],
    'Q': [1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25],
    'R': [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
    'S': [1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
    'T': [3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
    'V': [3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49],
    'W': [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
    'Y': [2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41],
    '-': [0, 0, 0, 0, 0, 0, 0] # -
    }
    encodings = []
    for i in fastas:
        name, sequence = i[0], i[1]
        code = []
        for aa in sequence:
            code = code + meiler[aa]
        encodings.append(code)
    B = np.array(encodings)
    print(B.shape)
    return B
import pandas as pd
from Bio import SeqIO

sequences = []
for record in SeqIO.parse(fasta_file_path, 'fasta'):
    data = record.seq
    data = re.sub('[^ARNDCQEGHILKMFPSTWYV]', '-', ''.join(data.upper()))
    sequences.append((record.id, str(data)))
for sequence in sequences:
    sequence_name = sequence[0]
    sequence_data = sequence[1]
    sequence_length = len(sequence_data)
    if sequence_length !=33:
        print("Found sequence with length 0!")
        print("Sequence Name:", sequence_name)
        print("Sequence Data:", sequence_data)
        print("-------------------")

meiler_feature =meiler_feature(sequences)
print(meiler_feature.shape)

csv_file_path = r'meiler_neg.csv'#41223
np.savetxt(csv_file_path,meiler_feature, delimiter=',')

