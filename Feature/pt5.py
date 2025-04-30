import torch
from transformers import T5EncoderModel, T5Tokenizer
import re  
import os
import requests
from tqdm.auto import tqdm
import tqdm
from Bio import SeqIO
import gc 
import pandas as pd
import numpy as np
#
local_model_path = r"D:\Rostlab\prot_t5_xl_uniref50"
tokenizer = T5Tokenizer.from_pretrained(local_model_path, do_lower_case=False)
model = T5EncoderModel.from_pretrained(local_model_path)
# model = model.half()
gc.collect()
device = torch.device('cpu')
model = model.to(device)
model = model.eval()
def find_features_full_seq(sequences_Example):
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]
    # add space in between amino acids
    sequence = [' '.join(sequences_Example)]
    ids = tokenizer.batch_encode_plus(sequence, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding = model(input_ids=input_ids,attention_mask=attention_mask)
    embedding = embedding.last_hidden_state.cpu()

    seq_len = (attention_mask[0] == 1).sum()
    seq_emd = embedding[0][:seq_len-1]#
    return seq_emd

from Bio import SeqIO
# input_fasta_file=r"negative.fasta"
input_fasta_file=r"positive.fasta"

from tqdm import tqdm
per_protein_list = []
for seq_record in tqdm(SeqIO.parse(input_fasta_file, "fasta")):
    prot_id = seq_record.id
    sequence = str(seq_record.seq)
    # extract protT5 for full sequence and store in temporary dataframe
    pt5_all = find_features_full_seq(sequence)
    print(pt5_all.shape)#torch.Size([33, 1024])
    per_protein1 = pt5_all.mean(dim=0)
    per_protein=per_protein1.numpy().reshape(1, -1)
    per_protein_list.append(per_protein)

final_result = np.vstack(per_protein_list)
print(final_result)
output_file = r'pt5_pos_40.csv'  
np.savetxt(output_file, final_result, delimiter=',')
