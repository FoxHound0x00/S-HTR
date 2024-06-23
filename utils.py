import os
import numpy as np

char_list = " -ँंःअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसह़ऽािीुूृॄॅेैॉॊोौ्ॐ॒॑॓॔क़ख़ग़ज़ड़ढ़फ़य़ॠॢ।॥०१२३४५६७८९॰ॱॲॻॼॽॾ≈–|"


def calc_class_weights(tau=10, dataset_dir=None):

    files = os.listdir(dataset_dir)

    char_dict = {}
    char_dict = {x:y for x, y in enumerate(char_list)}
    char_count = {x:0 for x in char_list}

    for f_ in files:
        if f_.endswith('.txt'):
            f = open(os.path.join(dataset_dir,f_), 'r')
            for j in f.read():
                char_count[j] = int(char_count[j]) + 1

    char_filtered_dict = {k:v for k, v in char_count.items() if v != 0}

    
    weight_ = {k:np.round(tau*(1/(x+1e-10)), 3) for k, x in char_filtered_dict.items()}
    
    return weight_


def get_max_sequence_length(dataset_dir=None):
    
    files = os.listdir(dataset_dir)
    lengths = []
    for x in files:
        if x.endswith('.txt'):
            with open(os.path.join(dataset_dir,x), "r") as f:
                lengths.append(len(f.read()))
    print(f"Longest Sequence : {max(lengths)}")
    return max(lengths)

def levenshtein_distance(str1, str2):
    len_str1 = len(str1) + 1
    len_str2 = len(str2) + 1

    matrix = [[0] * len_str2 for _ in range(len_str1)]

    for i in range(len_str1):
        matrix[i][0] = i
    for j in range(len_str2):
        matrix[0][j] = j

    # Calculate dist
    for i in range(1, len_str1):
        for j in range(1, len_str2):
            cost = 0 if str1[i - 1] == str2[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,  # Deletion
                matrix[i][j - 1] + 1,  # Insertion
                matrix[i - 1][j - 1] + cost,  # Substitution
            )

    return matrix[len_str1 - 1][len_str2 - 1]


def get_null_annot(dataset_dir=None):

    null_annot = []
    less_annot = []
    char_count = {x:0 for x in char_list}
    
    for f_ in os.listdir(dataset_dir):
        if f_.endswith('.txt'):
            f = open(os.path.join(dataset_dir,f_), 'r')
            for j in f.read():
                char_count[j] = int(char_count[j]) + 1
                
    for k, v in char_count.items():
        if v == 0:
            null_annot.append(k)
        if v != 0 and v < 5:
            less_annot.append(k)
    
    return null_annot, less_annot


    
