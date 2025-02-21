import time
import numpy as np
import pandas as pd


def sequence_to_mutation(sequence, starting_sequence):
    '''
    Parameters
    ----------
    sequence: str
    return: ';'.join(WiM) (wide-type W at position i mutated to M)
    '''
    mutant_points = []
    assert len(sequence) == len(starting_sequence)
    for i in range(len(sequence)):
        if sequence[i] != starting_sequence[i]:
            mutant_points += ["%s%s%s"%(starting_sequence[i], i+1, sequence[i])]
    return ';'.join(mutant_points), len(mutant_points)


def print_and_save_sequences(sequences, starting_sequence, save_name="results", save_json=True):
    # sequences: dict, key: sequence, value: [score, embedding, ensemble_uncertainty]
    # results: list, [mutation, score, ensemble_uncertainty, embedding]

    from collections import Counter
    mutant_sites_counter = Counter()

    results = []
    for sequence, value in sequences.items():
        mutation, mutant_sites_num = sequence_to_mutation(sequence, starting_sequence)
        results.append([mutation, value[0], value[1]])
        mutant_sites_counter[mutant_sites_num] += 1

    sorted_results = sorted(results, key=lambda x:x[1], reverse=True)
    print(mutant_sites_counter)

    with open(save_name + ".txt", "w") as f:
        for item in sorted_results:
            f.write("{0} {1}\n".format(item[0], item[1]))

    df = pd.DataFrame(sorted_results, columns=['mutation', 'score', 'embedding'])

    if save_json:
        df.to_json(save_name + ".json", orient='records', lines=True)