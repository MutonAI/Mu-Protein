import random
import numpy as np
import flexs
import torch
import torch.nn.functional as F
import numpy as np

def hamming_distance(seq_1, seq_2):
    return sum([x!=y for x, y in zip(seq_1, seq_2)])

def random_mutation(sequence, alphabet, num_mutations):
    wt_seq = list(sequence)
    # print(f"pex.py line 13 num_mutations: {num_mutations}")
    for _ in range(num_mutations):
        # print(f"pex.py line 14 len(sequence): {len(sequence)}")
        idx = np.random.randint(len(sequence))
        wt_seq[idx] = alphabet[np.random.randint(len(alphabet))]
    new_seq = ''.join(wt_seq)
    return new_seq

def sequence_to_one_hot(sequence, alphabet):
    # Input:  - sequence: [sequence_length]
    #         - alphabet: [alphabet_size]
    # Output: - one_hot:  [sequence_length, alphabet_size]
    
    alphabet_dict = {x: idx for idx, x in enumerate(alphabet)}
    one_hot = F.one_hot(torch.tensor([alphabet_dict[x] for x in sequence]).long(), num_classes=len(alphabet))
    return one_hot

def sequences_to_tensor(sequences, alphabet):
    # Input:  - sequences: [batch_size, sequence_length]
    #         - alphabet:  [alphabet_size]
    # Output: - one_hots:  [batch_size, alphabet_size, sequence_length]
    
    one_hots = torch.stack([sequence_to_one_hot(seq, alphabet) for seq in sequences], dim=0)
    one_hots = torch.permute(one_hots, [0, 2, 1]).float()
    return one_hots

def sequences_to_mutation_sets(sequences, alphabet, wt_sequence, context_radius):
    # Input:  - sequences:          [batch_size, sequence_length]
    #         - alphabet:           [alphabet_size]
    #         - wt_sequence:        [sequence_length]
    #         - context_radius:     integer
    # Output: - mutation_sets:      [batch_size, max_mutation_num, alphabet_size, 2*context_radius+1]
    #         - mutation_sets_mask: [batch_size, max_mutation_num]

    context_width = 2 * context_radius + 1
    max_mutation_num = max(1, np.max([hamming_distance(seq, wt_sequence) for seq in sequences]))
    
    mutation_set_List, mutation_set_mask_List = [], []
    for seq in sequences:
        one_hot = sequence_to_one_hot(seq, alphabet).numpy()
        one_hot_padded = np.pad(one_hot, ((context_radius, context_radius), (0, 0)), mode='constant', constant_values=0.0)
        
        mutation_set = [one_hot_padded[i:i+context_width] for i in range(len(seq)) if seq[i]!=wt_sequence[i]]
        padding_len = max_mutation_num - len(mutation_set)
        mutation_set_mask = [1.0] * len(mutation_set) + [0.0] * padding_len
        mutation_set += [np.zeros(shape=(context_width, len(alphabet)))] * padding_len
            
        mutation_set_List.append(mutation_set)
        mutation_set_mask_List.append(mutation_set_mask)
    
    mutation_sets = torch.tensor(np.array(mutation_set_List)).permute([0, 1, 3, 2]).float()
    mutation_sets_mask = torch.tensor(np.array(mutation_set_mask_List)).float()
    return mutation_sets, mutation_sets_mask

class ProximalExploration(flexs.Explorer):
    """
        Proximal Exploration (PEX)
    """
    
    def __init__(self, args, model, rounds, alphabet, starting_sequence, log_file=None):
        self.model = model
        name = "ProximalExploration"
        super().__init__(
            model, 
            name, 
            rounds, 
            args.num_queries_per_round, 
            args.batch_size, 
            starting_sequence, 
            log_file
        )
        self.alphabet = alphabet
        self.wt_sequence = starting_sequence
        self.num_queries_per_round = args.num_queries_per_round
        self.num_model_queries_per_round = args.num_model_queries_per_round
        self.batch_size = args.batch_size
        self.num_random_mutations = args.num_random_mutations
        self.frontier_neighbor_size = args.frontier_neighbor_size
    
    def propose_sequences(self, measured_sequences):
        # Input:  - measured_sequences: pandas.DataFrame
        #           - 'sequence':       [sequence_length]
        #           - 'true_score':     float
        # Output: - query_batch:        [num_queries, sequence_length]
        #         - model_scores:       [num_queries]
        
        # print("-----------------ProximalExploration-----------------")
        # print(f"measured_sequences", measured_sequences)
        # print("-----------------ProximalExploration-----------------")
        query_batch = self._propose_sequences(measured_sequences)
        # assert len(query_batch) == 2, f"query_batch: {type(query_batch[0])}"
        # print(f"query_batch: {query_batch[0]}")
        # model_scores = np.concatenate([
        #     self.model.get_fitness(query_batch[0][i:i+self.batch_size])
        #     for i in range(0, len(query_batch[0]), self.batch_size)
        # ])
        return query_batch[0], query_batch[1]

    def _propose_sequences(self, measured_sequences):
        measured_sequence_set = set(measured_sequences['sequence'])
        
        # Generate random mutations in the first round.
        if len(measured_sequence_set)==1:
            query_batch = []
            scores = []
            while len(query_batch) < self.num_queries_per_round:
                random_mutant = random_mutation(self.wt_sequence, self.alphabet, self.num_random_mutations)
                if random_mutant not in measured_sequence_set:
                    query_batch.append(random_mutant)
                    measured_sequence_set.add(random_mutant)
                    scores.append(np.nan)
            return query_batch, scores
        
        # Arrange measured sequences by the distance to the wild type.
        measured_sequence_dict = {}
        for _, data in measured_sequences.iterrows():
            distance_to_wt = hamming_distance(data['sequence'], self.wt_sequence)
            if distance_to_wt not in measured_sequence_dict.keys():
                measured_sequence_dict[distance_to_wt] = []
            measured_sequence_dict[distance_to_wt].append(data)
        
        # Highlight measured sequences near the proximal frontier.
        frontier_neighbors, frontier_height = [], -np.inf
        for distance_to_wt in sorted(measured_sequence_dict.keys()):
            data_list = measured_sequence_dict[distance_to_wt]
            data_list.sort(reverse=True, key=lambda x:x['true_score'])
            for data in data_list[:self.frontier_neighbor_size]:
                if data['true_score'] > frontier_height:
                    frontier_neighbors.append(data)
            frontier_height = max(frontier_height, data_list[0]['true_score'])

        # Construct the candiate pool by randomly mutating the sequences. (line 2 of Algorithm 2 in the paper)
        # An implementation heuristics: only mutating sequences near the proximal frontier.
        candidate_pool = []
        def n_choose_k(n, k):
            if k > n:
                return 0
            result = 1
            for i in range(k):
                result *= (n - i)
                result //= (i + 1)
            return result

        model_query_per_round = min(self.num_model_queries_per_round, 
                                   (len(self.alphabet) - 1) ** self.num_random_mutations * n_choose_k(len(random.choice(frontier_neighbors)['sequence']), self.num_random_mutations))
        while len(candidate_pool) < model_query_per_round:
            print(len(candidate_pool))
            candidate_sequence = random_mutation(random.choice(frontier_neighbors)['sequence'], self.alphabet, self.num_random_mutations)
            print(f"candidate_sequence in measured_sequence_set: {candidate_sequence in measured_sequence_set}")
            if candidate_sequence not in measured_sequence_set:
                candidate_pool.append(candidate_sequence)
                measured_sequence_set.add(candidate_sequence)
        
        # Arrange the candidate pool by the distance to the wild type.
        candidate_pool_dict = {}
        for i in range(0, len(candidate_pool), self.batch_size):
            candidate_batch =  candidate_pool[i:i+self.batch_size]
            model_scores = self.model.get_fitness(candidate_batch)
            for candidate, model_score in zip(candidate_batch, model_scores):
                distance_to_wt = hamming_distance(candidate, self.wt_sequence)
                if distance_to_wt not in candidate_pool_dict.keys():
                    candidate_pool_dict[distance_to_wt] = []
                candidate_pool_dict[distance_to_wt].append(dict(sequence=candidate, model_score=model_score))
        for distance_to_wt in sorted(candidate_pool_dict.keys()):
            candidate_pool_dict[distance_to_wt].sort(reverse=True, key=lambda x:x['model_score'])
        
        # Construct the query batch by iteratively extracting the proximal frontier. 
        query_batch = []
        scores = []
        while len(query_batch) < self.num_queries_per_round:
            # Compute the proximal frontier by Andrew's monotone chain convex hull algorithm. (line 5 of Algorithm 2 in the paper)
            # https://en.wikibooks.org/wiki/Algorithm_Implementation/Geometry/Convex_hull/Monotone_chain
            stack = []
            for distance_to_wt in sorted(candidate_pool_dict.keys()):
                if len(candidate_pool_dict[distance_to_wt])>0:
                    data = candidate_pool_dict[distance_to_wt][0]
                    new_point = np.array([distance_to_wt, data['model_score']])
                    def check_convex_hull(point_1, point_2, point_3):
                        return np.cross(point_2-point_1, point_3-point_1) <= 0
                    while len(stack)>1 and not check_convex_hull(stack[-2], stack[-1], new_point):
                        stack.pop(-1)
                    stack.append(new_point)
            while len(stack)>=2 and stack[-1][1] < stack[-2][1]:
                stack.pop(-1)
            
            # Update query batch and candidate pool. (line 6 of Algorithm 2 in the paper)
            for distance_to_wt, model_score in stack:
                if len(query_batch) < self.num_queries_per_round:
                    query_batch.append(candidate_pool_dict[distance_to_wt][0]['sequence'])
                    candidate_pool_dict[distance_to_wt].pop(0)
                    scores.append(model_score)
        print(f"query_batch: {query_batch}")
        print(f"scores: {scores}")

        return query_batch, scores