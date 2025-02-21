from flexs.baselines.explorers import Encoder
import flexs
import flexs.utils.sequence_utils as s_utils
from flexs.baselines.explorers import GwgPairSampler, GWG
import pandas as pd


if __name__=="__main__":
    problem = flexs.landscapes.additive_aav_packaging.registry()['blood']
    landscape = flexs.landscapes.additive_aav_packaging.AdditiveAAVPackaging(**problem['params'])
    starting_sequence = landscape.wild_type
    alphabet = s_utils.AAS
    encoder = Encoder(alphabet)
    print(encoder.vocab_size)
    encoded = encoder.encode(alphabet)
    onehot = encoder.onehotize(encoded)
    # print(onehot)
    sampler = GwgPairSampler(encoder, 10, sequences_batch_size=10, model_queries_per_batch=10, temperature=0.1, starting_sequence=starting_sequence, alphabet=alphabet, log_file='efficiency/ggs/blood/10_10.csv')
    explorer = GWG(sampler=sampler, rounds=3, sequences_batch_size=10, model_queries_per_batch=10, temperature=0.1, starting_sequence=starting_sequence, alphabet=alphabet)
    output = sampler([starting_sequence])
    # print(output[0]['source_sequence'].drop_duplicates())
    explorer.run(landscape)