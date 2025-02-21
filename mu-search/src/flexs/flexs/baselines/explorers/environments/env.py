import numpy as np
from functools import reduce
from gym import Env
from gym.spaces import MultiDiscrete


class LandscapeEnv(Env):
    """ RL environment for protein optimization. """
    def __init__(self, model, alphabet, horizon, starting_seq, for_rna=True) -> None:
        super().__init__()
        self.model = model
        self.horizon = horizon
        self.starting_seq = starting_seq
        self.alphabet = alphabet
        self.char2int = {}
        for i, a in enumerate(alphabet):
            self.char2int[a] = i
        self.array2str = lambda arr: reduce(lambda x,y:x+y, map(lambda x:self.alphabet[x], arr))
        self.str2array = lambda s: np.array(list(map(lambda x: self.char2int[x], s)))

        num_alphabets = len(alphabet)
        sequence_length = len(self.starting_seq)

        observation_space_list = [num_alphabets] * 2 * sequence_length
        action_space_list = [sequence_length, num_alphabets]
        observation_space1_list = observation_space_list
        observation_space2_list = observation_space1_list + [action_space_list[0]]
        action_space1_list = action_space_list[:1]
        action_space2_list = action_space_list[1:]

        self.observation_space = MultiDiscrete(observation_space_list)
        self.action_space = MultiDiscrete(action_space_list)
        self.observation_space1 = MultiDiscrete(observation_space1_list)
        self.observation_space2 = MultiDiscrete(observation_space2_list)
        self.action_space1 = MultiDiscrete(action_space1_list)
        self.action_space2 = MultiDiscrete(action_space2_list)

        self.reset()
   
    def step(self, action):
        loc = action[0]
        mutate_to = action[1]
        self.cur_mutations += 1
        self.cur_seq = self.cur_seq[:loc] + self.alphabet[mutate_to] + self.cur_seq[loc+1:]

        done = self.cur_mutations == self.horizon
        # print(f"wild type: {self.starting_seq}")
        # print(f"wild type fitness: {self.model.get_fitness([self.starting_seq])}")
        # print(f"self.cur_seq: {self.cur_seq}, fitness: {self.model.get_fitness([self.cur_seq])}, mutations: {self.cur_mutations}, done: {done}")
        if done:
            # try:
            #     reward, embedding = self.model.get_fitness([self.cur_seq])
            #     return self.obs, reward[0], done, {"current_sequence": self.str2array(self.cur_seq), "embedding": embedding[0], "ensemble_uncertainty": ensemble_uncertainty[0]}            
            # except:
            reward = self.model.get_fitness([self.cur_seq])
            embedding = None
            ensemble_uncertainty = None
            return self.obs, reward[0], done, {"current_sequence": self.str2array(self.cur_seq), "embedding": None, "ensemble_uncertainty": None}
        else:
            return self.obs, 0, done, {}
        
    def reset(self):
        self.cur_mutations = 0
        self.cur_seq = self.starting_seq
        return self.obs
    
    def change_starting_sequence(self, new_starting_seq):
        self.starting_seq = new_starting_seq
        self.reset()

    @property
    def obs(self):
        return self.str2array(self.starting_seq + self.cur_seq)
