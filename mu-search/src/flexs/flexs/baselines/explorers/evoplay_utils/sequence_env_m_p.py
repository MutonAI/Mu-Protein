# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import torch
import random
from typing import List, Union
import copy
import threading


AAS = "ILVAGMFYWEDQNHCRKSTP"
def string_to_one_hot(sequence: str, alphabet: str) -> np.ndarray:

    out = np.zeros((len(sequence), len(alphabet)))
    for i in range(len(sequence)):
        out[i, alphabet.index(sequence[i])] = 1
    return out

def one_hot_to_string(
    one_hot: Union[List[List[int]], np.ndarray], alphabet: str
) -> str:
    """
    Return the sequence string representing a one-hot vector according to an alphabet.

    Args:
        one_hot: One-hot of shape `(len(sequence), len(alphabet)` representing
            a sequence.
        alphabet: Alphabet string (assigns each character an index).

    Returns:
        Sequence string representation of `one_hot`.

    """
    residue_idxs = np.argmax(one_hot, axis=1)
    return "".join([alphabet[idx] for idx in residue_idxs])

def string_to_feature(string, alphabet=AAS):
    seq_list = []
    seq_list.append(string)
    seq_np = np.array(
        [string_to_one_hot(seq, AAS) for seq in seq_list]
    )
    one_hots = torch.from_numpy(seq_np)
    one_hots = one_hots.to(torch.float32)
    return one_hots

class Seq_env(object):
    """sequence space for the env"""
    def __init__(self,
                 seq_len,
                 alphabet,
                 model,
                 starting_seq,
                 trust_radus,
                 ):

        self.max_moves = trust_radus
        self.move_count = 0

        self.seq_len = seq_len#self.width = int(kwargs.get('width', 8))
        self.vocab_size = len(alphabet)#self.height = int(kwargs.get('height', 8))

        self.alphabet = alphabet
        self.model = model
        self.starting_seq = starting_seq
        self.seq = starting_seq

        self._state = string_to_one_hot(self.seq, self.alphabet).astype(np.float32) 

        self.init_state = string_to_one_hot(self.seq, self.alphabet).astype(np.float32)
        self.previous_init_state = string_to_one_hot(self.seq, self.alphabet).astype(np.float32)
        self.unuseful_move = 0
        self.states = {}
        self.episode_seqs = []
        self.episode_seqs.append(starting_seq)
        self.repeated_seq_ocurr = False
        self.init_state_count = 0
        #playout
        self.start_seq_exclude_list = []
        self.playout_dict = {}
        self._lock = threading.RLock()

        # self.model.eval()
    def init_seq_state(self): #start_player=0

        self.previous_fitness = -float("inf")
        self.move_count = 0
        self.unuseful_move = 0

        self._state = copy.deepcopy(self.init_state)
        combo = one_hot_to_string(self._state, self.alphabet)
        self.start_seq_exclude_list.append(combo)
        self.init_combo = combo
        #
        if combo not in self.episode_seqs:
            self.episode_seqs.append(combo)
      
        one_hots = torch.from_numpy(self._state)
        one_hots = one_hots.unsqueeze(0)
        one_hots = one_hots.to(torch.float32)
        with torch.no_grad():
            inputs = one_hots
            inputs = inputs.permute(0, 2, 1)

            inputs_string = one_hot_to_string(self._state, self.alphabet)
            # print("sequence_env_m_p.py 105 inputs_string: ", inputs_string)
            outputs = self.model.get_fitness([inputs_string])

            outputs = outputs.squeeze()
        if outputs:
            self._state_fitness = outputs
      
        self.availables = list(range(self.seq_len * self.vocab_size))
        #evo
        for i, a in enumerate(combo):
            self.availables.remove(self.vocab_size * i + self.alphabet.index(a))
        for i, e_s in enumerate(self.episode_seqs):
            a_e_s = string_to_one_hot(e_s, self.alphabet)
            a_e_s_ex = np.expand_dims(a_e_s, axis=0)
            if i == 0:
                nda = a_e_s_ex
            else:
                nda = np.concatenate((nda, a_e_s_ex), axis=0)

        c_i_s = string_to_one_hot(combo, self.alphabet)
        for i, aa in enumerate(combo):
            tmp_c_i_s = np.delete(c_i_s, i, axis=0)
            for slice in nda:
                tmp_slice = np.delete(slice, i, axis=0)
                if (tmp_c_i_s == tmp_slice).all():
                    bias = np.where(slice[i] != 0)[0][0]
                    to_be_removed = self.vocab_size * i + bias
                    if to_be_removed in self.availables:
                        self.availables.remove(to_be_removed)
        #evo
        self.states = {}
        self.last_move = -1
        #
        self.previous_init_state = copy.deepcopy(self._state)
        #

    def current_state(self):

        square_state = np.zeros((self.seq_len, self.vocab_size))
        square_state = self._state
        return square_state.T
    def do_mutate(self, move, playout=0):
        """
        Performs a mutation on the current state by applying a given move and updating fitness.
        This method applies a move to the one-hot encoded state representation. It updates the internal
        state, removes the move from available moves, calculates the new fitness using a model's prediction,
        and records the mutation history. It also checks for repeated sequences and resets the state if
        the new fitness improves over the previous state.
        Parameters:
            move (int): The encoded move, where the position is derived by integer division using the vocabulary size,
                and the token/word is determined by the modulus operation.
            playout (int, optional): Flag indicating whether the move is from a playout simulation (default is 0). 
                A non-zero value triggers caching of the computed fitness in the playout dictionary.
        Returns:
            None
        Side Effects:
            - Increments the move counter.
            - Removes the specified move from the list of available moves.
            - Updates the internal state (_state) based on the move.
            - Computes and updates the state's fitness (_state_fitness) using the model's get_fitness method.
            - Updates the playout_dict if the current combination (state converted to a string) is not already cached.
            - Appends the current sequence (as a string) to episode_seqs and flags a repeated sequence if detected.
            - Resets the initial state (init_state) upon improvement of fitness over the previous state.
            - Sets the last_move to the current move.
        Comments:
            - The method converts the state from a one-hot encoded representation to a string (via one_hot_to_string)
              in order to use as a key for caching fitness values (playout_dict).
            - The torch.no_grad() context is used to avoid tracking gradients during the fitness computation.
            - If the move is unuseful (i.e., it attempts to activate an already active position in the state),
              the fitness is immediately set to 0.0.
        """
        self.previous_fitness = self._state_fitness
        self.move_count += 1
        self.availables.remove(move)
        pos = move // self.vocab_size
        res = move % self.vocab_size

        if self._state[pos, res] == 1:
            self.unuseful_move = 1
            self._state_fitness = 0.0
        else:
            self._state[pos] = 0
            self._state[pos, res] = 1

            combo = one_hot_to_string(self._state, self.alphabet)
            if playout==0:
                if combo not in self.playout_dict.keys():
                    # one_hots_0 = self._state
                    # one_hots = torch.from_numpy(one_hots_0)
                    # one_hots = one_hots.unsqueeze(0)
                    # one_hots = one_hots.to(torch.float32)
                    with torch.no_grad():
                        # inputs = one_hots
                        # inputs = inputs.permute(0, 2, 1)

                        inputs_string = one_hot_to_string(self._state, self.alphabet)
                        outputs = self.model.get_fitness([inputs_string])
                        outputs = outputs.squeeze()
                    if outputs:
                        self._state_fitness = outputs
                else:
                    self._state_fitness = self.playout_dict[combo]

            else:
                if combo not in self.playout_dict.keys():
                    # one_hots_0 = self._state
                    # one_hots = torch.from_numpy(one_hots_0)
                    # one_hots = one_hots.unsqueeze(0)
                    # one_hots = one_hots.to(torch.float32)
                    with torch.no_grad():
                        # inputs = one_hots
                        # inputs = inputs.permute(0, 2, 1)
                        
                        inputs_string = one_hot_to_string(self._state, self.alphabet)
                        outputs = self.model.get_fitness([inputs_string])

                        outputs = outputs.squeeze()
                    if outputs:
                        self._state_fitness = outputs
                        self.playout_dict[combo] = outputs
                else:

                    self._state_fitness = self.playout_dict[combo]

        current_seq = one_hot_to_string(self._state, self.alphabet)
        if current_seq in self.episode_seqs:
            self.repeated_seq_ocurr = True
            self._state_fitness = 0.0
        else:
            self.episode_seqs.append(current_seq)
        if self._state_fitness > self.previous_fitness:  # and not repeated_seq_ocurr:  # 0.6* 0.75*

            self.init_state = copy.deepcopy(self._state)
            self.init_state_count = 0

        self.last_move = move




    def mutation_end(self):
        # print(f"sequence_env_m_p.py 219 self.move_count: {self.move_count}, self.max_moves: {self.max_moves}, self.unuseful_move: {self.unuseful_move}, self._state_fitness: {self._state_fitness}, self.previous_fitness: {self.previous_fitness}, self.repeated_seq_ocurr: {self.repeated_seq_ocurr}")
        if self.repeated_seq_ocurr == True:
            return True
        #
        if self.move_count >= self.max_moves: 
            # print(f"sequence_env_m_p.py 224 self.move_count: {self.move_count} >= self.max_moves: {self.max_moves}, return True")
            return True
        #
        if self.unuseful_move == 1:
            return True
        if self._state_fitness < self.previous_fitness:  # 0.6* 0.75*
            return True

        return False
    
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            # print(f"sequence_env_m_p.py 239 key: {key}")
            if key == 'model':
                # Shallow copy the model (or assign a shared reference)
                setattr(result, key, value)
            elif key == 'move_count':
                setattr(result, key, copy.deepcopy(value))
                # print(f"sequence_env_m_p.py 244 key: {key}, value: {value}")
            elif key == '_lock':
                # Reinitialize the lock instead of copying it
                setattr(result, key, threading.RLock())
            else:
                setattr(result, key, copy.deepcopy(value, memo))
        return result
    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     # Remove non-picklable attributes
    #     for key in ['_lock', 'model']:
    #         state.pop(key, None)
    #     return state

    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     # Reinitialize the lock
    #     self._lock = threading.RLock()
    #     # If necessary, reassign the model (perhaps from a global or an external reference)
    #     self.model = get_shared_model()  # Define this appropriately


class Mutate(object):
    """mutating server"""

    def __init__(self, Seq_env, **kwargs):
        self.Seq_env = Seq_env
        self.alphabet = Seq_env.alphabet

    def start_p_mutating(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.Seq_env.init_board(start_player)
        p1, p2 = self.Seq_env.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.Seq_env, player1.player, player2.player)
        while True:
            current_player = self.Seq_env.get_current_player()
            player_in_turn = players[current_player]
            move = player_in_turn.get_action(self.Seq_env)
            self.Seq_env.do_move(move)
            if is_shown:
                self.graphic(self.Seq_env, player1.player, player2.player)
            end, winner = self.Seq_env.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_mutating(self, mutater, is_shown=0, temp=1e-3):#mutater,
        """ start mutating using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """

        self.Seq_env.init_seq_state()
        print("starting sequence：{}".format(self.Seq_env.init_combo))
        generated_seqs = []

        fit_result = []
        play_seqs_list = []
        play_fit_list = []
        states, mcts_probs, reward_z = [], [], [] #, current_players #, []
        while True:
            move, move_probs, play_seqs, play_losses = mutater.get_action(self.Seq_env,
                                                 temp=temp,
                                                 return_prob=1)
            self.Seq_env.playout_dict.update(mutater.m_p_dict)
            # print(f"\033[93msequence_env_m_p.py 320 move: {move}, move_probs: {move_probs}, play_seqs: {play_seqs}, play_losses: {play_losses}\033[0m")
            # TODO: if the move is 0?
            if move or move == 0:
                # store the data
                states.append(self.Seq_env.current_state())
                mcts_probs.append(move_probs)
                reward_z.append(self.Seq_env._state_fitness)
                # perform a move
                self.Seq_env.do_mutate(move)
                generated_seqs.append(one_hot_to_string(self.Seq_env._state, self.alphabet))

                fit_result.append(self.Seq_env._state_fitness)
                print("move_fitness: %f\n" % (self.Seq_env._state_fitness))
                print("episode_seq len: %d\n" % (len(self.Seq_env.episode_seqs)))
                print("Mmove & playout dict len: %d\n" % (len(self.Seq_env.playout_dict)))
                state_string = one_hot_to_string(self.Seq_env._state, self.alphabet)
                print(state_string)
            # if move == []:
            #     print(f"\033[93mNo available move.\033[0m")
            # elif move or move == 0:
            #     print(f"\033[93mAvailable move: {move}\033[0m")
            end = self.Seq_env.mutation_end()
            if end or move == []:

                mutater.reset_Mutater()
                if is_shown:

                    print("Mutation end.")
                playout_dict = copy.deepcopy(self.Seq_env.playout_dict)
                return zip(states, mcts_probs, reward_z), zip(generated_seqs, fit_result), playout_dict