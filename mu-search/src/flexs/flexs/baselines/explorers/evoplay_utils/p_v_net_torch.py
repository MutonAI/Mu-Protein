# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height, alphabet):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height
        # common layers
        self.conv1 = nn.Conv1d(len(alphabet), 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        # action policy layers
        self.act_conv1 = nn.Conv1d(128, 4*len(alphabet), kernel_size=5, padding=2)
        self.act_fc1 = nn.Linear(4*board_width*board_height,
                                 board_width*board_height)
        # state value layers
        self.val_conv1 = nn.Conv1d(128, len(alphabet), kernel_size=5, padding=2)
        self.val_fc1 = nn.Linear(board_width*board_height, 128)
        self.val_fc2 = nn.Linear(128, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # action policy layers
        # print(f"p_v_net_torch.py 43 x shape: {x.shape}")
        x_act = F.relu(self.act_conv1(x))
        # print(f"p_v_net_torch.py 45 x_act shape: {x_act.shape}, board_width: {self.board_width}, board_height: {self.board_height}, 4*board_width*board_height: {4*self.board_width*self.board_height}")
        x_act = x_act.view(-1, 4*self.board_width*self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act))
        # state value layers
        # print(f"p_v_net_torch.py 48 x_act shape: {x_act.shape}")
        x_val = F.relu(self.val_conv1(x))
        # print(f"p_v_net_torch.py 50 x_val shape: {x_val.shape}")
        x_val = x_val.view(-1, self.board_width*self.board_height)
        # print(f"p_v_net_torch.py 52 x_val shape: {x_val.shape}")
        x_val = F.relu(self.val_fc1(x_val))
        # print(f"p_v_net_torch.py 54 x_val shape: {x_val.shape}")
        x_val = F.tanh(self.val_fc2(x_val))
        # print(f"p_v_net_torch.py 56 x_val shape: {x_val.shape}")
        return x_act, x_val


class PolicyValueNet():
    """policy-value network """
    def __init__(self, board_width, board_height,
                 model_file=None, use_gpu=True, alphabet=None):
        self.use_gpu = use_gpu
        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(board_width, board_height, alphabet).cuda()
        else:
            self.policy_value_net = Net(board_width, board_height, alphabet)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            log_act_probs, value = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())
            return act_probs, value.data.numpy()

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = board.availables
        current_state_0 = np.expand_dims(board.current_state(), axis = 0)
        current_state = np.ascontiguousarray(current_state_0)  ##

        # print(f"p_v_net_torch.py 100 current_state: {current_state}")
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(current_state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            mcts_probs = Variable(torch.FloatTensor(mcts_probs).cuda())
            # print(f"p_v_net_torch.py 119 winner_batch: {winner_batch}")
            # Convert winner_batch to a list of floats before concatenation
            winner_batch = np.concatenate([np.array([w], dtype=np.float32) for w in winner_batch])
            winner_batch = Variable(torch.FloatTensor(winner_batch).cuda())
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            mcts_probs = Variable(torch.FloatTensor(mcts_probs))
            winner_batch = Variable(torch.FloatTensor(winner_batch))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)
        # print(f"p_v_net_torch.py 135 state_batch shape: {state_batch.shape}, value shape: {value.shape}, winner_batch shape: {winner_batch.shape}")
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        # print(f"p_v_net_torch.py 143 mcts_probs shape: {mcts_probs.shape}, log_act_probs shape: {log_act_probs.shape}")
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )

        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
