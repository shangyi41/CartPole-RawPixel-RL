import numpy as np
import torch as tc

from ddqn import DDQN
from collections import deque
from torch.nn import MSELoss
from torch.optim import Adam

class Agent:
    """An agent that uses DDQN to solve CartPole."""

    def __init__(self, mem_size, batch_size, update_rate_target, lr=10**-3, gamma=0.99, eps_init=1.0, eps_min=0.01, eps_decay=5*10**-5, last_n_frames=4, is_trained=True):
        """
        Create agent.

        Parameters
        --------------------
        mem_size: int
            memory buffer size.

        batch_size: int
            batch size.

        update_rate_target: int
            how many steps to wait to update target net.

        lr: float, optional
            learning rate for optimizer.

        gamma: float, optional
            discount factor.

        eps_init: float, optional
            epsilon initial value.

        eps_min: float, optional
            epsilon minimum value.

        eps_decay: float, optional
            epsilon decay value.

        last_n_frames: int
            last n frames to be processed.

        is_trained: bool, optional
            True if agent is trained, False otherwise.
        """

        self.action_size = 2
        self.memory = deque(maxlen = mem_size)              #Memory buffer.
        self.batch_size = batch_size
        self.update_rate_target = update_rate_target
        self.gamma = gamma
        self.epsilon = eps_init
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        self.is_trained = is_trained
        self.rng = np.random.default_rng()
        self.last_n_frames= last_n_frames

        #Training model.
        self.model = DDQN(last_n_frames)
        self.target = DDQN(last_n_frames)
        self.loss_function = MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        self.target.copy_from(self.model)

    def _adjust_observation(self, obs):
        """
        Preprocess an observation.

        Parameter
        --------------------
        obs: arraylike
            an observation stacked.

        Return
        --------------------
        new_obs: ndarray
            obs preprocessed.
        """

        new_obs = np.zeros((self.last_n_frames, 84, 84), dtype=np.uint8)
        for i in range(self.last_n_frames):
            new_obs[i] = obs[i]
        
        return new_obs

    def choose_action(self, obs):
        """
        Choose an action for obs.

        Parameter
        --------------------
        obs: arraylike
            an observation stacked.

        Return
        --------------------
        action: int
            an action choosen by agent.

        q: float
            value estimated by agent of action. 
        """

        new_obs = self._adjust_observation(obs)
        x = tc.Tensor(np.array([new_obs])).to(self.model.device)
        q = self.model.forward(x)

        if self.rng.uniform() <= self.epsilon and self.is_trained:
            #A random action is choosen.
            action = self.rng.integers(0, self.action_size)
        else:
            #Best action is choosen.
            action = tc.argmax(q).item()
        
        return action, (q.cpu())[0, action].item()

    def store(self, obs, action, reward, next_obs, is_next_obs_terminal):
        """
        Store transiction into memory buffer.

        Parameters
        --------------------
        obs: arraylike
            an observation stack.

        action: int
            an action accomplished for obs.

        reward: float
            reward obtained from action accomplished for obs.

        next_obs: arraylike
            next observation stack.

        is_next_obs_terminal: bool
            True if next_obs is a terminal state, False otherwise.
        """

        if self.is_trained:
            new_obs      = self._adjust_observation(obs)
            new_next_obs = self._adjust_observation(next_obs)

            self.memory.append((new_obs, action, reward, new_next_obs, is_next_obs_terminal))

    def _sample_batch(self):
        """
        Sample a minibatch from memory buffer.
        
        Return
        --------------------
        obs_b: ndarray
            observation batch.
              
        act_b: ndarray
            action batch
            
        rew_b: ndarray
            reward batch
         
        next_obs_b: ndarray 
            next observation batch
            
        next_obs_done_b: ndarraay
            next_obs terminal state batch.
        """

        #Minibatch.
        obs_batch               = np.zeros((self.batch_size, self.last_n_frames, 84, 84), dtype=np.uint8)
        next_obs_batch          = np.zeros((self.batch_size, self.last_n_frames, 84, 84), dtype=np.uint8)
        action_batch            = np.zeros(self.batch_size, dtype=np.int8)
        reward_batch            = np.zeros(self.batch_size, dtype=np.int8)
        next_obs_terminal_batch = np.zeros(self.batch_size, dtype=bool)

        #Some experience are choosen randomly from memory.
        indices_batch = self.rng.permutation(np.arange(0, len(self.memory), 1, dtype=np.int32))[:self.batch_size]

        for i in range(self.batch_size):
            obs_batch[i] = self.memory[indices_batch[i]][0]
            action_batch[i] = self.memory[indices_batch[i]][1]
            reward_batch[i] = self.memory[indices_batch[i]][2]
            next_obs_batch[i] = self.memory[indices_batch[i]][3]
            next_obs_terminal_batch[i] = self.memory[indices_batch[i]][4]

        return obs_batch, action_batch, reward_batch, next_obs_batch, next_obs_terminal_batch

    def train(self):
        """Do training step."""

        if not self.is_trained or len(self.memory) < self.batch_size:
            return
        
        #A minibatch is built.
        obs_b, action_b, reward_b, next_obs_b, next_obs_done_b = self._sample_batch()
        obs_b = tc.Tensor(obs_b).to(self.model.device)
        next_obs_b = tc.Tensor(next_obs_b).to(self.model.device)
        reward_b = tc.Tensor(reward_b).to(self.model.device)

        #Compute q-values.
        idxs = np.arange(0, self.batch_size, 1, dtype=np.int32)
        
        q = self.model.forward(obs_b)[idxs, action_b]
        
        best_action_batch = np.array( tc.argmax(self.model.forward(next_obs_b), dim=1).cpu() )
        q_next = self.target.forward(next_obs_b)[idxs, best_action_batch]
        q_next[next_obs_done_b] = 0.0

        q_target = reward_b + self.gamma * q_next

        #Do training step.
        self.optimizer.zero_grad()
        loss = self.loss_function(q, q_target).to(self.model.device)
        loss.backward()
        self.optimizer.step()

        #Update epsilon
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

    def update_target(self):
        self.target.copy_from(self.model)

    def save_model(self, name):
        """
        Save model.

        Parameter
        --------------------
        name: str
            filename of model to save.
        """

        tc.save(self.model.state_dict(), name)

    def load_model(self, name):
        """
        Load model.

        Parameter
        --------------------
        name: str
            filename of model to load.
        """

        self.model.load_state_dict(tc.load(name))