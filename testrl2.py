import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import string
from sklearn.model_selection import train_test_split
import gymnasium as gym
from gymnasium import spaces
import torch.optim as optim
from torch.distributions import Categorical


class Vocab:
    def __init__(self):
        self.char2id = dict()
        self.char2id['#'] = 0 # for masking token
        self.char_list = string.ascii_lowercase
        for i, c in enumerate(self.char_list):
            self.char2id[c] = len(self.char2id)
        self.char2id['_'] = len(self.char2id) # for padding token
        self.id2char = {v: k for k, v in self.char2id.items()}

    def __len__(self):
        return len(self.char2id)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            if mask.dim() < attn.dim():
                mask = mask.unsqueeze(2)
            mask = mask.permute(2, 0, 1, 3).contiguous()
            # print(attn.size(), mask.size())
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        return self.fc2(self.leaky_relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # print('X:', x.size(), 'Pos: ', self.pos_table.size(), 'Pos_ trunc:', self.pos_table.clone()[:, :x.size(1)].size())
        return x + self.pos_table.clone()[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(n_head=num_heads, d_model=d_model, d_k=d_model // num_heads, d_v=d_model // num_heads, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn = self.mha(q=x, k=x, v=x, mask=mask)
        attn_out = self.dropout1(attn_out)
        x = self.layernorm1(x + attn_out)

        ff_out = self.ff(x)
        ff_out = self.dropout2(ff_out)
        x = self.layernorm2(x + ff_out)

        return x


class BERT(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, vocab_size, max_len, dropout):
        super(BERT, self).__init__()
        self.d_model = d_model
        # Embedding layers
        self.token_embedding = Embeddings(d_model, vocab_size)
        self.position_encoding = PositionalEncoding(d_model, max_len)
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.fc1 = nn.Linear(d_model + 26, 64)
        self.layer_norm = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 26)

    def forward(self, x: torch.Tensor, prev_guess: torch.Tensor, mask: torch.Tensor):
        x = self.token_embedding(x)
        x = self.position_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        x = x.mean(dim=1)
        x = torch.cat((x, prev_guess), dim=1)
        x = F.leaky_relu(self.layer_norm(self.fc1(x)))
        x = self.fc2(x)
        return x


class HangmanEnv(gym.Env):
    def __init__(self, word_list, vocab, max_attempts=6, render_mode='human'):
        super(HangmanEnv, self).__init__()

        self.word_list = word_list
        self.vocab = vocab  # Use Vocab instance for tokenization
        self.max_attempts = max_attempts
        self.attempts_left = max_attempts
        self.render_mode = render_mode
        self.current_word = None
        self.masked_word = None
        self.guessed_letters = None
        self.word_length = None
        self.is_first_guess = True  # Track if it's the first guess
        self.curr_guess = '_'

        self.vowels = {'a', 'e', 'i', 'o', 'u'}  # Set of vowels

        # Action space: Choosing a letter from the available characters in vocab (excluding padding and masking tokens)
        self.action_space = spaces.Discrete(len(self.vocab.char2id) - 2)  # Exclude padding and masking token

        # Observation space:
        self.observation_space = spaces.Dict({
            'masked_word': spaces.Box(low=0, high=len(self.vocab.char2id)-1, shape=(30,), dtype=np.int32),
            'guessed_letters': spaces.MultiBinary(len(self.vocab.char2id)-2)
        })

    def reset(self, *, seed=None, options=None):
        # Set the seed if provided
        super().reset(seed=seed)
        np.random.seed(seed)  # Set the numpy random seed

        # Reset the game state
        self.current_word = np.random.choice(self.word_list)
        print('Current Word: ', self.current_word)

        self.word_length = len(self.current_word)
        self.attempts_left = self.max_attempts
        self.guessed_letters = np.zeros(len(self.vocab.char2id)-2, dtype=np.int32)

        # Initialize the masked word using the mask token from Vocab (e.g., '#')
        self.masked_word = np.full(self.word_length, self.vocab.char2id['#'], dtype=np.int32)
        self.is_first_guess = True  # Reset first guess tracking

        return self._get_obs(), {}


    def _get_obs(self):
        # Pad the masked_word to a fixed length of 30, using padding token
        padded_masked_word = np.pad(self.masked_word, (0, 30 - len(self.masked_word)), 'constant',
                                    constant_values=self.vocab.char2id['_'])

        return {
            'masked_word': padded_masked_word,
            'guessed_letters': self.guessed_letters
        }

    def step(self, action):
        if isinstance(action, np.ndarray):
          action = action.item()

        # Decode action: map to letter using the inverse vocab mapping (id2char)
        guessed_letter = chr(action + ord('a'))
        self.curr_guess = guessed_letter

        # Update guessed letters
        self.guessed_letters[action] = 1

        # Check if the guessed letter is in the word
        reward = 0.0
        if guessed_letter in self.current_word and self.vocab.char2id[guessed_letter] not in self.masked_word:
            # Correct guess: reveal the letter in the masked word
            indices = [i for i, letter in enumerate(self.current_word) if letter == guessed_letter]
            for idx in indices:
                self.masked_word[idx] = self.vocab.char2id[guessed_letter]
            reward += 0.3  # Reward for a correct guess
        else:
            # Incorrect guess: reduce attempts and apply a small penalty
            self.attempts_left -= 1
            reward -= 0.1  # Penalty for incorrect guess

        # First guess heuristic: reward for guessing a vowel on the first guess
        if self.is_first_guess and guessed_letter in self.vowels:
            reward += 0.2  # Bonus for guessing a vowel as the first guess
        self.is_first_guess = False  # After the first guess, disable the heuristic

        # Check if the game is terminated (win/loss condition)
        terminated = False
        truncated = False  # We won't use truncated in this environment, but return it for compatibility
        if np.all(self.masked_word != self.vocab.char2id['#']):
            reward += 1.0  # Additional reward for winning the game
            terminated = True
        elif self.attempts_left == 0:
            reward -= 1.0  # Penalty for losing the game
            terminated = True

        return self._get_obs(), reward, terminated, truncated, {}


    def render(self):
      if self.render_mode == 'human':
        # Display the current state of the game
        word_display = ''.join([self.vocab.id2char[c] if c != self.vocab.char2id['#'] else '_' for c in self.masked_word])
        print(f"Word: {word_display}, Attempts left: {self.attempts_left}, Guessed letters: {self.curr_guess}, Previous_guesses: {self.guessed_letters}")

    def close(self):
        pass

class BERTPolicyNetwork(nn.Module):
    def __init__(self, bert_model, action_dim=26):
        super(BERTPolicyNetwork, self).__init__()
        self.bert_model = bert_model
        self.fc = nn.Linear(26, action_dim)  # Output layer to map BERT output to action space

    def forward(self, masked_word, guessed_letters, mask_token=27):
        masked_word = torch.tensor(masked_word, dtype=torch.long)
        guessed_letters = torch.tensor(guessed_letters, dtype=torch.float32)
        if guessed_letters.dim() < 2:
            guessed_letters = guessed_letters.unsqueeze(0)
        # print('Before BERT', masked_word.size(), guessed_letters.size())
        mask = torch.tensor(masked_word != mask_token).long().unsqueeze(0)
        # print('guessed_letters', guessed_letters, 'mask', mask)
        x = self.bert_model(masked_word, guessed_letters, mask)
        x = self.fc(x)
        for i in range(x.size(0)):
            guessed_letter_indices = guessed_letters[i].nonzero()
            # print(x.size(), guessed_letter_indices.size())
            # print('prev guesses: ', [chr(65 + letter) for letter in guessed_letter_indices])
            x[i, guessed_letter_indices] = -1e9  # Mask guessed letters
            # print('masked letters', [chr(65 + j) for j in (x[i] == -1e9).nonzero()])

        return F.log_softmax(x, dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, bert_model):
        super(ValueNetwork, self).__init__()
        self.bert_model = bert_model
        self.fc = nn.Linear(26, 1)  # Output a single value

    def forward(self, masked_word, guessed_letters, mask_token=27):
        masked_word = torch.tensor(masked_word, dtype=torch.long)
        guessed_letters = torch.tensor(guessed_letters, dtype=torch.float32)
        if guessed_letters.dim() < 2:
            guessed_letters = guessed_letters.unsqueeze(0)
        # print('Before BERT', masked_word.size(), guessed_letters.size())
        mask = torch.tensor(masked_word != mask_token).long().unsqueeze(0)
        # print('guessed_letters', guessed_letters, 'mask', mask)
        x = self.bert_model(masked_word, guessed_letters, mask)
        # for i in range(x.size(0)):
        #     guessed_letter_indices = guessed_letters[i].nonzero()
        #     # print(x.size(), guessed_letter_indices.size())
        #     # print([chr(65 + letter) for letter in guessed_letter_indices])
        #     x[i, guessed_letter_indices] = -1e9  # Mask guessed letter
        value = self.fc(x)
        return value


class PPO:
    def __init__(self, policy_model, value_model, env, learning_rate=1e-5, gamma=0.95, eps_clip=0.2, epochs=10, weight_decay=1e-5):
        self.policy_model = policy_model
        self.value_model = value_model
        policy_model.apply(self._init_weights)
        value_model.apply(self._init_weights)
        self.env = env
        self.optimizer = optim.Adam(list(policy_model.parameters()) + list(value_model.parameters()), lr=learning_rate, weight_decay=weight_decay)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)


    def select_action(self, state):
        masked_word, guessed_letters = self._get_states(state)
        logits = self.policy_model(masked_word, guessed_letters)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_advantages(self, rewards, values, next_value, dones):
        advantages = []
        running_advantage = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_advantage = 0
            td_error = rewards[t] + self.gamma * next_value[t] - values[t]
            running_advantage = td_error + self.gamma * running_advantage
            advantages.insert(0, running_advantage)
        return advantages

    def _get_states(self, states):
        if isinstance(states, list):
            masked_words = []
            guessed_letters = []
            for state in states:
                masked_words.append(state['masked_word'])
                guessed_letters.append(state['guessed_letters'])
        elif isinstance(states, tuple):
            if isinstance(states[0], tuple) or isinstance(states[0], np.ndarray):
              masked_words = states[0]
              guessed_letters = states[1]
            elif isinstance(states[0], dict):
              state = states[0]
              masked_words = state['masked_word']
              guessed_letters = state['guessed_letters']
            else:
                raise ValueError(f"Unsupported state format {type(states[0])}, {states}")
        else:
            masked_words = states['masked_word']
            guessed_letters = states['guessed_letters']

        return masked_words, guessed_letters

    def update(self, trajectories):
        for _ in range(self.epochs):
            for trajectory in trajectories:
                states, actions, old_log_probs, returns, advantages = trajectory
                actions = torch.tensor(actions, dtype=torch.long)
                old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
                returns = torch.tensor(returns, dtype=torch.float32)
                advantages = torch.tensor(advantages, dtype=torch.float32)

                # Policy loss
                masked_word, guessed_letters = self._get_states(states)
                logits = self.policy_model(masked_word, guessed_letters)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)
                ratios = torch.exp(new_log_probs - old_log_probs)

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                masked_word, guessed_letters = self._get_states(states)
                value_pred = self.value_model(masked_word, guessed_letters).squeeze(0)
                value_loss = nn.MSELoss()(value_pred.squeeze(), returns)

                # Total loss
                loss = policy_loss + 0.5 * value_loss

                self.optimizer.zero_grad()
                loss.backward()

                # Apply gradient clipping
                # max_norm = 10.0
                # torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=max_norm)
                # torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), max_norm=max_norm)
                self.optimizer.step()


    def train(self, num_episodes):
        policy_network.train()
        value_network.train()
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

            while not done:
                masked_word, guessed_letters = self._get_states(state)
                action, log_prob = self.select_action(state)
                value = self.value_model(masked_word, guessed_letters).item()
                # print(env.step(action))
                next_state, reward, done, _, _ = self.env.step(action)

                states.append({'masked_word': masked_word, 'guessed_letters': guessed_letters})
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob.item())
                values.append(value)
                dones.append(done)

                state = next_state

            # Compute advantages
            masked_word, guessed_letters = self._get_states(states)
            next_value = self.value_model(masked_word, guessed_letters).squeeze().detach().numpy()
            returns = self.compute_advantages(rewards, values, next_value, dones)

            # Prepare the trajectory
            trajectories = [(states, actions, log_probs, returns, values)]
            self.update(trajectories)

            if episode % 100 == 0:
                print(f"Episode {episode}, Total reward: {sum(rewards)}")
                # self.test_models_grad(policy_network, 'Policy', clipping=False)
                # self.test_models_grad(value_network, 'Value', clipping=False)


    def test_models_grad(self, model, model_name, clipping=False):
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if not clipping:
                  print(f"For {model_name}, Original gradient norm for {name}: {grad_norm}")
                else:
                  print(f"For {model_name}, Clipped gradient norm for {name}: {grad_norm}")



if __name__ == '__main__':
    word_list = []
    input_file = 'words_250000_train.txt'
    with open(input_file, 'r') as f:
        for line in f:
            word = line.strip()
            word_list.append(word)

    # max_len = max([len(word) for word in word_list])
    # Create the environment
    vocab = Vocab()
    env = HangmanEnv(word_list=word_list, vocab=vocab, max_attempts=6)

    # BERT model (from your custom implementation)
    bert_model = BERT(
        num_layers=4,
        d_model=128,
        num_heads=8,
        d_ff=512,
        vocab_size=28,
        max_len=30,
        dropout=0.1
    )

    # Initialize policy network (BERT for policy) and value network
    policy_network = BERTPolicyNetwork(bert_model, action_dim=26)  # Assuming 26 letters for actions
    value_network = ValueNetwork(bert_model)

    # Instantiate PPO agent
    ppo_agent = PPO(policy_model=policy_network, value_model=value_network, env=env, epochs=10, weight_decay=1e-4)

    # # Train the agent
    ppo_agent.train(num_episodes=1000000)
    print('Training complete.......')


    # Save the policy network
    torch.save(policy_network.state_dict(), 'policy_network.pth')
    # Save the value network
    torch.save(value_network.state_dict(), 'value_network.pth')
    # Save the BERT model
    torch.save(bert_model.state_dict(), 'bert_model.pth')
    # Save the PPO agent
    torch.save(ppo_agent, 'ppo_agent.pth')
    # Save the environment
    torch.save(env, 'env.pth')

    # Load the policy network, value network, BERT model, PPO agent, and environment
    # policy_network.load_state_dict(torch.load('policy_network.pth'))
    # value_network.load_state_dict(torch.load('value_network.pth'))
    # ppo_agent = torch.load('ppo_agent.pth')
    # env = torch.load('env.pth')

    # Test the PPO agent
    win_count = 0
    for _ in range(20):
        rewards = []
        state = env.reset()
        done = False
        while not done:
            action, _ = ppo_agent.select_action(state)
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            env.render()
            rewards.append(reward)
        print(f"Total reward: {sum(rewards)}")
        if sum(rewards) > 0:
            win_count += 1
    print(f"Win count: {win_count}")
        
        
    