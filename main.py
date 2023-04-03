from cv2 import resize
import gym
from gym.core import ObservationWrapper
from gym.spaces import Box
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import torch.nn as nn
import torch
from tqdm import trange

import atari_wrappers
from framebuffer import FrameBuffer
from replay_buffer import ReplayBuffer
import utils


ENV_NAME = "BreakoutNoFrameskip-v4"
REPLAY_BUFFER_SIZE = 10**4
N_STEPS = 100


class PreprocessAtariObs(ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and grayscales it."""
        ObservationWrapper.__init__(self, env)

        self.img_size = (1, 64, 64)
        self.observation_space = Box(0.0, 1.0, self.img_size)


    def _to_gray_scale(self, rgb, channel_weights=(0.8, 0.1, 0.1)):
        result = np.zeros(rgb.shape[:-1])
        for i, coef in enumerate(channel_weights):
            result += rgb[:,:,i] * coef
        return result


    def observation(self, img):
        """what happens to each observation"""
        result = self._to_gray_scale(resize(img, self.img_size[1:]))
        result.resize(self.img_size)
        result = (result / result.max()).astype('float32')
        return result


def PrimaryAtariWrap(env, clip_rewards=True):
    assert 'NoFrameskip' in env.spec.id

    # This wrapper holds the same action for <skip> frames and outputs
    # the maximal pixel value of 2 last frames (to handle blinking
    # in some envs)
    env = atari_wrappers.MaxAndSkipEnv(env, skip=4)

    # This wrapper sends done=True when each life is lost
    # (not all the 5 lives that are givern by the game rules).
    # It should make easier for the agent to understand that losing is bad.
    env = atari_wrappers.EpisodicLifeEnv(env)

    # This wrapper laucnhes the ball when an episode starts.
    # Without it the agent has to learn this action, too.
    # Actually it can but learning would take longer.
    env = atari_wrappers.FireResetEnv(env)

    # This wrapper transforms rewards to {-1, 0, 1} according to their sign
    if clip_rewards:
        env = atari_wrappers.ClipRewardEnv(env)

    # This wrapper is yours :)
    env = PreprocessAtariObs(env)
    return env


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0.):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        self.network = nn.Sequential()
        in_size = state_shape[0]
        out_size = n_actions

        self.network.add_module('conv1', nn.Conv2d(in_size, 16, 3, 2))
        self.network.add_module('relu1', nn.ReLU())
        self.network.add_module('conv2', nn.Conv2d(16, 32, 3, 2))
        self.network.add_module('relu2', nn.ReLU())
        self.network.add_module('conv3', nn.Conv2d(32, 64, 3, 2))
        self.network.add_module('relu3', nn.ReLU())
        self.network.add_module('flatten', nn.Flatten())
        self.network.add_module('linear1', nn.Linear(448 * 7, 256))
        self.network.add_module('relu4', nn.ReLU())
        self.network.add_module('linear2', nn.Linear(256, out_size))

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        qvalues = self.network(state_t)
        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


def make_env(clip_rewards=True, seed=None):
    env = gym.make(ENV_NAME)  # create raw env
    if seed is not None:
        env.seed(seed)
    env = PrimaryAtariWrap(env, clip_rewards)
    env = FrameBuffer(env, n_frames=4, dim_order='pytorch')
    return env


def evaluate(env, agent, n_games=1, greedy=False, t_max=10000):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    for _ in range(n_games):
        s = env.reset()
        reward = 0
        for _ in range(t_max):
            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)
    return np.mean(rewards)


def play_and_record(initial_state, agent, env, exp_replay, n_steps=1):
    """
    Play the game for exactly n_steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    PLEASE DO NOT RESET ENV UNLESS IT IS "DONE"

    :returns: return sum of rewards over time and the state in which the env stays
    """
    s = initial_state
    sum_rewards = 0

    # Play the game for n_steps as per instructions above
    for t in range(n_steps):
        a = agent.sample_actions(agent.get_qvalues([s]))[0]
        next_s, r, done, _ = env.step(a)

        if exp_replay is not None:
            exp_replay.add(s, a, r, next_s, done)

        s = next_s
        sum_rewards += r
        if done:
            s = env.reset()

    return sum_rewards, s


def compute_td_loss(states, actions, rewards, next_states, is_done,
                    agent, target_network, device,
                    gamma=0.99,
                    check_shapes=False):
    """ Compute td loss using torch operations only. Use the formulae above. """
    states = torch.tensor(states, device=device, dtype=torch.float32)  # shape: [batch_size, *state_shape]
    actions = torch.tensor(actions, device=device, dtype=torch.int64)  # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float32)  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float)
    is_done = torch.tensor(
        is_done.astype('float32'),
        device=device,
        dtype=torch.float32,
    )  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)  # shape: [batch_size, n_actions]

    # compute q-values for all actions in next states
    predicted_next_qvalues = target_network(next_states)  # shape: [batch_size, n_actions]

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]  # shape: [batch_size]

    # compute V*(next_states) using predicted next q-values
    next_state_values = torch.max(predicted_next_qvalues, dim=1)[0]

    assert next_state_values.dim() == 1 and next_state_values.shape[0] == states.shape[0], \
        "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    target_qvalues_for_actions = rewards + gamma * next_state_values * is_not_done

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions - target_qvalues_for_actions.detach()) ** 2)

    if check_shapes:
        assert predicted_next_qvalues.data.dim() == 2, \
            "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim() == 1, \
            "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim() == 1, \
            "there's something wrong with target q-values, they must be a vector"

    return loss


def wait_for_keyboard_interrupt():
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    env = make_env(seed=seed)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    state = env.reset()

    agent = DQNAgent(state_shape, n_actions, epsilon=1).to(device)
    target_network = DQNAgent(state_shape, n_actions).to(device)
    target_network.load_state_dict(agent.state_dict())

    exp_replay = ReplayBuffer(REPLAY_BUFFER_SIZE)
    for i in trange(REPLAY_BUFFER_SIZE // N_STEPS):
        if not utils.is_enough_ram(min_available_gb=0.1):
            print("""
                Less than 100 Mb RAM available. 
                Make sure the buffer size in not too huge.
                Also check, maybe other processes consume RAM heavily.
                """
                  )
            break
        play_and_record(state, agent, env, exp_replay, n_steps=N_STEPS)
        if len(exp_replay) == REPLAY_BUFFER_SIZE:
            break

    timesteps_per_epoch = 1
    batch_size = 16
    total_steps = 3 * 10 ** 6
    decay_steps = 10 ** 6

    opt = torch.optim.Adam(agent.parameters(), lr=1e-4)

    init_epsilon = 1
    final_epsilon = 0.1

    loss_freq = 50
    refresh_target_network_freq = 5000
    eval_freq = 5000

    max_grad_norm = 50

    mean_rw_history = []
    td_loss_history = []
    grad_norm_history = []
    initial_state_v_history = []
    step = 0

    state = env.reset()
    with trange(step, total_steps + 1) as progress_bar:
        for step in progress_bar:
            if not utils.is_enough_ram():
                print('less that 100 Mb RAM available, freezing')
                print('make sure everything is ok and use KeyboardInterrupt to continue')
                wait_for_keyboard_interrupt()

            agent.epsilon = utils.linear_decay(init_epsilon, final_epsilon, step, decay_steps)

            # play
            _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

            # train
            s_, a_, r_, next_s_, done_ = exp_replay.sample(batch_size)

            loss = compute_td_loss(s_, a_, r_, next_s_, done_, agent, target_network, device)

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            opt.step()
            opt.zero_grad()

            if step % loss_freq == 0:
                td_loss_history.append(loss.data.cpu().item())
                grad_norm_history.append(grad_norm)

            if step % refresh_target_network_freq == 0:
                # Load agent weights into target_network
                target_network.load_state_dict(agent.state_dict())

            if step % eval_freq == 0:
                torch.save(agent.state_dict(), './weights.pt')

                mean_rw_history.append(evaluate(
                    make_env(seed=step), agent, n_games=3, greedy=True, t_max=1000)
                )
                initial_state_q_values = agent.get_qvalues(
                    [make_env(seed=step).reset()]
                )
                initial_state_v_history.append(np.max(initial_state_q_values))

                clear_output(True)
                print("buffer size = %i, epsilon = %.5f" %
                    (len(exp_replay), agent.epsilon))

                plt.figure(figsize=[16, 9])

                plt.subplot(2, 2, 1)
                plt.title("Mean reward per episode")
                plt.plot(mean_rw_history)
                plt.grid()

                assert not np.isnan(td_loss_history[-1])
                plt.subplot(2, 2, 2)
                plt.title("TD loss history (smoothened)")
                plt.plot(utils.smoothen(td_loss_history))
                plt.grid()

                plt.subplot(2, 2, 3)
                plt.title("Initial state V")
                plt.plot(initial_state_v_history)
                plt.grid()

                plt.subplot(2, 2, 4)
                plt.title("Grad norm history (smoothened)")
                plt.plot(utils.smoothen(list(map(lambda x: x.cpu(), grad_norm_history))))
                plt.grid()

                plt.show()


if __name__ == '__main__':
    main()