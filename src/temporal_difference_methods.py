import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
import datetime
import os,sys
import enum
import time
import math

from tools import HyperParam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from multiprocessing.dummy import Pool
import multiprocessing

from scipy import stats

class Game():
    def __init__(self, idx, name, max_episode_steps, td_steps, model_pi, model_v, writer):
        self.idx = idx
        self.env = gym.make(name) 
        self.env._max_episode_steps = max_episode_steps 
        self.state = self.env.reset()
        self.td_steps = td_steps
        self.model_pi = model_pi
        self.model_v = model_v
        self.writer = writer
        self.total_steps = 0
        self.epoches = 0

    def get_action(self, distribution):
        output_dim = len(distribution)
        a = np.random.choice(np.arange(output_dim), p=distribution)
        prob = distribution[a]
        return a, prob

    def play(self):
        states = []
        pred_vs = []
        actions = []
        rewards = []
        next_states = []
        probs = []
        dones = []
        steps = 0
        done = False
        while not done and steps < self.td_steps:
            pred_v = self.model_v(np.expand_dims(self.state, 0)) # [batch_size, 1]
            pred_vs.append(pred_v) # [steps, batch_size, 1] list of v for each state

            distribution = np.squeeze(self.model_pi(np.expand_dims(self.state, 0))) # [action_size,]
            #distribution = self.model_pi(np.expand_dims(self.state, 0))[0] # [action_size,]
            action, prob = self.get_action(distribution)
            state_, reward, done, _ = self.env.step(action)

            states.append(self.state) # [steps, state_size] 
            actions.append(action) # [steps, 1]
            probs.append(prob) # [steps, action_size]
            rewards.append(reward) # [steps, 1]
            next_states.append(state_) # [steps, state_size]
            dones.append(done) # [steps, 1]
            self.state = state_ # [steps, state_size]

            steps += 1
        values = self.calc_values(rewards) # [steps, 1]
        self.total_steps += steps
        
        if done:
            self.epoches += 1
            print(f"game {self.idx}: total_steps {self.total_steps}")
            if (self.idx == 0):
                with self.writer.as_default():
                    tf.summary.scalar(f'total_steps_{self.idx}', self.total_steps, step=self.epoches)
            self.state = self.env.reset()
            self.total_steps = 0

        return (np.vstack(states), values, np.asarray(pred_vs),
                np.asarray(actions), np.asarray(probs), dones)

    def calc_values(self, r):
        values = np.zeros_like(r, dtype=np.float32)
        values[-1] = r[-1]
        for i in reversed(range(len(r)-1)):
            values[i] = 0.999*values[i+1] + r[i]
        return values


class LossEnum(enum.Enum):
    VANILLA = 1
    PPO = 2
    BASELINE = 3
    def __str__(self):
        if self.value == 1:
            return 'Vanilla'
        elif self.value == 2:
            return 'ppo'
        elif self.value ==3:
            return 'baseline'
        else:
            return 'error'


class Agent():
    def __init__(self, idx, params, params_name, writer):
        print (params)
        (self.loss_fun, self.MAX_ENV_EPISODE_SIZE, self.EPOCH,
         self.EPISILON, self.c_entropy, self.td_steps, self.num_agents) = params
        self.params_name = params_name
        self.writer = writer
        self.idx = idx

        self.build_net()

        self.games = []
        for i in range(self.num_agents):
            self.games.append(Game(i, 'CartPole-v0', self.MAX_ENV_EPISODE_SIZE, self.td_steps, self.model_pi, self.model_v, self.writer))

    def build_net(self):
        # policy network
        self.model_pi = Sequential([
            Dense(100, activation='relu', input_shape=(4,)),
            Dense(2, activation='softmax')
        ])
        self.optimizer_pi = tf.keras.optimizers.Adam()

        # value network
        self.model_v = Sequential([
            Dense(100, activation='relu', input_shape=(4,)),
            Dense(1)
        ])
        self.model_v.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=['mean_absolute_error'])

    def rollout(self, game):
        return game.play()

    def train(self, epoch):
        start_time = time.time()

        num_cpus = multiprocessing.cpu_count()
        pool = Pool(num_cpus)
        results = pool.map(self.rollout, self.games)
        pool.close()
        pool.join()
        (s, v, pred_v, a, prob, d) = results[0]
        for one_output in results[1:]:
            (s1, v1, pred_v1, a1, p1, d1) = one_output

            s = np.concatenate((s, s1))
            v = np.concatenate((v, v1))
            pred_v = np.concatenate((pred_v, pred_v1))
            a = np.concatenate((a, a1))
            prob = np.concatenate((prob, p1))
            d = np.concatenate((d, d1))
            
        advantage = v + pred_v[-1] - pred_v
        with tf.GradientTape() as tape:
            predictions = self.model_pi(s)
            if self.loss_fun == LossEnum.PPO:
                loss = self.surogate_loss(a, predictions, prob, advantage)
            elif self.loss_fun == LossEnum.VANILLA:
                loss = self.vanilla_loss(a, predictions, v)
            elif self.loss_fun == LossEnum.BASELINE:
                loss = self.baseline_loss(a, predictions, advantage)
            else:
                raise NotImplementedError
        gradients = tape.gradient(loss, self.model_pi.trainable_variables)
        self.optimizer_pi.apply_gradients(zip(gradients, self.model_pi.trainable_variables))

        self.model_v.train_on_batch(s, v)

        avg_steps = len(s)/self.num_agents
        total_time = time.time() - start_time
        time_per_step = total_time/len(s)

        with self.writer.as_default():
            tf.summary.scalar('total_time', total_time, step=epoch)
            tf.summary.scalar('step_time', time_per_step, step=epoch)

    def vanilla_loss(self, a_choice, y_pred, v):
        entropy = np.mean([stats.entropy(i) for i in y_pred])
        scce = keras.losses.SparseCategoricalCrossentropy()
        loss = scce(a_choice, y_pred, sample_weight=v) + self.c_entropy * entropy
        return loss

    def baseline_loss(self, a_choice, y_pred, advantage):
        entropy = np.mean([stats.entropy(i) for i in y_pred])
        scce = keras.losses.SparseCategoricalCrossentropy()
        loss = scce(a_choice, y_pred, sample_weight=advantage) + self.c_entropy * entropy
        return loss

    def surogate_loss(self, a_choice, y_pred, pi_old, advantage):
        entropy = np.mean([stats.entropy(i) for i in y_pred])
        idx = tf.stack((tf.range(y_pred.shape[0]), a_choice), axis=1)
        pi_prob = tf.gather_nd(y_pred, idx) 
        ratio = tf.math.divide(pi_prob,pi_old) #[batchsize,] 
        s_loss = tf.math.minimum(ratio*advantage, tf.clip_by_value(ratio, 1-self.EPISILON, 1+self.EPISILON)*advantage)
        s_loss_mean = tf.reduce_mean(s_loss) + self.c_entropy * entropy
        return -s_loss

    def run(self):
        total_epoches = math.ceil(self.EPOCH / self.td_steps)
        for epoch in range(self.EPOCH):
            self.train(epoch)

if __name__ == '__main__':

    h1 = [
        ['loss_fun', [LossEnum.BASELINE]],
        ['MAX_ENV_EPISODE_SIZE', [100]],
        ['EPOCH', [300]],
        ['EPISILON', [0.0]],
        ['C_ENTROPY', [0.0, 0.3, 0.8]],
        ['TD_STEPS', [1,3,10]],
        ['NUM_AGENT', [3]]
    ]

    h2 = [
        ['loss_fun', [LossEnum.PPO]],
        ['MAX_ENV_EPISODE_SIZE', [100]],
        ['EPOCH', [9000]],
        ['EPISILON', [0.2]],
        ['C_ENTROPY', [0.0]],
        ['TD_STEPS', [3, 10,30,100]],
        ['NUM_AGENT', [1, 3, 10, 50]]
    ]

    h3 = [
        ['loss_fun', [LossEnum.VANILLA]],
        ['MAX_ENV_EPISODE_SIZE', [100]],
        ['EPOCH', [30000]],
        ['EPISILON', [0.0]],
        ['C_ENTROPY', [0.3]],
        ['TD_STEPS', [10, 30, 100]],
        ['NUM_AGENT', [3]]
    ]

   
    hp = HyperParam(h1, h2, h3)
    
    idx = 0
    while True:
        new_params = hp.next()
        if not new_params:
            sys.exit()
        print (f"agent {idx}, params: {new_params[1]}, start...")

        params_str = new_params[1]
        current_time = datetime.datetime.now().strftime("%H%M%S-%m%d")
        current_file = os.path.splitext(os.path.basename(__file__))[0]
        train_log_dir = "logs/" + current_file + '/' + current_time + "/" + params_str
        writer = tf.summary.create_file_writer(train_log_dir)

        agent = Agent(idx, *new_params, writer)
        agent.run()
        idx += 1


