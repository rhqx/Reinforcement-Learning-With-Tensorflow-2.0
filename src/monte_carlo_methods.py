import tensorflow as tf
from tensorflow import keras
import numpy as np
import gym
import datetime
import os,sys
import enum
import time

from tools import HyperParam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from multiprocessing.dummy import Pool
import multiprocessing

from scipy import stats

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
    def __init__(self, idx, params, params_name):
        (self.loss_fun, self.MAX_ENV_EPISODE_SIZE, self.EPOCH,
         self.EPISILON, self.c_entropy, self.gamma, self.num_agents) = params
        self.params_name = params_name
        self.idx = idx

        current_time = datetime.datetime.now().strftime("%H%M%S-%m%d")
        current_file = os.path.splitext(os.path.basename(__file__))[0]
        train_log_dir = "logs/" + current_file + '/' + current_time + "/" + self.params_name
        self.writer = tf.summary.create_file_writer(train_log_dir)

        self.build_net()

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

    def train(self, epoch):
        start_time = time.time()
        num_cpus = multiprocessing.cpu_count()
        pool = Pool(num_cpus)
        results = pool.map(self.rollout, range(self.num_agents))
        pool.close()
        pool.join()
        s, v, a, p = results[0]
        for one_output in results[1:]:
            s1, v1, a1, p1 = one_output
            s = np.concatenate((s, s1))
            v = np.concatenate((v, v1))
            a = np.concatenate((a, a1))
            p = np.concatenate((p, p1))
            
        pred_v = np.squeeze(self.model_v(s))
        advantage = v - pred_v
        with tf.GradientTape() as tape:
            predictions = self.model_pi(s)
            if self.loss_fun == LossEnum.PPO:
                loss = self.surogate_loss(a, predictions, p, advantage)
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
            tf.summary.scalar('steps', avg_steps, step=epoch)
            tf.summary.scalar('total_time', total_time, step=epoch)
            tf.summary.scalar('step_time', time_per_step, step=epoch)
        print ("agent {}, epoch {}: steps {:.0}, total_time {:.2}, step_time {:.2}".format(self.idx, epoch, avg_steps, total_time, time_per_step))

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

    def calc_values(self, r):
        values = np.zeros_like(r, dtype=np.float32)
        values[-1] = r[-1]
        for i in reversed(range(len(r)-1)):
            values[i] = self.gamma*values[i+1] + r[i]
        return values

    def get_action(self, distribution):
        output_dim = len(distribution)
        a = np.random.choice(np.arange(output_dim), p=distribution)
        prob = distribution[a]
        return a, prob

    def rollout(self, idx):
        env = gym.make('CartPole-v0')
        env._max_episode_steps = self.MAX_ENV_EPISODE_SIZE 
        done = False
        state = env.reset()

        states = []
        actions = []
        rewards = []
        next_states = []
        probs = []
        while not done:
            distribution = np.squeeze(self.model_pi(np.expand_dims(state, 0)))
            action, prob = self.get_action(distribution)
            state_, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            probs.append(prob)
            rewards.append(reward)
            next_states.append(state_)
            state = state_

        values = self.calc_values(rewards)
        return (np.vstack(states), values, np.asarray(actions), np.asarray(probs))

    def run(self):
        for epoch in range(self.EPOCH):
            self.train(epoch)

if __name__ == '__main__':

    h1 = [
        ['loss_fun', [LossEnum.BASELINE]],
        ['MAX_ENV_EPISODE_SIZE', [100]],
        ['EPOCH', [300]],
        ['EPISILON', [0.0]],
        ['C_ENTROPY', [0.0, 0.3, 0.8]],
        ['gamma', [0.99]],
        ['NUM_AGENT', [3]]
    ]

    h2 = [
        ['loss_fun', [LossEnum.PPO]],
        ['MAX_ENV_EPISODE_SIZE', [100]],
        ['EPOCH', [300]],
        ['EPISILON', [0.2]],
        ['C_ENTROPY', [0.0, 0.3, 0.8]],
        ['gamma', [0.99]],
        ['NUM_AGENT', [3]]
    ]

    h3 = [
        ['loss_fun', [LossEnum.VANILLA]],
        ['MAX_ENV_EPISODE_SIZE', [100]],
        ['EPOCH', [300]],
        ['EPISILON', [0.0]],
        ['C_ENTROPY', [0.0, 0.3, 0.8]],
        ['gamma', [0.99]],
        ['NUM_AGENT', [3]]
    ]

   
    hp = HyperParam(h1, h2, h3)
    
    idx = 0
    while True:
        new_params = hp.next()
        if not new_params:
            sys.exit()
        print (f"agent {idx}, params: {new_params[1]}, start...")
        agent = Agent(idx, *new_params)
        agent.run()
        idx += 1



