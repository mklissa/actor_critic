
import os
import gym
import itertools
import numpy as np
import tensorflow as tf

from estimators import PolicyEstimator, ValueEstimator


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('seed', 666, 'Random seed.')
flags.DEFINE_integer('num_episodes',1000,'Number of episodes.')



def actor_critic(sess, env, 
                estimator_policy, estimator_value, 
                num_episodes, discount_factor=.99):


    directory= os.getcwd() + '/res/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    name = "{}cartpole_seed{}.csv".format(directory,seed)


    save_every = 5
    totsteps = 0
    done=False
    stats = []      # Stats used to collect returns and save on disk
    actor_lr=1e-3         # Learning rate for the actor and the critic
    critic_lr=5e-3
    reward_list = []
    action_list = []
    states = []  

    for i_episode in range(num_episodes):

        rewards = 0.
        losses = 0.


        if i_episode % save_every == 0 and i_episode != 0:
            np.savetxt(name,stats,delimiter=',') 

        state = env.reset()
        states.append(state) 
    
        for t in itertools.count():
            

            action = estimator_policy.predict([state],sess=sess)
            next_state, reward, done, _ = env.step(action)
            rewards += reward

            reward_list.append(reward)
            action_list.append(action)
            state = next_state
            states.append(state) 


            if done:
                totsteps+=t 
                print("\rEpisode {}/{} Steps {} Total Steps {} ({}) ".format(i_episode, num_episodes, t, totsteps, rewards) )
                stats.append(totsteps)
                
                # Calculate the returns and advantage
                returns=[]
                curr_sum = 0.
                for r in reversed(reward_list):
                    curr_sum = r + discount_factor*curr_sum
                    returns.append(curr_sum)
                returns.reverse()
                advantage = returns - estimator_value.predict(states,sess=sess)[:-1]


                # Update the networks
                estimator_policy.update(np.array(states)[:-1], np.array(advantage), np.array(action_list), actor_lr,sess=sess)
                estimator_value.update(np.array(states)[:-1], returns, critic_lr,sess=sess)

                reward_list = []
                action_list = []
                states= []
                rewards = 0
                actor_lr*=.999
                break
            
            

with tf.Session() as sess:

    env = gym.envs.make("CartPole-v1")

    seed = FLAGS.seed
    np.random.seed(seed)
    env.seed(seed)
    tf.set_random_seed(seed)    
    print('seed: {} '.format(seed))

    policy_estimator = PolicyEstimator(env)
    value_estimator = ValueEstimator(env)

    sess.run(tf.global_variables_initializer())
    actor_critic(sess, env, policy_estimator,
                 value_estimator, FLAGS.num_episodes)










