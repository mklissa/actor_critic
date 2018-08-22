import numpy as np
import tensorflow as tf


class PolicyEstimator():
    """
    Policy neural network. 
    """
    def __init__(self, env, entropy_coeff=1e-3, hidden_size=24, scope="policy_estimator"):
        """
        This ``PolicyEstimator`` implements the neural network used to output the actions' probabilities of the environment's states. 
        It is also updated at the end of every episode in order to improve it's accuracy.

        Args:
            env (Gym environment) : the environment that we are training our reinforcement learning.
            entropy_coeff (float) : the coefficient multiplying the entropy which is used in the loss function.
            hidden_size (int) : the size of the neural networks' hidden layers.
            scope (str) : the scope used to define TensorFlow's parameters used by the neural network.

        """

        with tf.variable_scope(scope):

            # Define the placeholders
            self.state = tf.placeholder(tf.float32, [None,env.observation_space.shape[0]], "state")
            self.action = tf.placeholder(tf.int32, name="action")
            self.advantage = tf.placeholder(dtype=tf.float32, name="advantage")
            self.lr = tf.placeholder(dtype=tf.float32, name='learnrate')
            

            # Define the network
            hid = self.state

            hid = tf.contrib.layers.fully_connected(
                inputs=hid,
                num_outputs=hidden_size,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer()
                )            
            
            hid = tf.contrib.layers.fully_connected(
                inputs=hid,
                num_outputs=hidden_size,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer()
                )   
            
            self.actions = tf.contrib.layers.fully_connected(
                inputs=hid,
                num_outputs=env.action_space.n,
                activation_fn=tf.nn.softmax,
                weights_initializer=tf.contrib.layers.xavier_initializer()
                )[0]


            # Define the loss and optimizer
            entropy = - tf.reduce_sum( tf.log(self.actions)*self.actions )
            self.loss = - tf.log( self.actions[self.action]) * self.advantage
            self.loss -= entropy_coeff * entropy

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())
    
    def predict(self,state,sess=None):
        """
        The policy will choose an action given a state.

        Args:
            state : the state for which we want to select an action.
            sess : TensorFlow session.

        Returns:
            action: the selected action given the state.

        """
        actions = sess.run(self.actions, { self.state: state })
        return np.random.choice(len(actions),p=actions)

    def update(self, states, advantages, actions, lr, sess=None):
        """
        Update the policy's weights from a batch of states, advantages and actions.

        Args:
            states : the states from the past episode.
            advantages : the advantage function calculated for each of the given state.
            actions: the actions selected for each of the given state.
            lr: the current learning rate used for the updates.
            sess : TensorFlow session.

        """
        for state,advantage,action in zip(states, advantages, actions):
            feed_dict = { self.state: [state], self.advantage: advantage, self.action: action, self.lr: lr  }
            sess.run([self.train_op, self.loss], feed_dict)





class ValueEstimator():
    """
    Value function neural network. 
    """
    
    def __init__(self, env,  hidden_size=24, scope="value_estimator"):
        """
        This ``ValueEstimator`` implements the neural network used to predict the value of the environment's states. 
        It is also updated at the end of every episode in order to improve it's accuracy.

        Args:
            env (Gym environment) : the environment that we are training our reinforcement learning.
            hidden_size (int) : the size of the neural networks' hidden layers.
            scope (str) : the scope used to define TensorFlow's parameters used by the neural network.

        """
        with tf.variable_scope(scope):

            # Define the placeholders
            self.state = tf.placeholder(tf.float32, [None,env.observation_space.shape[0]], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")
            self.lr = tf.placeholder(dtype=tf.float32, name='learnrate')
            

            #Define the network
            hid= self.state

            hid = tf.contrib.layers.fully_connected(
                inputs=hid,
                num_outputs=hidden_size,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer()
                )

            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=hid,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer()
                )


            # Define the the loss and optimizer
            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.train.get_global_step())  
                    

    def predict(self, state, sess=None):
        """
        The policy will predict a value for a given state

        Args:
            state : the state for which we want to predict the value.
            sess : TensorFlow session.

        Returns:
            val: the value of the given state.

        """        
        val = sess.run(self.value_estimate, { self.state: state })
        return val 

    def update(self, state, targets, lr, sess=None):
        """
        Update the value function's weights from a batch of states and targets.

        Args:
            states : the states from the past episode.
            targets : the targets calculated for each of the states.
            lr: the current learning rate used for the updates.
            sess : TensorFlow session.

        """        
        feed_dict = { self.state: state, self.target: targets, self.lr:lr }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


