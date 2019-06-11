
# Navigation



In this project we use the Unity ML-Agents environment to teach an DRL-agent to collect yellow bananas:

<img src='images/banannasTest3.gif' style='zoom:50%'>

Running the code is done via the jupyter notebook : Navigation.ipynb.

The environment is considered **solved** when the agent obtains an average score greater than 13 on 100 consecutive episodes.

In the proceeding sections I explain how to run the code as well as what DRL setup was used to solve the environment.

### 1. Starting the Environment

Begin by importing some necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).


```python
from unityagents import UnityEnvironment
import numpy as np
from matplotlib import pyplot as plt
import torch
```

Next, we will start the environment!  **_Before running the code cell below_**, change the `file_name` parameter to match the location of the Unity environment that you downloaded.

- **Mac**: `"path/to/Banana.app"`
- **Windows** (x86): `"path/to/Banana_Windows_x86/Banana.exe"`
- **Windows** (x86_64): `"path/to/Banana_Windows_x86_64/Banana.exe"`
- **Linux** (x86): `"path/to/Banana_Linux/Banana.x86"`
- **Linux** (x86_64): `"path/to/Banana_Linux/Banana.x86_64"`
- **Linux** (x86, headless): `"path/to/Banana_Linux_NoVis/Banana.x86"`
- **Linux** (x86_64, headless): `"path/to/Banana_Linux_NoVis/Banana.x86_64"`

For instance, if you are using a Mac, then you downloaded `Banana.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
```
env = UnityEnvironment(file_name="Banana.app")
```

The environment info:

    INFO:unityagents:
    'Academy' started successfully!
    Unity Academy name: Academy
            Number of Brains: 1
            Number of External Brains : 1
            Lesson number : 0
            Reset Parameters :
    		
    Unity brain name: BananaBrain
            Number of Visual Observations (per agent): 0
            Vector Observation space type: continuous
            Vector Observation space size (per agent): 37
            Number of stacked Vector Observation: 1
            Vector Action space type: discrete
            Vector Action space size (per agent): 4


Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.


```python
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
```

As can be seen above, the action space is consisted of **4** actions : forward, backward, left and right.

The observation (state) consists of **37** dimensions which include mainly information about the surrounding environment of the agent + motion.

The environment supports multi agent (brains), but here we will use only one.  



### 4. Define the agent and training:

Initializing an agent:


```python
agent = Agent(state_size=37, action_size=4, seed=0, fc1_units=50, fc2_units=40)
```

The the basic network interface:

**Input :** state (37 dimension)

**Output :** action-value for each action (4 dimensions)

    QNetwork(
      (fc1): Linear(in_features=37, out_features=50, bias=True)
      (fc2): Linear(in_features=50, out_features=40, bias=True)
      (fc3): Linear(in_features=40, out_features=4, bias=True)
    )

An agent is comprised of 2 basic networks - one **local-network** and one **target-network** and a **replay buffer** which stores 100,000 of the last experience tuple (state, action, reward, next state, done). The local-network is used to select actions in the **act** function for every step in the episode.

In the **learn** function the local-network along with the states-actions are used to obtain a **expected values**, the target-network is used to assess the next-state value which is used as the **target values**. Finally the optimization step changes the weights of the local-network to reduce the difference between the the target values and the expected values. 

Once every **UPDATE_EVERY** the local-network is copied via a soft update (**TAU**) to the target network. This is done to enable a policy assessment period before updating the network.



Chosen hyper parameters:

```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 2e-3            # for soft update of target parameters  # ROEE mult by 2
LR = 5e-4               # learning rate 
UPDATE_EVERY = 8
```



The main learning loop is evoked by calling:

```python
scores = dqn(n_episodes=800, max_t=300, eps_start=1.0, eps_end=0.01, eps_decay=0.995)
```

Which yields the following:

    Episode 50	Eps 0.78	Average Score: 0.46
    Episode 100	Eps 0.61	Average Score: 1.01
    Episode 150	Eps 0.47	Average Score: 2.29
    Episode 200	Eps 0.37	Average Score: 4.45
    Episode 250	Eps 0.29	Average Score: 6.84
    Episode 300	Eps 0.22	Average Score: 7.86
    Episode 350	Eps 0.17	Average Score: 9.03
    Episode 400	Eps 0.13	Average Score: 10.50
    Episode 450	Eps 0.10	Average Score: 12.00
    Episode 500	Eps 0.08	Average Score: 13.55
    Episode 550	Eps 0.06	Average Score: 14.77
    Episode 600	Eps 0.05	Average Score: 15.46
    Episode 650	Eps 0.04	Average Score: 15.41
    Episode 700	Eps 0.03	Average Score: 15.10
    Episode 750	Eps 0.02	Average Score: 15.02
    Episode 800	Eps 0.02	Average Score: 15.63



<img src='images/output_9_1.png'>


## Test 100 episodes:


```python
agent.qnetwork_local.eval()
agent.qnetwork_target.eval()
with torch.no_grad():
    scores = dqn(n_episodes=100, eps_start=0.01)
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
```

    Episode 50	Eps 0.01	Average Score: 15.90
    Episode 100	Eps 0.01	Average Score: 15.75



<img src='images/output_11_1.png'>



# Saving testing and loading the agent 

### Make sure model weights are saved:


```python
torch.save(agent.qnetwork_local.state_dict(), r'checkpoint600_37fc50fc40fc4.pth')
```
I also saved the replay buffer for continuing the learning stage where it was stopped: 

```python
import pickle
memory = agent.memory.memory.copy()
with open('checkpoint600_37fc50fc40fc4_memory_pickelTest1.dat','wb') as outf:
    for mem in memory:
        for field in mem:
            pickle.dump(field, outf)
```

### Test model (visually):


```python
import time
agent.qnetwork_local.eval()
agent.qnetwork_target.eval()
env_info = env.reset(train_mode=False)[brain_name]  # Roee
#state = env.reset()  # Roee : added commented
state = env_info.vector_observations[0]

score = 0
steps = 0
with torch.no_grad():
    while True:
        steps+=1
        action = agent.act(state, 0)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        state = next_state
        score += reward
        time.sleep(0.05)
        if reward==1:
            print("Yellow!")
        elif reward==-1:
            print("Blue...")
        if done:
            break

            
print('Number of steps = ' + str(steps))
print('Score = ' + str(score))
```



## Loading previously trained agent (memory and network weights):


```python
from dqn_agent import Agent

agent = Agent(state_size=37, action_size=4, seed=0, fc1_units=50, fc2_units=40)

import pickle
buffSize = int(1e5)

with open('checkpoint600_37fc50fc40fc4_memory_pickelTest1.dat','rb') as inpf:
    for i in range(buffSize):
        state, action, reward, next_state, done = (pickle.load(inpf), pickle.load(inpf), pickle.load(inpf), pickle.load(inpf), pickle.load(inpf))
        agent.memory.add(state, action, reward, next_state, done)
        
import torch
state_dict = torch.load('checkpoint600_37fc50fc40fc4.pth')
agent.qnetwork_local.load_state_dict(state_dict)
agent.qnetwork_target.load_state_dict(state_dict)
```



Closing the environment:

```python
env.close()
```



### Future improvements:

* Implement dueling DRL setup
* Implement Prioritized Experience Replay
* Increase state size by adding previous states to input
* Tweak some more with the hyper parameters.