
# For Ch1 Introduction
## Details of TSP and FJSP
### FJSP
FJSP (Flexible Job Shop Scheduling Problem) is an important research topic in the field of CO (Combinatorial Optimization). It is an NP-hard Problem and an extension of the classic Job Shop Scheduling Problem (JSP, the Job Shop Scheduling Problem).
FJSP involves two main sub-problems:
Machine Routing: Determine on which machine each process is processed (Yazdani et al., 2010).
Process Scheduling: Determine the processing sequence of each process on each machine (Yazdani et al., 2010).

# For Ch2 Preliminaries
## Figure of expectile function 

# For Ch3 Methodology
## Neural network





# For Ch4 Experiment and results

## Hyperparameters set
### TSP5


### TSP20

### TSP100

### FJSP 3 jobs

### FJSP 10 jobs







## train.py (for TSP tau experiments)
```python
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning.pytorch.loggers import WandbLogger

from rl4co.envs import TSPEnv
from p_1215_0156 import PPO

from rl4co.utils.trainer import RL4COTrainer                  
from rl4co.models.zoo.am.policy import AttentionModelPolicy       

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TSPEnv(generator_params=dict(num_loc=5))                                                              
    policy = AttentionModelPolicy(env_name=env.name,
                                  embed_dim=128,
                                  )
    tau = 0.7                                                                                              
    gamma = 0.99

    model = PPO(
        env,
        policy=policy,
        mini_batch_size=32,                        
        batch_size=128,                      
        val_batch_size=64,
        test_batch_size=64,
        train_data_size=500,                          
        val_data_size=100,                           
        test_data_size=100,
        tau=tau,
        gamma=0.99,
        optimizer_kwargs={"lr": 3e-4},          
        entropy_lambda=0.01,
        clip_range=0.2,
        ppo_epochs=3,                                 
        normalize_adv=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",  
        filename="epoch_{epoch:03d}",  
        save_top_k=1,  
        save_last=True,  
        monitor="val/reward",  
        mode="max",  
    )

    rich_model_summary = RichModelSummary(max_depth=3)  
    callbacks = [checkpoint_callback, rich_model_summary]

    logger = WandbLogger(project="rl4co_w11", name="A_expectile_tau0.7_numloc=5",
                         config={"tau": tau})   

    trainer = RL4COTrainer(
        max_epochs=100,                               
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    trainer.fit(model)

    td_init = env.reset(batch_size=[128]).to(device)    
    policy = model.policy.to(device)  
    out = model(
        td_init.clone(), 
        env=env,  
        phase="test",
        decode_type="greedy",
        return_actions=True,
    )

    mean_reward = torch.mean(out['reward']).item()
    std_reward = torch.std(out['reward']).item()
    print(f"Avg tour length: {-mean_reward:.3f}")
    print(f"Reward standard deviation: {std_reward:.3f}")
    
if __name__ == "__main__":
    main()
```


## Tsp5_bruteforce.py
```python
import torch
import itertools
from rl4co.envs import TSPEnv

def brute_force_tsp(coordinates):
    num_cities = coordinates.shape[0]
    best_tour = None
    best_length = float('inf')
    
    for tour in itertools.permutations(range(num_cities)):
        tour = list(tour) + [tour[0]]  # 添加回起点
        length = sum(torch.norm(coordinates[tour[i]] - coordinates[tour[i+1]], p=2) for i in range(num_cities))
        if length < best_length:
            best_length = length
            best_tour = tour
    
    return best_tour, best_length

def main_brute_force():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = TSPEnv(generator_params=dict(num_loc=5))
    
    val_data_size = 100
    total_reward = 0
    
    for i in range(val_data_size):
        print(f"Processing iteration {i+1}/{val_data_size}")
        td_init = env.reset(batch_size=[1])
        print("TensorDict keys:", td_init.keys())
        
        if 'locs' in td_init:
            coordinates = td_init['locs'][0].to(device)
        else:
            print("Error: Could not find 'locs' in TensorDict. Available keys:", td_init.keys())
            return  # 终止程序，因为这是一个关键错误
        
        best_tour, best_length = brute_force_tsp(coordinates)
        reward = -best_length
        total_reward += reward
        
        print(f"Iteration {i+1} reward: {reward:.6f}")
    
    avg_reward = total_reward / val_data_size
    print(f"Brute Force - Average val/reward: {avg_reward:.6f}")

if __name__ == "__main__":
    main_brute_force()
```






# For Ch5 Conclusion, limitations, progress and future

## Wandb screenshot

### TSP5
### TSP20
### TSP100
### FJSP3jobs

![Fs5 (2)](https://github.com/user-attachments/assets/17ee5d15-029c-401b-9c22-8b816b8514fb)

<img src="https://github.com/user-attachments/assets/bd689c3f-6f3c-4144-bee2-d2d766c793be" alt="Fs1 (2)" width="300"/>


### FJSP10jobs





















## Data process to the results


