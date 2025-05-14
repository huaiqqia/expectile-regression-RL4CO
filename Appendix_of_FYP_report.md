
# Abstract (updated version)
This project explores the application efficacy of the expectile regression function of the value function in reinforcement learning in combinatorial optimization tasks through empirical research. Based on the RL4CO framework, we replace the value function in the  PPO algorithm with the expectile regression function with adjustable parameters (τ), and construct  experiments on the traveling salesman problem (TSP) and the flexible job-shop scheduling problem (FJSP). The experimental results show that in the TSP scenario, the high-bit parameter τ=0.9 is better in the long term, which can effectively improve the path exploration ability of medium to large-scale problems, such as TSP20, TSP100, FJSP 10 jobs. In small-scale TSP tasks, such as TSP5, the conservative strategy with τ=0.5 is more conducive to stable convergence. Specifically, the FJSP problem shows non-monotony correlation. τ=0.7 to achieve the optimal scheduling scheme in small-scale complex instances, such as FJSP 3 jobs, but τ=0.5 unexpectedly outperforms the τ=0.7 scheme by 9.3% in performance gains in complex FJSP 10 jobs. However, time was limited and the experiments were insufficient, including the number of environments and the lack of sufficient repetition of the experiments. These findings reveal that the selection of tau parameters has significant problem scale sensitivity and structural dependence. 

**Note:** The abstract in my original FYP report contains some mistakes due to my oversight. This is the corrected version.

# For Ch1 Introduction
## Details of TSP and FJSP
### TSP
TSP is a classic CO problem belongs to the NP-hard problem. The problem is described as : Given a series of cities and the distances between each pair of cities, find the shortest possible route to visit each city once and return to the starting city (Punnen, 2007)

### FJSP

FJSP is also a research topic in CO belongs to the NP-hard problem and an extension of the classic Job Shop Scheduling Problem (JSP, the Job Shop Scheduling Problem).
FJSP involves two main sub-problems:
1.Machine Routing: Determine on which machine each process is processed.
2.Process Scheduling: Determine the processing sequence of each process on each machine (Yazdani et al., 2010).

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

![2025-05-10 23_43_43-rl4co_w11 Workspace – Weights   Biases](https://github.com/user-attachments/assets/ded31024-dafa-4dc1-9df9-4f1898fb5da1)


<img src="https://github.com/user-attachments/assets/3df7b260-e770-4dfa-86c3-d8e64edf6a3f" alt="2025-05-10 23_43_57-rl4co_w11 Workspace – Weights   Biases" width="300"/>

#### Supplement
Note that TSP5 experiment only has 500 steps.


### TSP20

![20_1944_5](https://github.com/user-attachments/assets/45f47a47-05ae-4e74-adb7-1c27b77336fb)

<img src="https://github.com/user-attachments/assets/ea700e7d-9da9-476b-90a1-d9b74f83ca4c" alt="20_1944_r" width="300"/>

#### Supplement interpretation
TSP20 experiment has 1.5k steps. When at 500 steps(the same as TSP5 experiment), tau0.7 is better than tau0.9, and Later, tau0.9 overtook tau0.7. 
tau0.9 is better in the long term. However, time was limited and the experiments were insufficient, including the number of environments and the lack of sufficient repetition of the experiments.



### TSP100

![2025-05-11 02_27_43-rl4co_w11 Workspace – Weights   Biases](https://github.com/user-attachments/assets/33a41e2c-556a-41b1-ab04-3ac353a2b414)

<img src="https://github.com/user-attachments/assets/80b0795d-a1a6-45c0-87a9-5eca42893bef" alt="2025-05-11 02_27_43-rl4co_w11 Workspace – Weights & Biases" width="300"/>



### FJSP3jobs

![Fs5 (2)](https://github.com/user-attachments/assets/17ee5d15-029c-401b-9c22-8b816b8514fb)

<img src="https://github.com/user-attachments/assets/bd689c3f-6f3c-4144-bee2-d2d766c793be" alt="Fs1 (2)" width="300"/>


### FJSP10jobs





















## Data process to the results

This project choose the best performance as results.
Then can get results value in Table 4.2: Performance comparison,  Table 4.3: Comparison between huber baseline and existing results on TSP, Table 4.4: Results on FJSP, to use the results value to make figures after that. 

```python

import pandas as pd
import numpy as np

file_path = 'F_10_Explain.csv'

try:
    df = pd.read_csv(file_path)

    available_columns = []
    for col in target_columns:
        if col in df.columns:
            available_columns.append(col)
        else:
            print(f"警告: 列 '{col}' 不存在于CSV文件中")
    
    if not available_columns:
        raise ValueError("没有找到任何目标列")
    
    column_lengths = {}
    for col in available_columns:
        valid_data = df[col].dropna()
        column_lengths[col] = len(valid_data)
        print(f"列 '{col}' 包含 {column_lengths[col]} 个有效数据点")
    
    min_length = min(column_lengths.values())
    print(f"\n最短列长度: {min_length}")
    

    aligned_data = {}
    for col in available_columns:
        valid_data = df[col].dropna().values
        # 只取前min_length个值
        aligned_data[col] = valid_data[:min_length]
    

    aligned_df = pd.DataFrame(aligned_data)
    
    results = {}
    
    for col in aligned_df.columns:
        max_value = aligned_df[col].max()
        

        if 'tau0.5' in col and 'huber' in col:
            method_name = 'Huber'
        elif 'tau0.5' in col:
            method_name = 'τ=0.5'
        elif 'tau0.7' in col:
            method_name = 'τ=0.7'
        elif 'tau0.9' in col:
            method_name = 'τ=0.9'
        else:
            method_name = col
        

        results[method_name] = max_value
        
        print(f"\n{method_name}:")
        print(f"  全局最大值: {max_value:.4f}")
    
    print("\n最终结果汇总:")
    for method, reward in results.items():
        print(f"{method}: {reward:.4f}")
    
    print("\n用于绘图的结果字典:")
    print(f"tsp_results = {{'TSP5': {results}}}")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"处理文件时出错: {e}")


```


