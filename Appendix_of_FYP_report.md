
# For Ch1 Introduction



# For Ch2 Preliminaries


# For Ch3 Methodology
## Neural network


# For Ch4 Experiment and results

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







# For Ch5 Conclusion, limitations, progress and future




