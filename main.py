from XGNNPPO import PPOTrainer
ppo=PPOTrainer(1,3,64,0,1000)
ppo.train()
ppo.save_visualizations()
