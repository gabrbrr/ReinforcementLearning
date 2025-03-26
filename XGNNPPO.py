import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import numpy as np
import networkx as nx
from torch.distributions import Categorical
from policy_value import PolicyNN 
from policy_value import ValueNN  
from gnns import DisNets
from gnns_acyclic import GNNIsAcyclic
import matplotlib.pyplot as plt

class PPOTrainer:
    def __init__(self, node_type, max_node, max_step, target_class, max_iters, start_node=0,gamma=0.9, eps_clip=0.2):
        self.max_node = max_node
        self.node_type=node_type
        self.max_step = max_step
        self.max_iters = max_iters
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coeff=0.001
        self.start_node=start_node
        self.dict = {0:'C', 1:'N', 2:'O', 3:'F', 4:'I', 5:'Cl', 6:'Br'}
        self.color= {0:'g', 1:'r', 2:'b', 3:'c', 4:'m', 5:'w', 6:'y'}
        self.max_poss_degree = {0: 4, 1: 5, 2: 2, 3: 1, 4: 7, 5:7, 6: 5}
        self.model_path="model_path.pth"

        self.policy_net = PolicyNN(input_dim=node_type, node_type_num=node_type)
        self.value_net = ValueNN(input_dim=node_type)
        self.target_net=DisNets() if self.node_type==7 else GNNIsAcyclic(input_dim=1)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=5e-4)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=1e-4)

        self.graph= nx.Graph()
        self.target_class = target_class
        self.criterion = nn.MSELoss()
        self.episode_rewards = [] 
        target_path= "./MUTAG/ckpt.pth" if self.node_type==7 else "./checkpoint/gnn_is_acyclic_best.pth"
        checkpoint = torch.load(target_path)
        self.target_net.load_state_dict(checkpoint['net'])
        self.load_model()
        
        

    def compute_advantage(self, rewards, values,dones):
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1]*(1- dones[step]) - values[step]
            gae = delta + self.gamma * gae*(1 - dones[step])
            advantages.insert(0, gae)
        return torch.tensor(advantages)

    def train(self):
        episode=0
        best_avg_reward = float('-inf')  
        patience = 500  
        patience_counter = 0  
        for iter in range(self.max_iters):
            states, actions, rewards, values, log_probs, dones,num_nodes= [], [], [], [], [], [], []
            self.graph_reset()
            
            total_reward = 0
            episode_rewards=[]
            with torch.no_grad():
                for step in range(self.max_step):
                    # Convert graph to input features
                    X, A = self.read_from_graph(self.graph)
                    X = torch.tensor(X, dtype=torch.float32)
                    A = torch.tensor(A, dtype=torch.float32)
                    n=self.graph.number_of_nodes()
                    start_action,start_logits,tail_action, tail_logits, log_prob = self.policy_net(X, A, n+self.node_type)
                    
                    
                    num_nodes.append(n+self.node_type)
                    value = self.value_net(X, A)
                    
                    reward, done = self.perform_action(start_action.item(), tail_action.item())
                    total_reward+=reward
                    
                    if done:
                        episode_rewards.append(total_reward)
                        print(f"Episode {episode}: Total Reward = {total_reward}")
                        total_reward = 0
                        episode+=1
                        self.graph_reset()
                        
                    states.append((X, A))
                    actions.append((start_action, tail_action))
                    rewards.append(reward)
                    values.append(value)
                    log_probs.append(log_prob)
                    dones.append(int(done))
                    

            self.episode_rewards.extend(episode_rewards)
            values.append(self.value_net(X, A).detach())
            advantages = self.compute_advantage(rewards, values,dones)
            returns = advantages + torch.tensor(values)[:len(values)-1]
            
            self.optimize(states, actions, log_probs, advantages, returns,num_nodes)
            
            if episode_rewards:
                avg_reward = sum(episode_rewards) / len(episode_rewards)
                print(f"Iteration {iter}: Average Reward = {avg_reward}")
    
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    patience_counter = 0 
                    print(f"New best average reward: {best_avg_reward}, saving model...")
                    self.save(iter, self.model_path + "_best")
                else:
                    patience_counter += 1  
                    print(f"No improvement for {patience_counter}/{patience} iterations.")
                    
                if patience_counter >= patience:
                    print("Stopping training due to lack of improvement.")
                    break
            if iter % 20 == 0:
                self.save(iter,self.model_path)

    def optimize(self, states, actions, old_log_probs, advantages, returns,num_nodes):
        for _ in range(4): 
            for state, action, old_log_prob, advantage, ret,num_node in zip(states, actions, old_log_probs, advantages, returns,num_nodes):
                X, A = state
                start_action, tail_action = action

                _,start_logits,_, tail_logits, _ = self.policy_net(X, A,num_node)
                start_prob_dist = Categorical(logits=start_logits)
                tail_prob_dist = Categorical(logits=tail_logits)

                new_log_prob = start_prob_dist.log_prob(start_action) + tail_prob_dist.log_prob(tail_action)
                ratio = torch.exp(new_log_prob - old_log_prob)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
                entropy_loss = -(start_prob_dist.entropy() + tail_prob_dist.entropy()).mean()
                policy_loss = -torch.min(surr1, surr2).mean()+self.entropy_coeff*entropy_loss
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                value_pred = self.value_net(X, A).squeeze()
                value_loss = self.criterion(value_pred, ret)
                
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

    def perform_action(self, start_action, tail_action):
        len_graph=self.graph.number_of_nodes()
        if tail_action >= len_graph:  # Add node
            self.graph.add_node(len_graph, label=tail_action - len_graph)
            self.graph.add_edge(start_action, len_graph)
        else:  # Add edge
            if not self.graph.has_edge(start_action, tail_action) and start_action!=tail_action:
                self.graph.add_edge(start_action, tail_action)
            else:
                self.graph_reset()
                return -5.0, True

        
        if self.node_type==7 and  not self.check_validity():
            self.graph_reset()
            reward_pred=-5
            done=True
        else:
            X_new, A_new = self.read_from_graph_raw(self.graph)
            X_new = torch.from_numpy(X_new)
            A_new = torch.from_numpy(A_new)
            if self.node_type==1:
                A_new=self.normalize_adj(A_new)
            logits, probs = self.target_net(X_new.float(), A_new.float())
            _, prediction = torch.max(logits, 0)
            if self.target_class == prediction:
                reward_pred = probs[prediction] - 0.5 
            else: 
                reward_pred = probs[self.target_class] - 0.5  
            done = self.graph.number_of_nodes() >= self.max_node
                
                
        return reward_pred, done
    
    def check_validity(self):
        node_types = nx.get_node_attributes(self.graph,'label')
        for i in range(self.graph.number_of_nodes()):
            degree = self.graph.degree(i)
            max_allow = self.max_poss_degree[node_types[i]]
            if(degree> max_allow):
                return False
        return True

    def read_from_graph(self, graph): 
        n = graph.number_of_nodes()
        
        if isinstance(self.target_net, GNNIsAcyclic): 
            F = np.zeros((self.max_node + self.node_type, 1))  
            degrees = np.array([deg for _, deg in graph.degree()]).reshape(-1, 1)
            F[:n, 0] = degrees.flatten()  
            F[n:n + self.node_type, 0] = 0  
        else:
            F = np.zeros((self.max_node + self.node_type, self.node_type))
            attr = nx.get_node_attributes(graph, "label")
            targets = np.array(list(attr.values())).reshape(-1)
            F[:n, :] = np.eye(self.node_type)[targets]
            F[n:n + self.node_type, :] = np.eye(self.node_type)
        
        E = np.zeros((self.max_node + self.node_type, self.max_node + self.node_type))
        E[:n, :n] = np.asarray(nx.to_numpy_array(graph), dtype=np.float32)
        E += np.eye(self.max_node + self.node_type)
        
        return F, E



    def read_from_graph_raw(self, graph):
        n = graph.number_of_nodes()
        
        if isinstance(self.target_net, GNNIsAcyclic):  
            F = np.array([[deg] for _, deg in graph.degree()], dtype=np.float32) 
        else:
            attr = nx.get_node_attributes(graph, "label")
            targets = np.array(list(attr.values())).reshape(-1)
            F = np.eye(self.node_type)[targets]
        
        E = np.asarray(nx.to_numpy_array(graph), dtype=np.float32)
        
        return F, E



    def graph_reset(self):
        self.graph.clear()
        self.graph.add_node(0, label= self.start_node) 
        return 
    
    def draw_graph(self, save_path):
        plt.figure(figsize=(8, 6))
        attr = nx.get_node_attributes(self.graph, "label")
        labels = {n: self.dict[attr[n]] for n in attr}
        
        if self.node_type == 1:
            color = [self.color[0]] * len(attr)  # Single color
        else:
            color = [self.color[attr[n]] for n in attr]  # Multi-color
        
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, labels=labels, node_color=color, with_labels=False if self.node_type == 1 else True)
        plt.text(0.05, 0.95, f"Score: {self.score:.4f}", transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        plt.savefig(save_path)
        plt.close()


    def save(self, episode, model_path):
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
        }
        torch.save(checkpoint, model_path)
        print(f"Checkpoint saved at episode {episode}")

    def load_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}...")
            checkpoint = torch.load(self.model_path)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
            self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            print("Model loaded successfullyusid.")
        else:
            print(f"No existing model found at {self.model_path}. Starting from scratch.")

    def generate_optimal_graph(self):
        self.graph_reset()
        tot_reward=0
        with torch.no_grad():
            for step in range(self.max_step):
                X, A = self.read_from_graph(self.graph)
                X = torch.tensor(X, dtype=torch.float32)
                A = torch.tensor(A, dtype=torch.float32)
                n = self.graph.number_of_nodes()
                
                start_action, _, tail_action, _, _ = self.policy_net(X, A, n + self.node_type)
                reward, done = self.perform_action(start_action.item(), tail_action.item())
                
                tot_reward+=reward
                if done:
                    print("Done after", step,"steps with reward", tot_reward)
                    break
        
        X_new, A_new = self.read_from_graph_raw(self.graph)
        X_new = torch.from_numpy(X_new).float()
        A_new = torch.from_numpy(A_new).float()
        if self.node_type==1:
                A_new=self.normalize_adj(A_new)
        A_new=self.normalize_adj(A_new)
        logits, probs = self.target_net(X_new, A_new)
        
        _, prediction = torch.max(logits, 0)
        self.score = probs[prediction].item()
        return self.graph
    
    def save_visualizations(self):
        folder_name = f"results_{self.node_type}_{self.max_node}_{self.start_node}_{self.target_class}"
        os.makedirs(folder_name, exist_ok=True)
    
        self.generate_optimal_graph()
    
        graph_path = os.path.join(folder_name, "graph.png")
        self.draw_graph(graph_path)
    
        rewards_path = os.path.join(folder_name, "rewards.png")
        self.plot_rewards(rewards_path)

    
    
    
    def plot_rewards(self, save_path):
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, label="Episode Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title("Reward Progression Over Episodes")
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def normalize_adj(self,adj):
        adj = adj.numpy()
        N = adj.shape[0]
        adj = adj + np.eye(N)
        D = np.sum(adj, axis=0)
        D_hat = np.diag(D**-0.5)
        out = np.dot(D_hat, adj).dot(D_hat)
        return torch.tensor(out, dtype=torch.float32)
