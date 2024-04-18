import torch
import logging
import sys, os
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ARGO_WRAPPER.ARGO import ArgoWrapper
from ARGO_WRAPPER.CustomLLM import ARGO_LLM
from environments.hypothesis.hypothesis_environment import HypothesisEnvironment
from environments.hypothesis.state import State

def run_episode(env):
    """
    Run a single episode using the current policy and collect transitions.
    """
    states, actions, rewards, next_states, log_probs, dones = [], [], [], [], [], []
    state = env.reset()
    done = False
    while not done:
        action, log_prob, next_state, reward, done = take_action(env, state)
        state_values = [len(state.hypotheses), len(state.feedback), len(state.scores), state.previous_score if state.previous_score is not None else 0]
        states.append(torch.FloatTensor(state_values).unsqueeze(0))
        actions.append(action)
        rewards.append(reward)
        next_state_values = [len(next_state.hypotheses), len(next_state.feedback), len(next_state.scores), next_state.previous_score if next_state.previous_score is not None else 0]
        next_states.append(torch.FloatTensor(next_state_values).unsqueeze(0))
        log_probs.append(log_prob)
        dones.append(done)
        state = next_state
    return states, actions, rewards, next_states, log_probs, dones

def take_action(env, state):
    """
    Take an action given a state using the current policy.
    """
    # Updated to include previous_score. Assuming latest_feedback is not directly used for action decision.
    # Check if previous_score is None, if so, set a default value, e.g., 0
    previous_score = state.previous_score if state.previous_score is not None else 0
    state_values = [len(state.hypotheses), len(state.feedback), len(state.scores), previous_score]  # Updated line

    state_tensor = torch.FloatTensor(state_values).unsqueeze(0)
    action_probs = env.actor(state_tensor)
    dist = torch.distributions.Categorical(action_probs)
    action_index = dist.sample()
    log_prob = dist.log_prob(action_index).item()
    action_name = env.index_to_action[action_index.item()]
    next_state, reward, done, _ = env.step(action_name)
    return action_index.item(), log_prob, next_state, reward, done

def process_episode_data(env, states, actions, rewards, next_states, log_probs, dones):
    """
    Process data from an episode, calculate returns and advantages, and perform PPO update.
    """
    values = [env.critic(state).item() for state in states]
    returns, advantages = calculate_returns_and_advantages(rewards, values, dones)
    env.ppo_update(states, actions, log_probs, returns, advantages)

def calculate_returns_and_advantages(rewards, values, dones, gamma=0.99, tau=0.95):
    """
    Calculate GAE (Generalized Advantage Estimation) for returns and advantages.
    """
    returns = []
    advantages = []
    gae = 0
    next_value = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * tau * (1 - dones[t]) * gae
        returns.insert(0, gae + values[t])
        advantages.insert(0, gae)
    return returns, advantages

def main():
    #Initialize ARGO wrappers
    argo_wrapper_instance = ArgoWrapper()
    llm = ARGO_LLM(argo=argo_wrapper_instance, model="gpt4", temp=0.5)
    env = HypothesisEnvironment(llm)
    env.set_initial_hypothesis("What is an example of a phenomenon where humanity as a whole lacks a good explanation for, but, taking into account the full set of human generated knowledge, an explanation is actually possible to generate? Please write the explanation. It must not be a hypothesis that has been previously proposed. A good explanation will be hard to vary.")


    num_episodes = 10

    for episode in range(num_episodes):
        states, actions, rewards, next_states, log_probs, dones = run_episode(env)
        process_episode_data(env, states, actions, rewards, next_states, log_probs, dones)
        logging.info(f"Episode {episode+1} finished with total reward: {sum(rewards)}")

if __name__ == "__main__":
    main()
