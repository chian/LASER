import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ARGO_WRAPPER.ARGO import ArgoWrapper, ArgoEmbeddingWrapper
from ARGO_WRAPPER.CustomLLM import ARGO_LLM, ARGO_EMBEDDING
from environments.hypothesis.hypothesis_environment import HypothesisEnvironment
from environments.hypothesis.state import State

# Setup basic configuration for logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Initialize ARGO wrappers
    argo_embedding_wrapper_instance = ArgoEmbeddingWrapper()    
    argo_embedding = ARGO_EMBEDDING(argo_embedding_wrapper_instance)
    
    argo_wrapper_instance = ArgoWrapper()
    llm = ARGO_LLM(argo=argo_wrapper_instance, model_type='gpt4', temperature=0.5)

    # Initialize the environment
    env = HypothesisEnvironment(llm=llm)

    num_episodes = 10  # Define the number of episodes

    for episode in range(num_episodes):
        logging.info(f"Starting episode {episode+1}/{num_episodes}")
        state = env.reset()  # Reset the environment at the start of each episode
        done = False
        step_count = 0

        while not done:
            step_count += 1
            logging.debug(f"Episode {episode+1}, Step {step_count}: Current state: {state.current_phase}")

            # Select an action based on the current state
            action = env.select_action(state)
            logging.debug(f"Episode {episode+1}, Step {step_count}: Selected action: {action}")

            # Execute the action and update the environment
            next_state, reward, done = env.step(action)
            logging.debug(f"Episode {episode+1}, Step {step_count}: Action executed. Next state: {next_state.current_phase}, Reward: {reward}, Done: {done}")

            # Update the policy based on the reward
            env.update_policy(action, reward, state)

            # Log the results of the action
            log_results(action, next_state, reward, done)

            # Update the current state
            state = next_state

        logging.info(f"Episode {episode+1} concluded.")

def log_results(action, next_state, reward, done):
    # Implement logging of action results
    logging.info(f"Action: {action}, Reward: {reward}, Done: {done}")

if __name__ == "__main__":
    main()