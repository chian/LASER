import sys
import os
import json
import ast
import textwrap

from ..abstract_environment import AbstractEnvironment
from gymnasium import spaces
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from environments.hypothesis.state import State  # Assuming State class is defined in state.py within the same directory
from pydantic import BaseModel, ValidationError, validator
from ARGO_WRAPPER.ARGO import ArgoWrapper, ArgoEmbeddingWrapper
from ARGO_WRAPPER.CustomLLM import ARGO_LLM, ARGO_EMBEDDING
from environments.hypothesis.state import State
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from langchain_core.messages.human import HumanMessage

import logging

logging.basicConfig(level=logging.DEBUG)
DEBUG_MODE = True

class AgentResponse(BaseModel):
    role: str  # "student", "professor", or "critic"
    content: str
    feedback: Optional[str] = None  # For professor and critic responses
    quality_score: Optional[float] = None  # For critic's evaluation

    @validator('role')
    def role_must_be_valid(cls, v):
        if v not in ["student", "professor", "critic"]:
            raise ValueError('Role must be either "student", "professor", or "critic"')
        return v

class EvaluationResponse(BaseModel):
    score: float

    @validator('score')
    def score_must_be_valid(cls, v):
        if not (0 <= v <= 1):
            raise ValueError('Score must be between 0 and 1')
        return v

class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1),
        )
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
    def forward(self, x):
        return self.network(x)

class CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)

class HypothesisEnvironment(AbstractEnvironment):
    phase_to_index = {
        "initial": 0,
        "hypothesis_generation": 1,
        "feedback": 2,
        "conclusion": 3
    }

    action_to_index = {
        "generate_initial_hypothesis": 0,
        "provide_feedback": 1,
        "modify_hypothesis": 2,
        "conclude": 3
    }
    
    index_to_action = {v: k for k, v in action_to_index.items()}

    choosable_actions = ["provide_feedback", "modify_hypothesis", "conclude"]  # Initial choosable actions, excluding "generate_initial_hypothesis"
    
    def action_to_one_hot(self, action):
        vector_size = len(self.action_to_index)  # Number of actions
        one_hot_vector = [0] * vector_size
        action_index = self.action_to_index.get(action, None)
        if action_index is not None:
            one_hot_vector[action_index] = 1
        return one_hot_vector

    @staticmethod
    def phase_to_one_hot(phase, vector_size=5):
        one_hot_vector = [0] * vector_size
        phase_index = HypothesisEnvironment.phase_to_index.get(phase, -1)
        if phase_index != -1:
            one_hot_vector[phase_index] = 1
        return one_hot_vector

    def __init__(self, llm: ARGO_LLM):
        super(HypothesisEnvironment, self).__init__()
        self.action_space = spaces.Discrete(3)  # Three roles: student, professor, critic
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)
        self.llm = llm 
        self.actor = ActorNetwork(input_size=len(self.phase_to_index), hidden_size=128, output_size=len(self.choosable_actions))
        self.critic = CriticNetwork(input_size=len(self.phase_to_index), hidden_size=128)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.01)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.01)
        self.reset()

    def set_initial_hypothesis(self, initial_hypothesis):
        self.initial_hypothesis = initial_hypothesis
        print(f"Initial hypothesis set to: {self.initial_hypothesis}")  # Debug print

    def reset(self):
        # Initialize or reset the state here, for example:
        self.state = State()  # Assuming State is a class that represents the state
        # Set phase_to_index from HypothesisEnvironment to State
        self.state.set_phase_to_index(self.phase_to_index)
        self.state.current_phase = "initial"
        return self.state

    def step(self, action):
        # Existing step logic here
        logging.debug(f"Executing action: {action} in state: {self.state.current_phase}")
        # Ensure this method correctly handles the action and updates the state
        if action == "generate_initial_hypothesis":
            self.generate_initial_hypothesis(self.initial_hypothesis)
            self.state.transition(action) # Transition to the next logical phase
        elif action == "provide_feedback":
            self.provide_feedback()
            self.state.transition(action)  # Example transition, adjust as needed
        elif action == "conclude":
            self.state.transition(action)
            # Additional logic for concluding the session
        elif action == "modify_hypothesis":
            self.modify_hypothesis_based_on_feedback()
            self.state.transition(action)  # Transition back for potentially more feedback

        # After executing the action and transitioning the state, log the new state
        logging.debug(f"New state after action: {self.state.current_phase}, Reward: {self.state.previous_score}")

        # Evaluate the current hypothesis to calculate the reward
        reward = self.calculate_reward()  # This line is adjusted to use the calculate_reward method

        # Check if the environment has reached a terminal state
        done = self.state.current_phase == "conclusion"

        # Adjusted part: Directly use state attributes or a custom method instead of converting to numpy
        # Example: Assuming a method in State class that returns relevant data as a list or any non-numpy format
        state_data = self.state.get_state_data(self.phase_to_index)  # Pass phase_to_index to get_state_data
        state_tensor = torch.FloatTensor(state_data).unsqueeze(0)  # Convert the list to a tensor directly

        action_probs = self.actor(state_tensor)
        dist = torch.distributions.Categorical(action_probs)

        action_index = self.action_to_index.get(action, None)
        if action_index is None:
            raise ValueError(f"Invalid action: {action}")

        log_prob = dist.log_prob(torch.tensor(action_index))
        
        return self.state, reward, done, log_prob.item()

    def select_action(self, state):
        if state.current_phase == "initial":
            logging.info("Initial phase detected. Generating initial hypothesis.")
            selected_action = "generate_initial_hypothesis"
        else:
            logging.debug(f"Selecting action for state: {state.current_phase}")
            one_hot_vector = HypothesisEnvironment.phase_to_one_hot(state.current_phase)
            state_tensor = torch.FloatTensor(one_hot_vector).unsqueeze(0)
            
            action_probabilities = self.actor(state_tensor)
            dist = torch.distributions.Categorical(action_probabilities)
            action = dist.sample().item()
            
            selected_action = self.choosable_actions[action]  # Use choosable_actions to map index to action

        logging.info(f"Selected action: {selected_action} for phase: {state.current_phase}")

        return selected_action

    def get_llm_response(self, prompt):
        if DEBUG_MODE:
            print(f"Sending prompt to LLM: {prompt}")
        response = self.llm(prompt)  # Assuming this calls the LLM and gets the response
        if DEBUG_MODE:
            print(f"LLM Response (get_llm_response): {response}")
        return response

    def process_llm_response(self, response: dict):
        try:
            agent_response = AgentResponse(**response)
        except ValidationError as e:
            print("LLM response validation error:", e.json())

    def provide_feedback(self):
        # This method simulates the professor providing feedback on the current hypothesis
        prompt = f"Provide feedback on the hypothesis: {self.state.current_hypothesis}"
        response = self.get_llm_response(prompt)
        self.state.add_feedback(response)

    def modify_hypothesis_based_on_feedback(self):
        # Assuming the current hypothesis and feedback are stored in the state
        current_hypothesis = self.state.current_hypothesis
        feedback = self.state.latest_feedback  # Assuming there's a way to access the latest feedback

        prompt = textwrap.dedent(f"""
            You will act as a collaborative team of thinkers, each wearing a different Thinking Hat, to refine a hypothesis based on provided feedback. Your goal is to enhance the hypothesis's creativity, factual accuracy, and overall quality. The Thinking Hat roles are:

            - **Green Hat (Innovator):** Suggest creative improvements or new angles.
            - **Black Hat (Realist):** Point out factual inaccuracies or logical flaws to correct.
            - **White Hat (Analyst):** Offer evidence or data that could strengthen the hypothesis.

            Based on the feedback: '{feedback}', refine the following hypothesis: '{current_hypothesis}'
        """)
        
        # Invoke the LLM to generate a modified hypothesis
        modified_hypothesis_response = self.get_llm_response(prompt)
        
        # Assuming the response directly contains the modified hypothesis
        modified_hypothesis = modified_hypothesis_response
        logging.debug(f"Modified hypothesis: {modified_hypothesis}")
        
        # Update the current hypothesis in the state
        self.state.modify_current_hypothesis(modified_hypothesis)

    def calculate_reward(self):
        # Evaluate the hypothesis to get the current quality score
        current_score = self.evaluate_hypothesis_with_llm(self.state.current_hypothesis)
        
        # Calculate penalty for overused words
        penalty_score = penalize_overused_words(self.state.current_hypothesis)
        
        # Adjust the current score based on the penalty
        adjusted_current_score = max(0, current_score - penalty_score * 0.05)
        
        # Calculate the reward based on the difference between the current score and the previous score
        if self.state.previous_score is not None:
            reward = adjusted_current_score - self.state.previous_score
        else:
            reward = adjusted_current_score  # If there is no previous score, use the current score as the reward
        
        # Update the state with the new score
        self.state.add_score(adjusted_current_score)
        
        return reward

    def ppo_update(self, states, actions, old_log_probs, returns, advantages, clip_param=0.2):
        # Convert lists to tensors
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int64)
        old_log_probs = torch.tensor(old_log_probs)
        returns = torch.tensor(returns)
        advantages = torch.tensor(advantages)

        # Calculate current log probs and state values
        action_probs = self.actor(states)
        dist = torch.distributions.Categorical(action_probs)
        current_log_probs = dist.log_prob(actions)

        state_values = self.critic(states).squeeze(1)

        # Calculate ratios
        ratios = torch.exp(current_log_probs - old_log_probs)

        # Calculate actor loss (PPO clipped objective)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Calculate critic loss
        critic_loss = (returns - state_values).pow(2).mean()

        # Update actor and critic
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def evaluate_hypothesis_with_llm(self, hypothesis):
        if hypothesis is None or hypothesis == "None":
            print("Error: Hypothesis is None or not set.")  # Debug print
            exit(0)  # Or handle the error appropriately
        prompt = textwrap.dedent(f"""
            You will act as a Six Thinking Hat System (STHS), working together to evaluate a hypothesis and come to a group consensus on its score. The score should reflect the hypothesis's creativity, factual accuracy, and overall quality, on a scale from 0 to 1.

            Each Hat will contribute their perspective based on their expertise, and then you will collectively decide on a final score. The Thinking Hat personalities are as follows:

            - **Blue Hat (Organizer):** Oversees the evaluation process, ensuring that each Hat's perspective is considered in the final score.
            - **Green Hat (Innovator):** Assesses the hypothesis's creativity and originality.
            - **Red Hat (Empath):** Provides insight into the hypothesis's emotional impact and relevance.
            - **Yellow Hat (Optimist):** Highlights the hypothesis's strengths and potential positive outcomes.
            - **Black Hat (Realist):** Critiques the hypothesis for factual accuracy, potential risks, and flaws.
            - **White Hat (Analyst):** Analyzes the hypothesis for logical consistency and evidence-based support.

            After each Hat has provided their input, collaborate to determine a final score that reflects the consensus of the group. 

            Evaluate this hypothesis: {hypothesis}
        """)
        response = self.get_llm_response(prompt)
        if DEBUG_MODE:
            print(f"LLM Response (evaluate_hypothesis): {response}")

        # Parse and validate the response using the EvaluationResponse model
        try:
            evaluation_response = EvaluationResponse.parse_raw(response)
            score = evaluation_response.score
        except ValidationError as e:
            print("Evaluation response validation error:", e.json())
            score = 0.01  # Default score if parsing or validation fails

        return score

    def generate_initial_hypothesis(self, initial_hypothesis=None):
        hypothesis_prompt = initial_hypothesis
        print(hypothesis_prompt)
        response = self.get_llm_response(hypothesis_prompt)
        print(response)
        if DEBUG_MODE:
            print(f"LLM Response (generate_initial_hypothesis): {response}")
        self.state.add_hypothesis(response)
    
    def get_human_prompt(self):
        print("Please provide the initial information for hypothesis generation:")
        human_message = input()
        return human_message

def penalize_overused_words(hypothesis):
    overused_word_groups = {
        "rigor": ["rigor", "rigorous", "meticulous", "thorough"],
        "innovative": ["innovative", "groundbreaking", "pioneering", "cutting-edge"],
        "original": ["original", "unique", "novel", "unprecedented"],
        "creative": ["creative", "imaginative", "inventive", "ingenious"],
        "significant": ["significant", "remarkable", "noteworthy", "outstanding"],
        "robust": ["robust", "reliable", "dependable", "consistent"],
        "comprehensive": ["comprehensive", "extensive", "in-depth", "exhaustive"],
        "state-of-the-art": ["state-of-the-art", "advanced", "sophisticated", "cutting-edge"],
        "breakthrough": ["breakthrough", "game-changing", "transformative", "disruptive"],
        "paradigm-shifting": ["paradigm-shifting", "revolutionary", "trailblazing", "landmark"]
    }
    
    if hypothesis is not None:
        word_counts = Counter(re.findall(r'\b\w+\b', hypothesis.lower()))
    else:
        word_counts = Counter()
    
    penalty_score = 0
    
    for words in overused_word_groups.values():
        for word in words:
            penalty_score += word_counts[word]
    
    # Adjust the penalty score based on your criteria, here we simply count occurrences
    return penalty_score

import torch

def check_for_nan_inf(tensor):
    """Check if the tensor contains any NaN or Inf values."""
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        return False
    return True

def check_for_negative_values(tensor):
    """Check for any negative values in the tensor."""
    if (tensor < 0).any():
        return False
    return True

def check_one_hot_vector(tensor):
    """Check if the tensor is a valid one-hot encoded vector."""
    if torch.sum(tensor) == 1 and len(torch.unique(tensor)) == 2 and torch.all((tensor == 0) | (tensor == 1)):
        return True
    return False
