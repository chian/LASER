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

class EvaluationScore(BaseModel):
    score: float

    @validator('score')
    def score_must_be_in_range(cls, v):
        if not (0 <= v <= 1):
            raise ValueError('Score must be between 0 and 1')
        return v

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.initialize_network()

    def forward(self, x):
        x = self.network(x)
        return torch.softmax(x, dim=-1)  # Apply softmax here

    def initialize_network(self):
        # Initialize the weights and biases of the last layer to zero
        if isinstance(self.network[-1], nn.Linear):
            nn.init.constant_(self.network[-1].weight, 0)
            nn.init.constant_(self.network[-1].bias, 0)
    
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
        "generate_hypothesis": 1,
        "provide_feedback": 2,
        "modify_hypothesis": 3,
        "conclude": 4
    }

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
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)  # Adjusted shape to (5,)
        self.llm = llm
        # Initialize the policy network with input_size=5
        self.policy_network = PolicyNetwork(input_size=5, hidden_size=128, output_size=3)  # Adjusted input_size to 5
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.01)
        self.reset()  # Call reset to initialize the state

    def reset(self):
        # Initialize or reset the state here, for example:
        self.state = State()  # Assuming State is a class that represents the state
        return self.state

    def step(self, action):
        # Ensure this method correctly handles the action and updates the state
        if action == "generate_initial_hypothesis":
            self.generate_initial_hypothesis()
            self.state.transition("hypothesis_generation")  # Transition to the next logical phase
        elif action == "generate_hypothesis":
            self.generate_hypothesis()
            self.state.transition("feedback")  # Example transition, adjust as needed
        elif action == "provide_feedback":
            self.provide_feedback()
            self.state.transition("modify_hypothesis")  # Example transition, adjust as needed
        elif action == "conclude":
            self.state.transition("conclusion")
            # Additional logic for concluding the session
        elif action == "modify_hypothesis":
            self.modify_hypothesis_based_on_feedback()
            self.state.transition("feedback")  # Transition back for potentially more feedback

        reward = self.calculate_reward()
        done = self.state.current_phase == "conclusion"

        return self.state, reward, done

    def select_action(self, state):
        if state.current_phase == "initial":
            # Dynamically generate next_phases from action_to_index keys, excluding the initial action
            # This assumes that the keys in action_to_index correspond to valid actions for transitioning phases
            next_phases = [action for action in self.action_to_index.keys() if action != "generate_initial_hypothesis"]
            
            # Choose a random next phase from the dynamically generated list
            random_next_phase = np.random.choice(next_phases)
            
            # Transition to the randomly chosen next phase
            state.transition(random_next_phase)  # Assuming a transition method exists in the State class.
            
            # Return the action that corresponds to generating the initial hypothesis
            return "generate_initial_hypothesis"
        else:
            one_hot_vector = HypothesisEnvironment.phase_to_one_hot(state.current_phase)
            state_tensor = torch.FloatTensor(one_hot_vector).unsqueeze(0)
            
            if not check_for_nan_inf(state_tensor):
                print("Input contains NaN or Inf values.")
                return None
            if not check_for_negative_values(state_tensor):
                print("Input contains negative values.")
                return None
            if not check_one_hot_vector(state_tensor):
                print("Input is not a valid one-hot encoded vector.")
                return None
            
            action_probabilities = self.policy_network(state_tensor)
            action = torch.multinomial(action_probabilities, 1).item()
            return action

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

    def generate_hypothesis(self):
        # This method simulates the student generating a hypothesis
        prompt = "Generate a hypothesis based on the current knowledge."
        response = self.get_llm_response(prompt)
        self.state.add_hypothesis(response['content'])

    def provide_feedback(self):
        # This method simulates the professor providing feedback on the current hypothesis
        prompt = f"Provide feedback on the hypothesis: {self.state.current_hypothesis}"
        response = self.get_llm_response(prompt)
        self.state.add_feedback(response['feedback'])

    def modify_hypothesis_based_on_feedback(self):
        # Assuming the current hypothesis and feedback are stored in the state
        current_hypothesis = self.state.current_hypothesis
        feedback = self.state.latest_feedback  # Assuming there's a way to access the latest feedback

        # Generate a prompt for modifying the hypothesis based on feedback
        prompt = f"Based on the feedback: '{feedback}', modify the following hypothesis: '{current_hypothesis}'"
        
        # Invoke the LLM to generate a modified hypothesis
        modified_hypothesis_response = self.get_llm_response(prompt)
        
        # Assuming the response contains a 'content' field with the modified hypothesis
        modified_hypothesis = modified_hypothesis_response['content']
        
        # Update the current hypothesis in the state
        self.state.modify_current_hypothesis(modified_hypothesis)

    def calculate_reward(self):
        # Evaluate the hypothesis to get the quality score
        quality_score = self.evaluate_hypothesis_with_llm(self.state.current_hypothesis)
        
        # Calculate penalty for overused words
        penalty_score = penalize_overused_words(self.state.current_hypothesis)
        
        # Adjust the quality score based on the penalty
        adjusted_score = max(0, quality_score - penalty_score * 0.05)
        
        return adjusted_score

    def evaluate_hypothesis_with_llm(self, hypothesis):
        prompt = textwrap.dedent(
            f"""
            You are an AI capable of evaluating scientific hypotheses. After evaluating a hypothesis, 
            you will provide a score between 0 and 1, where 0 means the hypothesis is completely 
            invalid and 1 means it is fully valid. Your response should be in JSON format, including 
            the score and any feedback you might have. For example, if you find a hypothesis to be 
            moderately plausible, you might return {{"score": 0.5, "feedback": "The hypothesis is 
            plausible but lacks empirical evidence."}}.

            Evaluate this hypothesis: {hypothesis}
        """)
        response = self.get_llm_response(prompt)
        if DEBUG_MODE:
            print(f"LLM Response (evaluate_hypothesis): {response}")  # Debugging print statement

        # Assuming the response is a string in JSON format, parse it into a Python dictionary
        try:
            response_dict = json.loads(response)
            if DEBUG_MODE:
                print(response_dict)
            evaluation_result = EvaluationScore(**response_dict)
            return evaluation_result.score
        except json.JSONDecodeError:
            if DEBUG_MODE:
                print("Failed to decode LLM response as JSON.")
            return 0.01  # Return a default score in case of JSON parsing failure
        except ValidationError as e:
            if DEBUG_MODE:
                print(f"Validation error for evaluation score: {e.json()}")
            return 0.01  # Return a default score in case of validation failure

    def update_policy(self, action, reward, current_state):
        one_hot_vector = HypothesisEnvironment.phase_to_one_hot(current_state.current_phase)
        state_tensor = torch.FloatTensor(one_hot_vector).unsqueeze(0)
        action_hot_vector = self.action_to_one_hot(action)
        action_tensor = torch.FloatTensor(action_hot_vector).unsqueeze(0)
        log_prob = torch.log(self.policy_network(state_tensor)[0, action_tensor.long()])
        loss = -log_prob * reward
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def generate_initial_hypothesis(self):
        human_prompt = self.get_human_prompt()
        initial_hypothesis_prompt = textwrap.dedent(
            f"""
            Based on the following information, generate an initial hypothesis that involves science, 
            requires critical thinking, demonstrates original thoughts, and is based in facts: 
            {human_prompt}
            """
        )
        response = self.get_llm_response(initial_hypothesis_prompt)
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

# Example usage:
state_tensor = torch.FloatTensor([0, 1, 0, 0, 0])  # Example one-hot encoded vector

if not check_for_nan_inf(state_tensor):
    print("Input contains NaN or Inf values.")
if not check_for_negative_values(state_tensor):
    print("Input contains negative values.")
if not check_one_hot_vector(state_tensor):
    print("Input is not a valid one-hot encoded vector.")
