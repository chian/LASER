from ..abstract_environment import AbstractEnvironment
from gymnasium import spaces
import numpy as np
from state import State  # Assuming State class is defined in state.py within the same directory
from pydantic import BaseModel, ValidationError
from ARGO_WRAPPER.ARGO import ArgoWrapper

class LLMResponse(BaseModel):
    action: str
    item_id: str | None = None
    options: dict[str, str] | None = None

    @validator('action')
    def action_must_be_valid(cls, v, values, **kwargs):
        current_phase = values.get('current_phase', None)
        if current_phase == "search" and v not in ["search_items"]:
            raise ValueError('Action is not recognized for the search phase')
        elif current_phase == "results" and v not in ["select_item", "next_page", "prev_page"]:
            raise ValueError('Action is not recognized for the results phase')
        elif current_phase == "item_details" and v not in ["add_to_cart", "return_to_results"]:
            raise ValueError('Action is not recognized for the item details phase')
        return v

class WebShopEnvironment(AbstractEnvironment):
    def __init__(self, argo: ArgoWrapper):
        super(WebShopEnvironment, self).__init__()
        self.action_space = spaces.Discrete(6)  # As previously defined
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.state = self.reset()
        self.argo = argo
        
        # Extend state to include item interaction tracking
        self.items_viewed = []
        self.items_to_check = {}
        self.items_to_buy = []
        self.current_phase = "initial"

    def reset(self):
        super().reset()
        self.items_viewed = []
        self.items_to_check = {}
        self.items_to_buy = []
        return self.state

    def policy(self):
        # Adjusted implementation of policy function using state information
        if self.state.current_phase == "initial":
            user_instruction = self.state.user_instruction
            # Handle the initial search based on the user instruction
            return self.handle_search_state(user_instruction)
        elif self.state.current_phase == "search":
            # Delegate to handle_search_state for deciding the next action
            return self.handle_search_state(self.state.user_instruction)
        elif self.state.current_phase == "results":
            # Delegate to handle_result_state for deciding the next action
            return self.handle_result_state(self.state.user_instruction, self.state.current_page, self.state.total_results, self.state.items)
        elif self.state.current_phase == "item_details":
            # Delegate to handle_item_state for deciding the next action
            return self.handle_item_state(self.state.user_instruction, self.state.item_details, self.state.customization_options)
        elif self.state.current_phase == "finish":
            # Logic for finish phase, e.g., printing final results and ending the program
            print("Final results: ...")  # Placeholder for actual result printing
            return "end_program"
        else:
            # Handling for any other phases not explicitly mentioned
            print("Unhandled phase:", self.state.current_phase)
            return "no_action"

    def get_llm_response(self, prompt):
        response = self.argo.invoke(prompt)
        return response

    def process_llm_response(self, response: dict):
        try:
            llm_response = LLMResponse(**response)
            # Before proceeding, validate the action based on the current state
            if not self.state.is_valid_action(llm_response.action):
                raise ValueError(f"Action '{llm_response.action}' is not valid from the current state '{self.state.current_phase}'.")
            # Proceed with processing the validated action
        except ValidationError as e:
            print("LLM response validation error:", e.json())
        except ValueError as e:
            print("Action validation error:", e)

    def step(self, action):
        # Execute the action determined by the policy
        self.execute_action(action)
        # Update the environment's state based on the action
        # ... rest of the step logic ...
        done = False
        reward = 0
        info = {}

        return self.state, reward, done, info

    def execute_action(self, action, item_id=None, details=None):
        try:
            # Pass item_id and details to the transition method
            self.state.transition(action, item_id, details)
        except ValueError as e:
            # Handle the error, possibly by logging or returning it in the 'info' dict
            print(e)

    def handle_search_state(self, user_instruction):
        prompt = f"""You are an intelligent shopping assistant that can help users find the right item. You are given an
observation of the current web navigation session, in the following format:
Current Observation:
WebShop
Instruction:
{user_instruction}
[button] Search [button_] (generate a search query based on the user instruction and select this button to
find relevant items)
Every button in the observation represents a possible action you can take. Based on the current
observation, your task is to generate a rationale about the next action you should take. Note that if an
history of past rationales and actions is provided, you should also consider the history when generating
the rationale."""
        response = self.get_llm_response(prompt)
        return response

    def handle_result_state(self, user_instruction, current_page, total_results, items):
        prompt = f"""You are an intelligent shopping assistant that can help users find the right item. You are given an
observation of the current web navigation session, in the following format:
Current Observation:
Instruction:
{user_instruction}
[button] Back to Search [button_] (select this button to go back to the search page)
Page {current_page} (Total results: {total_results})
{items}
At this stage, you want to select an item that might match the user instruction. Note that even if an item
has non-matching details with the user instruction, it might offer different customization options to
allow you to match. E.g. an item may have color x in its name, but you can customize it to color y later,
the customization options are shown after you select the item. Thus if an item name seems relevant or
partially matches the instruction, you should select that item to check its details. If an item has been
selected before (the button has been clicked), you should not select the same item again. In other words,
do not select an item with [clicked button] item_id [clicked button_]."""
        response = self.get_llm_response(prompt)
        return response

    def handle_item_state(self, user_instruction, item_details, customization_options):
        prompt = f"""You are an intelligent shopping assistant that can help users find the right item. You are given an
observation of the current web navigation session, in the following format:
Current Observation:
Instruction:
{user_instruction}
{item_details}
{customization_options}
At this stage, you want to verify if the item matches the user instruction. You should consider the
available customization options when deciding whether an item matches the user instruction. If an item
can be customized to match the user instruction, or if the customization options cover the user
specification, it is also a good match. If the item does not match the user instruction and it does not
provide enough customization options, you can go to previous page to view other items. You can also
check the item’s description, features and reviews to view more details (Note that description, features
and reviews could be "None", do not check them again if they are already given)."""
        response = self.get_llm_response(prompt)
        return response

