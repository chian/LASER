class State:
    def __init__(self):
        self._current_phase = "initial"  # Use a private attribute to store the value

        # Define allowed phases based on the paper
        self.allowed_phases = {"initial", "search", "results", "item", "finish"}
        
        # Define allowed transitions based on the actions
        self.transitions = {
            "search_query": ("search", "results"),
            "back_to_search": ("results", "search"),
            "next_page": ("results", "results"),
            "prev_page": ("results", "results"),
            "item_selection": ("results", "item"),
            "reject_item": ("item", "results"),
            "detail_checking": ("item", "item"),
            "buy": ("item", "finish"),
        }

        self.knowledge_state = {
            "items_viewed": [],
            "items_to_check": {},
            "items_to_buy": []
        }

    @property
    def current_phase(self):
        return self._current_phase

    @current_phase.setter
    def current_phase(self, value):
        if value not in self.allowed_phases:
            raise ValueError(f"{value} is not a valid phase. Allowed phases are: {self.allowed_phases}")
        self._current_phase = value

    def transition(self, action, item_id=None, details=None):
        if action in self.transitions:
            current_state, next_state = self.transitions[action]
            if self._current_phase == current_state:
                self.current_phase = next_state
                # If transitioning to the 'item' state, update the knowledge state
                if next_state == "item":
                    if item_id is not None:
                        self.add_viewed_item(item_id)
                    if details is not None:
                        self.add_item_to_check(item_id, details)
            else:
                raise ValueError(f"Action '{action}' is not valid from the current phase '{self._current_phase}'.")
        else:
            raise ValueError(f"Action '{action}' is not recognized.")

    def add_viewed_item(self, item_id):
        self.knowledge_state["items_viewed"].append(item_id)

    def add_item_to_check(self, item_id, details):
        self.knowledge_state["items_to_check"][item_id] = details

    def add_item_to_buy(self, item_id):
        self.knowledge_state["items_to_buy"].append(item_id)

    def add_customization_options(self, item_id, options):
        if item_id in self.knowledge_state["items_to_buy"]:
            self.knowledge_state["items_to_buy"][item_id] = options

    def is_valid_action(self, action: str) -> bool:
        # Check if the action is valid for the current phase
        if action in self.transitions:
            current_state, next_state = self.transitions[action]
            return self._current_phase == current_state
        return False

    def __str__(self):
        # Return a string representation of the state for debugging/printing
        return f"Phase: {self.current_phase}, Items Viewed: {self.items_viewed}, Items to Buy: {self.items_to_buy}"

    # Add other methods as necessary for accessing or modifying the state