class State:
    def __init__(self):
        self.current_phase = "initial"  # Possible phases: "initial", "hypothesis_generation", "feedback", "evaluation", "conclusion"
        self.hypotheses = []
        self.feedback = []
        self.scores = []
        self.current_hypothesis = None  # Track the current hypothesis under discussion
        self.knowledge_state = []  # Keep track of past information that is relevant and useful
        self.previous_score = None  # Add this line to keep track of the previous score
        self.latest_feedback = None  # Add this line to track the latest feedback
        self.phase_to_index = None  # Add this line to allow phase to index mapping

    def set_phase_to_index(self, phase_to_index):
        self.phase_to_index = phase_to_index

    def transition(self, action):
        # Example transition logic
        if action == "generate_initial_hypothesis":
            self.current_phase = "hypothesis_generation"
        elif action == "provide_feedback":
            self.current_phase = "feedback"
        elif action == "modify_hypothesis":
            self.current_phase = "hypothesis_generation"
        elif action == "conclude":
            self.current_phase = "conclusion"

    def add_hypothesis(self, hypothesis):
        self.hypotheses.append(hypothesis)
        self.current_hypothesis = hypothesis  # Update the current hypothesis

    def add_feedback(self, feedback):
        self.feedback.append(feedback)
        self.latest_feedback = feedback  # Update the latest feedback


    def add_score(self, score):
        self.previous_score = self.scores[-1] if self.scores else None  # Update the previous score
        self.scores.append(score)

    def modify_current_hypothesis(self, modified_hypothesis):
        self.current_hypothesis = modified_hypothesis

    def get_state_data(self, phase_to_index):
        if phase_to_index is None:
            raise ValueError("phase_to_index is not set.")
        numeric_phase = self.phase_to_index.get(self.current_phase, -1)
        return [numeric_phase, len(self.hypotheses), len(self.feedback), sum(self.scores)]

