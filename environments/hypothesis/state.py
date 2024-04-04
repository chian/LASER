class State:
    def __init__(self):
        self.current_phase = "initial"  # Possible phases: "initial", "hypothesis_generation", "feedback", "evaluation", "conclusion"
        self.hypotheses = []
        self.feedback = []
        self.scores = []
        self.current_hypothesis = None  # Track the current hypothesis under discussion
        self.knowledge_state = []  # Keep track of past information that is relevant and useful

    def transition(self, action):
        # Example transition logic
        if action == "generate_hypothesis":
            self.current_phase = "hypothesis_generation"
        elif action == "provide_feedback":
            self.current_phase = "feedback"
        elif action == "conclude":
            self.current_phase = "conclusion"

    def add_hypothesis(self, hypothesis):
        self.hypotheses.append(hypothesis)
        self.current_hypothesis = hypothesis  # Update the current hypothesis

    def add_feedback(self, feedback):
        self.feedback.append(feedback)

    def add_score(self, score):
        self.scores.append(score)
    def modify_current_hypothesis(self, modified_hypothesis):
        self.current_hypothesis = modified_hypothesis
