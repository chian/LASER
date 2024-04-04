import gymnasium as gym
from gymnasium import spaces

class AbstractEnvironment(gym.Env):
    """
    This abstract class represents the template for creating custom environments.
    It defines the basic structure and required methods.
    """
    def __init__(self):
        super(AbstractEnvironment, self).__init__()
        # Define action and observation spaces
        self.action_space = None
        self.observation_space = None

    def step(self, action):
        """
        Apply the action and return the new state, reward, done, and info.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the environment to an initial state and return the initial observation.
        """
        raise NotImplementedError

    def render(self, mode='human'):
        """
        Render the environment.
        """
        pass

    def close(self):
        """
        Perform any necessary cleanup.
        """
        pass