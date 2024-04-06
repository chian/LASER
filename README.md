# LASERplus - Refactor + RL from original LASER repo

This codebase is designed to facilitate the development and deployment of intelligent agents capable of navigating and performing tasks within web-based environments, specifically tailored for web shopping and hypothesis generation scenarios. It leverages reinforcement learning (RL) principles, where agents learn to make decisions to maximize a cumulative reward. The agents interact with a simulated web environment, performing actions such as searching, selecting items, and navigating through pages based on the state of the environment and a specified reward function. The environment's state variables and the reward function can be customized to guide the agent toward completing specific tasks, such as finding a product that meets certain criteria or generating a scientific hypothesis.
 
## Detailed Examples with Pseudocode
 
## Web Shopping Scenario
In the web shopping scenario, the agent is tasked with finding and selecting products that match a user's instructions. The environment simulates a web shop, and the agent's actions include searching for products, selecting products to view more details, and navigating through search results.
 
## Pseudocode for web shopping scenario

```bash
initialize environment with web shop parameters
while not done:
    observe current state
    decide on action based on state (e.g., search, select item, navigate)
    perform action
    receive reward based on action outcome
    update agent's policy based on reward
```

##Hypothesis Generation Scenario
 
In the hypothesis generation scenario, the agent is tasked with generating a scientific hypothesis based on provided information. This scenario is more abstract and focuses on the agent's ability to process and generate text-based responses.
 
## Pseudocode for hypothesis generation scenario

```bash
initialize environment with hypothesis generation parameters
while not done:
    observe current state (e.g., provided information)
    generate hypothesis based on current state
    perform action (e.g., submit hypothesis)
    receive reward based on hypothesis quality
    update agent's policy based on reward
```
 
The environment for hypothesis generation is set up in environments/hypothesis/hypothesis_environment.py, where it defines how the agent interacts with the environment to generate hypotheses and receive feedback.
 
## Customization of Reward Function and State Variables
 
Both scenarios allow for the customization of the reward function and state variables. The reward function can be defined to incentivize certain behaviors or outcomes, such as finding a product that exactly matches the user's needs or generating a highly original hypothesis. State variables can include any relevant information that the agent needs to make decisions, such as the current page of search results or the details of a selected product.
 
