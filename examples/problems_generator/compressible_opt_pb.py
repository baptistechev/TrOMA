import numpy as np

def estimate_cost(dit_string, rules_rewards):
    """
    Estimate the cost of a sequence based on the rules, rewards, and starting positions.
    """
    if rules_rewards == {}:
        return 0

    reward = 0
    for rule, rule_reward in rules_rewards.items():
        rule_length = len(rule)
        for i in range(len(dit_string) - rule_length + 1):
            if i + rule_length <= len(dit_string) and np.all(dit_string[i:i + rule_length] == rule):
                reward += rule_reward
                
    return reward

def generate_rules_and_rewards(sequence_length, num_rules, rule_length=4, reward_range=(1, 20), dit_dimension=2):
    """
    Generate rules, rewards, and starting positions for the rules.
    """

    rule_rewards = {}

    while len(rule_rewards) < num_rules:
        rule = tuple([ int(i) for i in np.random.randint(0, dit_dimension, size=rule_length) ])  # Generate a random rule
        if rule not in rule_rewards:
            reward = np.random.randint(reward_range[0], reward_range[1])  # Generate a random reward
            rule_rewards[rule] = reward
           
    return rule_rewards

def compute_full_spectrum(dit_string_length, rules_rewards, dit_dimension=2):
    """
    Compute the full spectrum of costs for all possible sequences of a given length.
    """
    num_sequences = dit_dimension ** dit_string_length
    spectrum = np.zeros(num_sequences)

    for i in range(num_sequences):
        dit_string = np.array([int(x) for x in np.base_repr(i, base=dit_dimension).zfill(dit_string_length)])
        spectrum[i] = estimate_cost(dit_string, rules_rewards)

    return spectrum