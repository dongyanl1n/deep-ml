import numpy as np
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def self_play_train(payoff_matrix: list, initial_logits: list, num_iterations: int, learning_rate: float, sync_interval: int) -> dict:
    """
    Run a self-play training loop for a two-player zero-sum game.

    Args:
        payoff_matrix: 2D list of shape (n, n) - payoff matrix for player 1
        initial_logits: 1D list of shape (n,) - initial policy logits for both players
        num_iterations: int - number of training iterations
        learning_rate: float - learning rate for policy gradient ascent
        sync_interval: int - how often to sync opponent to current policy

    Returns:
        Dictionary with 'policy_logits', 'opponent_logits', 'policy_probs', 'expected_payoff'
    """
    payoff_matrix = np.array(payoff_matrix, dtype=float)
    initial_logits = np.array(initial_logits, dtype=float)
    # at the beginning
    # theta = theta_opp = initial_logits
    # aliasing issue! ^
    theta = initial_logits
    theta_opp = theta.copy()
    for t in range(num_iterations):
        # sync check
        if t > 0 and t % sync_interval == 0:
            theta_opp = theta.copy()
        softmax_player, softmax_opp = softmax(theta), softmax(theta_opp) # policy prob
        expected_payoff_player = payoff_matrix @ softmax_opp # num_actions for player 1
        expected_value_player = np.dot(expected_payoff_player, softmax_player)  # float
        policy_grad = softmax_player * (expected_payoff_player-expected_value_player) # num_actions
        theta += learning_rate * policy_grad
    softmax_player = softmax(theta)
    softmax_opp = softmax(theta_opp)
    expected_payoff_player = payoff_matrix @ softmax_opp
    expected_value_player = np.dot(expected_payoff_player, softmax_player)
    
    return{
        'policy_logits': [round(i, 4) for i in theta],
        'opponent_logits': [round(i, 4) for i in theta_opp],
        'policy_probs': [round(i, 4) for i in softmax_player],
        'expected_payoff': round(expected_value_player, 4)
    }
