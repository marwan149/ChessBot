import chess as ch
import random
import math
import numpy as np


class AlphaZeroNetwork:
    def __init__(self, model=None):
        """
        This would be an actual trained neural network model.
        A model can be loaded or initialized here.
        """
        self.model = model

    def predict(self, board):
        """
        Predict the policy and value for a given board state using the neural network.
        - Policy: Probabilities for all legal moves.
        - Value: Expected outcome of the game from the current position (win/loss).
        """
        # TODO: Implement actual neural network prediction
        # Encode the position using a position encoding mechanism.
        # Feed the encoded position into the network to get policy and value.
        
        
        legal_moves = list(board.legal_moves)
        policy = {move: 1 / len(legal_moves) for move in legal_moves}
        
        # Dummy value: Random number between -1 and 1
        value = random.uniform(-1, 1)
        return policy, value



class MCTSNode:
    def __init__(self, board, move=None, parent=None):
        self.board = board
        self.move = move
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.policy = None

    def is_fully_expanded(self):
        return len(self.children) == len(list(self.board.legal_moves))

    def expand(self, policy):
        legal_moves = list(self.board.legal_moves)
        unexplored_moves = [move for move in legal_moves if move not in [child.move for child in self.children]]
        
        # TODO: Add move ordering here for better exploration
        move = random.choice(unexplored_moves)
        
        new_board = self.board.copy()
        new_board.push(move)
        child_node = MCTSNode(new_board, move, self)
        child_node.policy = policy.get(move, 1 / len(legal_moves))
        self.children.append(child_node)
        return child_node

    def best_child(self, exploration_weight=1.4):
        choices_weights = [
            (child.value / child.visits) + exploration_weight * child.policy * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(-reward)


# MCTS with transposition tables and parallel search support
class MCTS:
    def __init__(self, board, max_iterations, color, network):
        self.board = board
        self.max_iterations = max_iterations
        self.color = color
        self.network = network
        self.transposition_table = {}  # To store already evaluated states

    def simulate(self, node):
        temp_board = node.board.copy()
        while not temp_board.is_game_over():
            policy, _ = self.network.predict(temp_board)
            legal_moves = list(temp_board.legal_moves)
            move = random.choices(legal_moves, weights=[policy[move] for move in legal_moves], k=1)[0]
            temp_board.push(move)
        
        outcome = temp_board.outcome()
        if outcome.winner == self.color:
            return 1
        elif outcome.winner is None:
            return 0
        else:
            return -1

    def search(self):
        root = MCTSNode(self.board)

        for _ in range(self.max_iterations):
            node = root
            # Selection
            while not node.board.is_game_over() and node.is_fully_expanded():
                node = node.best_child()

            # Expansion
            if not node.board.is_game_over():
                policy, _ = self.network.predict(node.board)
                node = node.expand(policy)

            # Simulation (Rollout guided by policy from the neural network)
            _, value = self.network.predict(node.board)
            reward = value

            # Backpropagation
            node.backpropagate(reward)

        # Return the best move based on visits
        return max(root.children, key=lambda child: child.visits).move


class AlphaZeroEngine:
    def __init__(self, board, max_iterations, color, network):
        self.board = board
        self.color = color
        self.max_iterations = max_iterations
        self.network = network

    def get_best_move(self):
        mcts = MCTS(self.board, self.max_iterations, self.color, self.network)
        return mcts.search()


# Playable game class with game saving/loading and analysis mode
class Main:
    def __init__(self, board=ch.Board):
        self.board = board

    # Play human move
    def play_human_move(self):
        try:
            print(self.board.legal_moves)
            print("""To undo your last move, type "undo".""")
            play = input("Your move: ")
            if play == "undo":
                self.board.pop()
                self.board.pop()
                self.play_human_move()
                return
            self.board.push_san(play)
        except:
            self.play_human_move()

    # Play engine move
    def play_engine_move(self, max_iterations, color):
        network = AlphaZeroNetwork()  # Using the neural network for AlphaZero
        engine = AlphaZeroEngine(self.board, max_iterations, color, network)
        self.board.push(engine.get_best_move())

    # Game saving feature
    def save_game(self, filename):
        with open(filename, 'w') as file:
            file.write(self.board.fen())  # Save the game in FEN format

    # Game loading feature
    def load_game(self, filename):
        with open(filename, 'r') as file:
            fen = file.read()
        self.board.set_fen(fen)

    # Check and display the game outcome
    def display_outcome(self):
        if self.board.is_checkmate():
            print("Checkmate!")
            winner = "White" if self.board.turn == ch.BLACK else "Black"
            print(f"{winner} wins!")
        elif self.board.is_stalemate():
            print("Stalemate! The game is a draw.")
        elif self.board.is_insufficient_material():
            print("Draw due to insufficient material!")
        elif self.board.is_seventyfive_moves():
            print("Draw due to the seventy-five-move rule!")
        elif self.board.is_fivefold_repetition():
            print("Draw due to fivefold repetition!")
        elif self.board.is_variant_draw():
            print("The game is a draw (variant rules).")
        else:
            print("The game ended in a draw.")

    # Analysis mode to suggest moves based on engine evaluation
    def analysis_mode(self, max_iterations, color):
        print("Analysis mode enabled.")
        while not self.board.is_game_over():
            print(self.board)
            if input("Get engine's best move suggestion? (y/n): ") == 'y':
                network = AlphaZeroNetwork()  # Using the neural network
                engine = AlphaZeroEngine(self.board, max_iterations, color, network)
                print(f"Engine's best move: {engine.get_best_move()}")
            else:
                self.play_human_move()

    # Start a game
    def start_game(self):
        color = None
        while color not in ["b", "w"]:
            color = input("""Play as (type "b" or "w"): """)
        max_iterations = int(input("""Choose number of MCTS iterations ,recommended ({0< easy <5},{5< medium <10},{10< Hard <15}): """))
        if max_iterations >15 :
            print("Error")
            mt=int(input("""Choose number of MCTS iterations ,recommended ({0< easy <5},{5< medium <10},{10< Hard <15}): """))
            max_iterations=mt
        if color == "b":
            while not self.board.is_game_over():
                print("The engine is thinking...")
                self.play_engine_move(max_iterations, ch.WHITE)
                print(self.board)
                if self.board.is_game_over():
                    break
                self.play_human_move()
                print(self.board)
            self.display_outcome()
        elif color == "w":
            while not self.board.is_game_over():
                print(self.board)
                self.play_human_move()
                print(self.board)
                if self.board.is_game_over():
                    break
                print("The engine is thinking...")
                self.play_engine_move(max_iterations, ch.BLACK)
            self.display_outcome()

        # Reset the board
        self.board.reset()
        # Start another game
        self.start_game()




# Create an instance and start a game
new_board = ch.Board()
game = Main(new_board)
game.start_game()
game.analysis_mode()
game.save_game()
