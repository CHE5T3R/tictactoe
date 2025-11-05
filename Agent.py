import numpy as np
import random
import pickle


class Agent:

    def __init__(self, game, player='X', episode=100000, epsilon=0.9, discount_factor=0.6, eps_reduce_factor=0.01):
        """
        game : it is the TicTacToe game which you train the agent on
        player : Agent is X(1) or O(-1). If you train the agent, player variable expresses that you train the agent which player for
        brain : holds the q values of different states in the game
        episode : how many game will be played at the end of the training
        epsilon : the value that specify how often the agent do random move or move from q table
        we decrease this value by time because our agent will be reached enough level, and we will not need at all
        when we train it over about 10.000 times. So that the agent will only use its q-table.
        discount_factor : backpropagation coefficient
        """
        self.game = game
        self.player = player
        self.brain = dict()
        self.episode = episode
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.results = {'X': 0, 'O': 0, 'D': 0}
        self.eps_reduce_factor = eps_reduce_factor

    def save_brain(self, player):
        with open('brain' + player, 'wb') as brain_file:
            pickle.dump(self.brain, brain_file)

    def load_brain(self, player):
        try:
            with open('brain' + player, 'rb') as brain_file:
                self.brain = pickle.load(brain_file)
        except:
            print('No brain yet. You should train the agent first. So this game agent will play randomly')

    def reward(self, player, move_history, result):
        _reward = 0
        if player == 1:
            if result == 1:
                _reward = 1
                self.results['X'] += 1
            elif result == -1:
                _reward = -1
                self.results['O'] += 1
        elif player == -1:
            if result == 1:
                _reward = -1
                self.results['X'] += 1
            elif result == -1:
                _reward = 1
                self.results['O'] += 1
        if result == -2:
            self.results['D'] += 1
        move_history.reverse()
        # If you don't want the Agent to play randomly for first move in the game change move_history[:-1] --> move_history
        for state, action in move_history:
            self.brain[state, action] = self.brain.get((state, action), 0.0) + _reward
            _reward *= self.discount_factor

    def use_brain(self):
        possible_actions = self.game.get_available_positions()
        # arbitrary negative low number
        max_qvalue = -1000
        best_action = possible_actions[0]
        for action in possible_actions:
            qvalue = self.brain.get((self.game.get_current_game_tuple(), action), 0.0)
            if qvalue > max_qvalue:
                best_action = action
                max_qvalue = qvalue
            # if they have same value, current action stays or they change with 50% chance
            elif qvalue == max_qvalue and random.random() < 0.5:
                best_action = action
                max_qvalue = qvalue
            elif len(possible_actions) == 9:
                best_action = random.choice(possible_actions)
                break

        return best_action

    def train_brain_x_byrandom(self):
        for _ in range(self.episode):
            if _ % 1000 == 0:
                print('Episode: ' + str(_))
                self.epsilon -= self.eps_reduce_factor
            move_history = []
            # One game is played in each while loop
            while True:
                # don't want agent to play the same start, the agent will choose random action at the first action
                if sum(self.game.get_current_game() == 1) == 0 or random.random() < self.epsilon:

                    available_actions = self.game.get_available_positions()
                    action_x = random.choice(available_actions)

                    move_history.append([self.game.get_current_game_tuple(), action_x])

                    # self.game always let the X play firstly and it changes the player then in backend
                    self.game.make_move(action_x)

                else:
                    action_x = self.use_brain()

                    move_history.append([self.game.get_current_game_tuple(), action_x])

                    self.game.make_move(action_x)

                # checking if game is over after X played
                if self.game.is_winner():
                    self.reward(1, move_history, self.game.winner)
                    break

                # O always plays randomly
                available_actions = self.game.get_available_positions()
                action_o = random.choice(available_actions)
                self.game.make_move(action_o)

                # checking if game is over after O played
                if self.game.is_winner():
                    self.reward(1, move_history, self.game.winner)
                    break

        self.save_brain('X')
        print('TRAINING IS DONE!')
        print('RESULTS:')
        print(self.results)

    def train_brain_o_byrandom(self):
        for _ in range(self.episode):
            if _ % 1000 == 0:
                print('Episode: ' + str(_))
                self.epsilon -= self.eps_reduce_factor
            move_history = []
            # One game is played in each while loop
            while True:

                available_actions = self.game.get_available_positions()
                action_x = random.choice(available_actions)
                # self.game always let the X play firstly and it changes the player then in backend
                self.game.make_move(action_x)

                # checking if game is over after X played
                if self.game.is_winner():
                    self.reward(-1, move_history, self.game.winner)
                    break

                # don't want agent to play the same start, the agent will choose random action at the first action
                if random.random() < self.epsilon:

                    available_actions = self.game.get_available_positions()
                    action_o = random.choice(available_actions)

                    move_history.append([self.game.get_current_game_tuple(), action_o])

                    self.game.make_move(action_o)

                else:
                    action_o = self.use_brain()

                    move_history.append([self.game.get_current_game_tuple(), action_o])

                    self.game.make_move(action_o)

                if self.game.is_winner():
                    self.reward(-1, move_history, self.game.winner)
                    break

        self.save_brain('O')
        print('TRAINING IS DONE!')
        print('RESULTS:')
        print(self.results)

    # Agent.py
    def play_with_human(self):
        # Kayıtlı beyni yükle
        self.load_brain(self.player)

        # Başlangıç: her koşulda boş tahta çiz
        print("\nYeni oyun başlıyor. X başlar.")
        self.game.draw_current_game()

        while True:
            # --- Agent 'O' ise: İnsan (X) önce oynar ---
            if self.player == 'O':
                # İNSAN (X) HAMLESİ
                try:
                    move = int(input("Sen X'sin. 1-9 arasında hamle gir: ")) - 1
                except Exception:
                    print("Geçersiz giriş.")
                    continue

                if move not in self.game.get_available_positions():
                    print("Bu kare dolu veya geçersiz. Tekrar dene.")
                    continue

                self.game.make_move(move)
                self.game.draw_current_game()  # insan hamlesinden sonra çiz

                result = self.game.is_winner(isgame=True)
                if result is not False:
                    # Oyun resetlendi; yeni oyuna boş tahta ile başla
                    self.game.draw_current_game()
                    break

                # AGENT (O) HAMLESİ
                state_tuple = self.game.get_current_game_tuple()
                # Geçerli hamlelerden beyinle seç
                chosen = self.use_brain() if self.brain else np.random.choice(self.game.get_available_positions())
                self.game.make_move(chosen)
                print("Agent (O) oynadı.")
                self.game.draw_current_game()

                result = self.game.is_winner(isgame=True)
                if result is not False:
                    # Oyun resetlendi; yeni oyuna boş tahta ile başla
                    self.game.draw_current_game()
                    break

            # --- Agent 'X' ise: Agent (X) önce oynar ---
            else:  # self.player == 'X'
                # AGENT (X) HAMLESİ
                state_tuple = self.game.get_current_game_tuple()
                chosen = self.use_brain() if self.brain else np.random.choice(self.game.get_available_positions())
                self.game.make_move(chosen)
                print("Agent (X) oynadı.")
                self.game.draw_current_game()

                result = self.game.is_winner(isgame=True)
                if result is not False:
                    self.game.draw_current_game()
                    break

                # İNSAN (O) HAMLESİ
                try:
                    move = int(input("Sen O'sun. 1-9 arasında hamle gir: ")) - 1
                except Exception:
                    print("Geçersiz giriş.")
                    continue

                if move not in self.game.get_available_positions():
                    print("Bu kare dolu veya geçersiz. Tekrar dene.")
                    continue

                self.game.make_move(move)
                self.game.draw_current_game()

                result = self.game.is_winner(isgame=True)
                if result is not False:
                    self.game.draw_current_game()
                    break





