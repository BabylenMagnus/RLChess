import operator

import chess
from torch import nn
import numpy as np
from back import encryption_board, give_moves_prob


# Q это все V цепочек Нод делённая на количество проходов через ноду
# Q = \frac{\sum_i^N{V_i}}{N}


class Node:
    """
    Это Ноды дерева по которым будет ходить AI.
    Они максимально похожи на Ноды из Alpha Go Zero.
    """

    def __init__(self, parent=None, probability=None):
        self.parent = parent  # это родители Ноды
        self.children = {}

        # с большой буквы т.к. эти значения будут указываться в формулах

        self.N = 1  # количество проходов через ноду
        self.Quality = 0  # вышу формула
        self.Value = 0  # цена ноды (выход из NN)
        self.Probability = probability  # вероятность хода на эту ноду (выход из NN)

    def leaf_expansion(self, moves, quartile=0.8):
        """
        из допустимых ходов проверяються только 20% лучших
        """
        minimum = np.quantile(np.array(list(moves.values())), quartile)
        moves = dict([(k, v) for k, v in moves.items() if v > minimum])

        for move in moves:

            if move not in self.children:
                self.children[move] = Node(parent=self, probability=moves[move])

        self.N += 1

    def upper_confidence_bounds(self, c=0.1):
        """
        Ход выбираеться не только из-за выхода из нейросети,
        но и по количеству проходов через ноду. Чтобы большее количество
        ходов использовать (которые имеют смысл).
        """
        result = {}

        for move, node in self.children.items():
            element = np.sqrt(np.log(self.N) / node.N)
            result[move] = node.Probability + c * element

        return result

    def show_children(self):
        for child in self.children:
            print('{:<3} - {:.3f}'.format(child, self.children[child].Probability))


class MCTS:

    def __init__(self, agent, board):
        self.root = Node(probability=1)
        self.agent = agent
        self.board = board

    def start_self_play(self):

        current = self.root

        while True:

            moves = list(self.board.legal_moves)
            encrypt_board, inp = encryption_board(self.board)
            probability, value = self.agent(inp)
            current.Value = value

            moves_prob = give_moves_prob(moves, probability, encrypt_board)

            current.leaf_expansion(moves_prob)
            move = max(current.upper_confidence_bounds().items(),
                       key=operator.itemgetter(1))[0]
            current = current.children[move]
            move = chess.Move.from_uci(str(move))
            self.board.push(move)

            print(self.board)

            if self.board.is_game_over():
                break


class SkyNetModel(nn.Module):
    """
    Это агент, который на взод получает доску, а отправляет
    тензор вероятностей хода и Value для позиции, где -1 - победа чёрных
    1 - победа белых и 0 ничья, это очень пригодиться для Нод.
    Сама архитекрута не сильно отличаеться от AlphaGo.
    На одну доску нужно всего 4 мс.
    """

    def __init__(self):
        super(SkyNetModel, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Conv2d(32, 7, kernel_size=3, padding=1),
            nn.BatchNorm2d(7),
            nn.LeakyReLU(0.2)
        )

        self.probability = nn.Sequential(
            nn.Conv2d(7, 7, kernel_size=1),
            nn.LeakyReLU(0.4),

            nn.Conv2d(7, 7, kernel_size=1),
            nn.Softmax2d()
        )

        self.value = nn.Sequential(
            nn.Linear(448, 64),
            nn.LeakyReLU(0.4),

            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, board):
        out = self.main(board)

        probability = self.probability(out)
        value = self.value(out.view(-1, 448))
        return probability, value
