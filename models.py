from torch import nn


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

        self.N = 0  # количество проходов через ноду
        self.Quality = 0  # вышу формула
        self.Value = 0  # цена ноды (выход из NN)
        self.Probability = probability  # вероятность хода на эту ноду (выход из NN)


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

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
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

