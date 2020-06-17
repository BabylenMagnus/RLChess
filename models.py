from torch import nn
# Q это все V цепочек Нод делённая на количество проходов через ноду
# Q = \frac{\sum_i^N{V_i}}{N}


class Node:
    """
    Это Ноды дерева по которым будет ходить AI.
    Они максимально похожи на Ноды из Alpha Go Zero.
    """

    def __init__(self, parent, probability):
        self.parent = parent  # это родители Ноды
        self.children = {}

        # с большой буквы т.к. эти значения будут указываться в формулах

        self.N = 0  # количество проходов через ноду
        self.Quality = 0  # вышу формула
        self.Value = 0  # цена ноды (выход из NN)
        self.Probability = probability  # вероятность хода на эту ноду (выход из NN)


class SkyNet(nn.Module):
    def __init__(self):
        super(SkyNet, self).__init__()
