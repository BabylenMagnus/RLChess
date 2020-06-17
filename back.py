import torch


def encryption_board(chessboard):
    """
    Кодирует доску в виде строки из модуля chess в
    Тензор из 7ми матриц torch. 7 потому-что для каждой
    фигуры по одной и белые/чёрные.
    В модели не будет памяти, т.к. выбор хода будет всегда из
    допустимых ходов и по этой причине не будет за кого играет модель.
    """
    chessboard = str(chessboard)
    chessboard = chessboard.replace('\n', '').replace(' ', '')

    chess_pieces = ['r', 'n', 'b', 'k', 'q', 'p']

    matrix = []
    i = 0

    for piece in chess_pieces:
        matrix.append([])
        for name in chessboard:
            matrix[i].append(int(name.lower() == piece))
        i += 1

    matrix.append([])

    for name in chessboard:
        matrix[-1].append(int(name.islower()))

    matrix = torch.tensor(matrix).view(1, 7, 8, 8)

    return matrix
