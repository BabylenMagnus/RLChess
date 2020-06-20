import torch


def encryption_board(chessboard):
    """
    Кодирует доску в виде строки из модуля chess в
    Тензор из 7ми матриц torch. 7 потому-что для каждой
    фигуры по одной и белые/чёрные.
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

    matrix = torch.tensor(matrix).view(1, 7, 8, 8).float()

    return chessboard, matrix


def give_moves_prob(moves, probability, encrypt_board):
    """
    на вход получает все возможные ходы от list(chessboard.legal_moves)
    и на каждый ход из тензора вероятностей находит шанс этого хода
    """

    boards = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']  # индексы в шахматах
    chess_pieces = ['r', 'n', 'b', 'k', 'q', 'p']  # порядок матриц по
    # фигурам из encryption_board

    moves_prob = {}

    for move in moves:
        move = str(move)

        piece = encrypt_board[boards.index(move[0]) +
                              (int(move[1]) - 1) * 8].lower() # так узнаётся фигура,
        # encrypt_board - это выход из encryption_board, а именно прсто строка всех
        # фигур, берёться столбец из индексов шахматной доски и номер строки

        moves_prob[move] = probability[0, chess_pieces.index(piece),
                                       int(move[3]) - 1, boards.index(move[2])].item()
        # когда элемент который будет ходить известен, в матрице этой фигуре
        # ищетьс вероятность перехода на другое поле

    return moves_prob
