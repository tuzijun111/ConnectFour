import numpy as np
import copy
import random



def game_win(board, player_num):      # same with game_completed in ConnectFour
    player_win_str = '{0}{0}{0}{0}'.format(player_num)
    to_str = lambda a: ''.join(a.astype(str))

    def check_horizontal(b):
        for row in b:
            if player_win_str in to_str(row):
                return True
        return False

    def check_verticle(b):
        return check_horizontal(b.T)

    def check_diagonal(b):
        for op in [None, np.fliplr]:
            op_board = op(b) if op else b
            
            root_diag = np.diagonal(op_board, offset=0).astype(np.int)
            if player_win_str in to_str(root_diag):
                return True

            for i in range(1, b.shape[1]-3):
                for offset in [i, -i]:
                    diag = np.diagonal(op_board, offset=offset)
                    diag = to_str(diag.astype(np.int))
                    if player_win_str in diag:
                        return True

        return False

    return (check_horizontal(board) or
            check_verticle(board) or
            check_diagonal(board))   
    

def actions(board):
    #return list(self.succs.get(board, {}).keys())
    valid_cols = []
    for col in range(board.shape[1]):
        if 0 in board[:,col]:
            valid_cols.append(col)
    return valid_cols

def results(board, move):   
    for a in range(5, -1, -1):   # because game play from board[5,:]
        if board[a][move] == 0:
            return a
            break
            

def player_position(board, row, col, player):
        board[row][col] = player
            
def terminal_test(board):
    return game_win(board, 1) or game_win(board, 2) or len(actions(board)) == 0
            
def minimax(board, alpha, beta, depth, Turn, number):
    if number == 2:
        opponent =1
    else:
        opponent =2

    if depth == 0 or terminal_test(board):
        if terminal_test(board):
            if game_win(board, number):
                return (None, 9999)
            elif game_win(board, opponent):
                return (None, -9999)
            else:          #game over
                return (None, 0)
        else:                                              
            return (None, evaluation_function(board, number))     #depth=0
    
    if Turn:    # Max layer    max_value(board, alpha, beta, depth):
        v = -np.inf
        move = random.choice(actions(board))
        for a in actions(board):
            board_next = board.copy()
            player_position(board_next, results(board, a), a, number)
            score = minimax(board_next, alpha, beta, depth-1, False, number)[1] # get value [0] is move
            if score > v:
                v = score
                move = a        # denote the next move
            alpha = max(alpha, v)
            if alpha>=beta:
                break   # pruning
        return move, v

    else:      #Min layer     def min_value(board, alpha, beta, depth):
        v = np.inf
        move = random.choice(actions(board))
        for a in actions(board):
            board_next = board.copy()
            player_position(board_next, results(board, a), a, opponent)
            score = minimax(board_next, alpha, beta, depth-1, True, number)[1]
            if score < v:
                v = score
                move = a        # denote the next move
            beta = min(beta, v)
            if alpha>=beta:
                break   # pruning
        return move, v

def expectimax(board, depth, Turn, number):
    if number == 2:
        opponent =1
    else:
        opponent =2
    if depth == 0 or terminal_test(board):
        if terminal_test(board):
            if game_win(board, number):
                return (None, 99999999)
            elif game_win(board, opponent):
                return (None, -99999999)
            else: # Game is over, no more valid moves
                return (None, 0)
        else: # Depth is zero
            return (None, evaluation_function(board, number))     
    
    if Turn:    # Max layer    max_value(board, alpha, beta, depth):
        v = -np.inf
        move = random.choice(actions(board))
        for a in actions(board):
            board_next = board.copy()
            player_position(board_next, results(board, a), a, number)
            score = expectimax(board_next, depth-1, False, number)[1] # get value not move [0]   
            
            if score > v:
                v = score
                move = a        # denote the next move
        return move, v

    else:                #Min layer     def exp_value(board, alpha, beta, depth):
        v = np.inf
        move = random.choice(actions(board))
        expect_score = []     # store all the scores of successors
        for a in actions(board):
            board_next = board.copy()
            player_position(board_next, results(board, a), a, opponent)
            score = expectimax(board_next, depth-1, True, number)[1]
            expect_score.append(score)
        avg= np.sum(expect_score)/len(expect_score)   # expectation
        if avg < v:
            v = avg
            move = random.choice(actions(board))        # denote the next move
        return move, v

def evaluation_function(board, number):
    """
    Given the current stat of the board, return the scalar value that 
    represents the evaluation function for the current player
    
    INPUTS:
    board - a numpy array containing the state of the board using the
            following encoding:
            - the board maintains its same two dimensions
                - row 0 is the top of the board and so is
                    the last row filled
            - spaces that are unoccupied are marked as 0
            - spaces that are occupied by player 1 have a 1 in them
            - spaces that are occupied by player 2 have a 2 in them

    RETURNS:
    The utility value for the current board
    """
    # similar to game_winning, we check those arrays which could get us win
    if number == 2:
        opponent =1
    else:
        opponent =2
    score = 0

    center_array =  list(board[:, 3])         # begin from the center
    center_count = center_array.count(number)
    score += center_count*3

    for col in range(7):    #vertical check
        col1 = list(board[:,col])
        for row in range(3):
            array= col1[row: row+4]
            score += evaluate_array(array, number)
    
    for row in range(6):             #horizontal check
        row1 = list(board[row,:])
        for col in range(4):  # 0-4 as it is symmetric
            array = row1[col: col+4]
            score += evaluate_array(array, number)
                                
    for row in range(3):      #positive diagnal check
        for col in range(4):
            array = [board[row+i][col+i] for i in range(4)]
            score += evaluate_array(array, number)
                
    for row in range(3):      #negative diagnal check
        for col in range(4):
            win = [board[row+3-i][col+i] for i in range(4)]
            score += evaluate_array(array, number)
                            
    return score

def evaluate_array(array, number):   
        if number == 2:
            opponent =1
        else:
            opponent =2
        score = 0

        if array.count(number) == 4:
            score += 100
        elif array.count(number) == 3 and array.count(0) == 1:
            score += 5
        elif array.count(number) == 2 and array.count(0) == 2:
            score += 2
        #if array.count(opponent) == 4:
            #score -= 50
        if array.count(opponent) == 3 and array.count(0) == 1:
            score -= 4
        

        return score


class AIPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}:ai'.format(player_number)

    def get_alpha_beta_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the alpha-beta pruning algorithm

        This will play against either itself or a human player

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        move, minimax_score = minimax(board, -np.inf, np.inf, 4, True, self.player_number)
        return move
    
         
    def get_expectimax_move(self, board):
        """
        Given the current state of the board, return the next move based on
        the expectimax algorithm.

        This will play against the random player, who chooses any valid move
        with equal probability

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        move, expectimax_score = expectimax(board, 4, True, self.player_number)
        return move        


class RandomPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'random'
        self.player_string = 'Player {}:random'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state select a random column from the available
        valid moves.

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """
        
        valid_cols = []
        for col in range(board.shape[1]):
            if 0 in board[:,col]:
                valid_cols.append(col)               

        return np.random.choice(valid_cols)
        


class HumanPlayer:
    def __init__(self, player_number):
        self.player_number = player_number
        self.type = 'human'
        self.player_string = 'Player {}:human'.format(player_number)

    def get_move(self, board):
        """
        Given the current board state returns the human input for next move

        INPUTS:
        board - a numpy array containing the state of the board using the
                following encoding:
                - the board maintains its same two dimensions
                    - row 0 is the top of the board and so is
                      the last row filled
                - spaces that are unoccupied are marked as 0
                - spaces that are occupied by player 1 have a 1 in them
                - spaces that are occupied by player 2 have a 2 in them

        RETURNS:
        The 0 based index of the column that represents the next move
        """

        valid_cols = []
        for i, col in enumerate(board.T):
            if 0 in col:
                valid_cols.append(i)

        move = int(input('Enter your move: '))
    

        while move not in valid_cols:
            print('Column full, choose from:{}'.format(valid_cols))
            move = int(input('Enter your move: '))

        return move

