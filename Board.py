import re,sys
import numpy as np

class Board:
    """
    
    """

    SPACING = "  "
    PLAYER_X_NUMBER = 1
    PLAYER_O_NUMBER = 2

    def __init__(self):
        """
        
        """
        self.board = np.full((3,3),0,dtype=int)

    def __repr__(self):
        str_mat = [["" for _ in range(self.board.shape[0])] for _ in range(self.board.shape[1])]        
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                str_mat[i][j] = str(self.board[i][j])
        
        return "\n".join([Board.SPACING.join(cell) for cell in str_mat])

    def move(self,cell:tuple[int,int],X_to_move:bool = True) -> None:
        """Put a move on the board
            Arguments:
                cell: coordinates of where to place the move
                playerX: indicates whether to place an 'X' or 'O'
            
            Returns:
                None
        """
        row = cell[0]
        col = cell[1]
        if self.board[row][col] != 0:
            print("Invalid Move")
            return
        
        self.board[row][col] = Board.PLAYER_X_NUMBER if X_to_move else Board.PLAYER_O_NUMBER


    def get_state(self):
        """
        
        """
        state = []
        for i in self.board:
            state.extend(i)
        
        return state


    def print(self):
        """
        
        """
        str_mat = [["" for _ in range(self.board.shape[0])] for _ in range(self.board.shape[1])]        
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                symb = ""
                if self.board[i][j] == Board.PLAYER_X_NUMBER:
                    symb = "X"
                elif self.board[i][j] == Board.PLAYER_O_NUMBER:
                    symb = "O"
                else:
                    symb = "-"
                str_mat[i][j] = symb
        
        print("\n".join([Board.SPACING.join(cell) for cell in str_mat]))

        
