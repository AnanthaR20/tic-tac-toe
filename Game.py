import re,sys
import numpy as np
import Board


class Game:
    """
    
    """

    def __init__(self):
        """
        
        """
        board = Board()
        X_to_move = True
        move_counter = 0
        record = [{"state":board.get_state(),"playerX":X_to_move,"move":(-1,-1)}]


    def move(self,cell:tuple[int,int]) -> None:
        """
        
        """
        self.board.move(cell,X_to_move=self.X_to_move)

        self.X_to_move = not self.X_to_move
        self.move_counter += 1
    



    