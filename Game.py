import re,sys,random
import numpy as np
from Board import Board


class Game:
    """
    
    """
    def __init__(self):
        """
        
        """
        self.board = Board()
        self.X_to_move = True
        self.move_counter = 0
        self.record = []
        self.result = None


    def move(self,cell:tuple[int,int]) -> None:
        """Change the state of the board depending on the current player
            Arguments:
                cell: cell to change
        """
        valid = self.board.is_legal_move(cell)
        if not valid:
            return
        
        self.record.append({
            "move_number":self.move_counter+1,
            "player_to_move": Board.PLAYER_X_NUMBER if self.X_to_move else Board.PLAYER_O_NUMBER,
            "state":self.board.get_state(),
            "move_coordinates":cell,
            "move": 3*cell[0]+ cell[1]
        })
        self.board.move(cell,X_to_move=self.X_to_move)
        self.move_counter += 1
        self.X_to_move = not self.X_to_move

    def is_game_over(self) -> bool:
        """Detects when game is over"""
        if 0 not in self.board.board: # in the event that all squares are taken up
            return True
        X3 = Board.PLAYER_X_NUMBER * 3
        O3 = Board.PLAYER_O_NUMBER * 3
        sums = []
        sums.extend(list(np.sum(self.board.board,axis=0))) # col sums
        sums.extend(list(np.sum(self.board.board,axis=1))) # row sums
        sums.extend([
            np.sum(np.diag(self.board.board)),
            np.sum(np.diag(np.fliplr(self.board.board)))
        ]) # diagonal sums
        return X3 in sums or O3 in sums 
        
    def game_result(self) -> int | None:
        "Returns the game result as an int or None"
        X3 = Board.PLAYER_X_NUMBER * 3
        O3 = Board.PLAYER_O_NUMBER * 3
        sums = []
        sums.extend(list(np.sum(self.board.board,axis=0))) # col sums
        sums.extend(list(np.sum(self.board.board,axis=1))) # row sums
        sums.extend([
            np.sum(np.diag(self.board.board)),
            np.sum(np.diag(np.fliplr(self.board.board)))
        ]) # diagonal sums
        if X3 in sums:
            return Board.PLAYER_X_NUMBER
        elif O3 in sums:
            return Board.PLAYER_O_NUMBER
        elif self.is_game_over():
            return 0
        else:
            return None
     
    def report_game_result(self) -> None:
        """Updates game result in record of game"""
        self.result = self.game_result()
        for rec in self.record:
            rec["game_result"] = self.result
            rec["total_moves"] = self.move_counter        

    def play(self) -> None:
        """Runs a CLI interface for playing Tic Tac Toe. Can quit by 
        pressing 'q' at any time."""

        while True:
            if self.is_game_over():
                r = self.game_result()
                msg = ""
                if r == Board.PLAYER_X_NUMBER:
                    msg = "Player X Wins!"
                elif r == Board.PLAYER_O_NUMBER:
                    msg = "Player O Wins!"
                elif r == 0:
                    msg = "A Tie!"
                else:
                    msg = "Game not Over!"
                print(msg)                
                self.board.print()
                self.report_game_result()
                break
            player = "Player X" if self.X_to_move else "Player O"
            print(f"{player} to move")
            self.board.print()
            move = input("Input your move as {row #}{column #} with no space\n")
            if move == 'q':
                break
            if len(move) != 2 or int(move[0]) > 3 or int(move[1]) > 3:
                print("Invalid input!\n\n\n")
                continue
            row, col = int(move[0])-1, int(move[1])-1
            valid = self.board.is_legal_move((row,col))
            if not valid:
                print("Invalid Move!\n\n\n")
                continue
            self.move((row,col))

    def generate_game() -> list[dict]:
        """Generates a complete game and returns the move records"""
        g = Game()
        while not g.is_game_over():
            possible_moves = []
            for i,v in enumerate(g.board.board):
                for j,z in enumerate(g.board.board[i]):
                    if z == 0:
                        possible_moves.append((i,j))

            mv = random.sample(possible_moves,1)[0]
            g.move(mv)
        
        g.report_game_result()

        return g.record




