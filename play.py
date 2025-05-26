from Game import Game
import argparse



def main(game_type:str):
    if game_type == "ttt":
        g = Game()
        g.play()
    else:
        print("Not built yet")



if __name__ == "__main__":

    # argparse logic
    parser = argparse.ArgumentParser()
    parser.add_argument("--game",default="ttt",type=str)
    args = parser.parse_args()

    main(args.game)


