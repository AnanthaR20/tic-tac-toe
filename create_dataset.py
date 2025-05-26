from Game import Game
import numpy as np
import argparse,time

# Writes a dataset of state/move pairs and write to txt
# Here is where the dataset curation takes place. I think this
# will determine a lot of the model's performance
def main(size:int,file:str):
    dataset_size = size
    examples = []
    while len(examples) < dataset_size:
        moves = Game.generate_game()
        if moves[0]["game_result"] != 0 and moves[0]["total_moves"] == 9:
            continue
        for i,move in enumerate(moves):
            if move["game_result"] == move["player_to_move"]:
                move['state'].append(np.int64(move['player_to_move']))
                examples.append({
                    "input_state": move['state'],
                    "target_move": move["move"]
                })
            if move["game_result"] == 0:
                move['state'].append(np.int64(move['player_to_move']))
                examples.append({
                    "input_state": move['state'],
                    "target_move": move["move"]
                })
    
    with open(file,'w') as f:
        for i,mv in enumerate(examples):
            f.write(f"{mv["target_move"]}: {",".join([str(i) for i in mv["input_state"]])}\n")



if __name__ == "__main__":

    # argparse logic
    parser = argparse.ArgumentParser()
    parser.add_argument("--size",default=100,type=int)
    parser.add_argument("--op_file",default="data/move_dataset.txt",type=str)
    args = parser.parse_args()

    start = time.time()
    main(args.size,args.op_file)
    print(f"Generated {args.size} examples in {time.time() - start} seconds")
