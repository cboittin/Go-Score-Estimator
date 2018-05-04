# go-sequence
Neural network (using tensorflow) to predict the winner of a go game from the board position.
Currently overfit on random positions of 10k high level games on tygem. Training for 400k games in is progress (using https://github.com/SThornewillvE/Pet-Project---Tygem-Fuseki-Web-Scraper-using-Python)

## How to use

### Training
    
    Create an `sgf` folder with a collection of sgf files.
    Then, run `parseSgf.py`, which will create the `records` dataset from the sgf files. Only files that are valid and give a score will be parsed, the rest will be discarded.
    
    Finally, run `scoreEstimation.py` to train the network using the `records` dataset
    
### Testing
    
    Simply run `scoreEstimation.py $SGF` where `$SGF` is the path to your sgf file
    A positive score indicates black has a lead on the board, a negative one indicates white has a lead on the board. Komi is not taken into account and has to be subtracted from the score.
