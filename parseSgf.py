import os
import sys
import io
import re
from go import Board, COLOR_BLACK, COLOR_WHITE, otherColor
from util import letterToCol, parseCoordinates

def playGame(moves):
    board = Board()
    try:
        color = COLOR_BLACK
        for move in moves:
            board.addStone(move, color)
            color = otherColor(color)
            return False
    except:
        print("Invalid record")
        return True

def parseSgfFile(sgfPath):
    # Parse the sgf into a sequence of moves
    with io.open(sgfPath, "r", encoding="utf-8") as sgf:
        for line in sgf:
            if "SZ[19]" not in line or "HA" in line: return None
        
            if "Resign" in line or "Time" in line or "Offline" in line: return None
        
            score = re.search("RE\[([BW])\+([0-9.]+)\]", line)
            if score is None:
                print("Found no valid result in %s - %s" % (sgfPath, line))  # We already removed the games that end in resign or time
                return None
            winner = COLOR_BLACK if score.group(1) == "B" else COLOR_WHITE
            score = float(score.group(2))
            
            # print("winner : %d - score %f" % (winner, score))
            komi = re.search("KM\[([0-9\.]+)\]", line)
            if komi is None:
                print("Found no komi in %s - %s" % (sgfPath, line))
                return None
            komi = float(komi.group(1))
            score = score + komi if winner == COLOR_BLACK else score - komi
            
            nodes = line.split(";")
            abort = False
            lastColor = None
            moves = []
            movesStr = []
            for node in nodes:
                match = re.search("(B|W)\[([a-z]{2})\]", node)
                passMatch = re.search("(B|W)\[\]", node)
                if match is not None:
                    if match.group(1) == "B":
                        color = COLOR_BLACK
                    else:
                        color = COLOR_WHITE
                    if color == lastColor:
                        print("ERROR in %s, one player moves twice !" % sgfPath)
                        print(node, line)
                        abort = True
                        break
                    pos = parseCoordinates(match.group(2))
                    moves.append(pos)
                    movesStr.append( "%d,%d" % pos)
                    lastColor = color
                elif passMatch is not None:
                    movesStr.append("-1")
                    lastColor = otherColor(color)
            if abort: return None
            
            if playGame(moves): return None
            
            record = "%d;%.1f;%d;%s" % (winner, score, len(movesStr), ":".join(movesStr))
            return record

if __name__ == "__main__":
    nGames = 0
    sgfDir = "./sgf"
    dataFile = "records2"
    records = open(dataFile, "w", buffering=5000)
    recordsList = "recordsList.log"

    log = open(recordsList, "w")

    print("Listing sgf files")
    sgfFiles = os.listdir(sgfDir)

    print("Parsing start")
    for fileName in sgfFiles:
        filePath = os.path.join(sgfDir, fileName)
        try:
            file, extension = fileName.split(".")
            if extension not in ("sgf", "SGF"):
                continue
        except:
            continue
        
        
        record = parseSgfFile(filePath)
        if record is None:
            continue
        records.write(record + "\n")
        log.write("%s : %d\n" % (fileName, nGames))
        
        nGames += 1
        if(nGames % 1000 == 0):
            print("Parsed %d games." % nGames)
    
    log.close()
    records.close()
    print("\nDone parsing %d games." % nGames)  # 385500

# Output data format:
#   winner;score;nMoves;list of moves