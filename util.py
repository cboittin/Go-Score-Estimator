VERBOSE_LEVEL=0
def trace(msg, level=1):
    if level <= VERBOSE_LEVEL:
        print(msg)

def coordsToStr(coords):
    return chr(coords[0]+97) + chr(coords[1]+97)

def letterToCol(letter):
    return ord(letter) - 97

def parseCoordinates(coordsStr):
    col = int(letterToCol(coordsStr[0]))
    row = int(letterToCol(coordsStr[1]))
    return (col, row)

def colRow2Goban(pos):
    col, row = pos
    if col > 7: col += 1
    c = chr(col + 65)
    r = str(19 - row)
    return c + r
    
def lineCount(filename):
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)
    
    f.close()
    return lines