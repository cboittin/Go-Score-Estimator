from util import trace

COLOR_BLACK = 1
COLOR_WHITE = -1

def otherColor(color):
    return color * -1
    
class Group:
    """ Describe the state of a group of connected stones within a Go game """
    def __init__(self, id, pos, color, libs):
        self.pos = pos
        self.moves = [pos]
        self.color = color
        self.liberties = libs
        self.id = id
        
    def isDead(self):
        return len(self.liberties) <= 0
        
    def addStone(self, pos, libs):
        self.moves.append(pos)
        self.removeLiberty(pos)
        self.liberties |= libs
        
    def addLiberty(self, lib):
        self.liberties.add(lib)
    
    def removeLiberty(self, lib):
        try:
            self.liberties.remove(lib)
        except KeyError as e:
            trace(" >>> Warning : tried to remove a liberty that doesn't exist on group %d : %s. Liberties %s <<<" % (self.id, self.moves, self.liberties), 0)
            raise
        
    def merge(self, other):
        """ Merge 2 groups together """
        trace("Merging group %d into group %d" % (other.id, self.id), 3)
        self.moves += other.moves
        self.liberties |= other.liberties
        
class Board:
    def __init__(self):
        self.groups = {}
        self.nextGroupId = 0
        self.board = []
        self.nMoves = 0
        for _ in range(19):
            col = []
            for _ in range(19):
                col.append(None)
            self.board.append(col)
        self.lastStone = None
        
    def __getitem__(self, pos):
        group = self.getGroupAt(pos)
        if group is None:
            return 0
        return group.color
        
    def getGroupAt(self, pos):
        groupID = self.board[pos[0]][pos[1]]
        if groupID is None:
            return None
        return self.groups[groupID]
        
    def addStone(self, pos, color):
        trace(" * Move %s" % str(pos), 4)
        friends, enemies, libs = self.adjacent(pos, color)
        groupID = None
        if len(friends) == 0:
            # Create new group
            groupID = self.createGroup(pos, color, libs)
        else:
            # Add the stone to one nearby group, then bind any other adjacent group to it
            ref = friends.pop()
            for i in range(len(friends)):
                group = friends.pop()
                self.mergeGroups(ref, group)
            ref.addStone(pos, libs)
            groupID = ref.id
        x, y = pos
        self.board[x][y] = groupID
        self.nMoves += 1
        self.checkCaptures(pos, enemies)
        self.lastStone = groupID
        return groupID
        
    def mergeGroups(self, ref, other):
        if ref.id == other.id:
            return
        ref.merge(other)
        for move in other.moves:
            x, y = move
            self.board[x][y] = ref.id
        self.groups.pop(other.id)
        
    def checkCaptures(self, pos, toCheck):
        for group in toCheck:
            group.removeLiberty(pos)
            if group.isDead():
                self.removeGroup(group)
    
    def removeGroup(self, group):
        trace("Killing group %d" % group.id, 3)
        for move in group.moves:
            _, enemies, _ = self.adjacent(move, group.color)
            x, y = move
            self.board[x][y] = None
            for enemy in enemies:
                enemy.addLiberty(move)
        self.groups.pop(group.id)
    
    def createGroup(self, pos, color, libs):
        group = Group(self.nextGroupId, pos, color, libs)
        self.groups[group.id] = group
        trace("Created group %d at %d - %d" % (group.id, pos[0], pos[1]), 3)
        self.nextGroupId += 1
        return group.id
    
    def adjacent(self, pos, color):
        adjacent = []
        x, y = pos
        if x > 0: adjacent.append( (x-1, y) )
        if x < 18: adjacent.append( (x+1, y) )
        if y > 0: adjacent.append( (x, y-1) )
        if y < 18: adjacent.append( (x, y+1) )
        friends = set([])
        enemies = set([])
        libs = set([])
        for adj in adjacent:
            group = self.getGroupAt(adj)
            if group is None:
                libs.add(adj)
            elif group.color == color:
                friends.add(group)
            else:
                enemies.add(group)
        return (friends, enemies, libs)
        
    def toStr(self):
        s = "   "
        for x in range(19):
            c = x if x < 8 else x+1
            s+= chr(c+65) + " "
        s += "\n"
        for y in range(19):
            s+= str(y+1) + " "
            if y < 9:
                s += " "
            for x in range(19):
                if self[x, y] == COLOR_BLACK: s+= "+ "
                elif self[x, y] == COLOR_WHITE: s+= "- "
                else: s+= "  "
            s += "\n"
        return s
    
