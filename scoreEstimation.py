from go import Board, otherColor, COLOR_BLACK, COLOR_WHITE
import numpy as np
import random
import tensorflow as tf
import time
from util import colRow2Goban, trace, lineCount
import os
import sys

COLORS = {COLOR_BLACK: 0, COLOR_WHITE: 1}

MINIMUM_MOVES_RANGE = range(10, 15)
MOVES_INCREMENT = 10
BUFFER_SIZE = 100000

BATCH_SIZE = 128
N_FILTERS = 16
N_LAYERS = 6

x = tf.placeholder("float", [None, 2, 19, 19])
y = tf.placeholder("float", [None, 1])

initializer = tf.keras.initializers.he_normal()
training = tf.placeholder(tf.bool)

TF_LOG = os.path.join("model", "tf.log")
NETWORK_CHECKPOINT = os.path.join("model", "model.ckpt")

DEBUG_LOG = "debug.log"

def weight_variable(shape):
    return tf.Variable(initializer(shape))

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME", data_format="NCHW")
    
def maxpool2d(x):   # Currently not used, we only have batch_norm
    return tf.layers.max_pooling2d(x, pool_size=(2,2), strides=(2,2), padding="SAME", data_format="NCHW")

def convolutionalLayer(inputs, input_channels, n_filters, filter_size=3):
    W_conv = weight_variable([filter_size, filter_size, input_channels, n_filters])
    
    conv = conv2d(inputs, W_conv)
    conv_bn = tf.layers.batch_normalization(conv, fused=True, training=training)
    output = tf.nn.relu(conv_bn)
    
    return output
    
def outputLayer(inputs):
    W_conv = weight_variable([3, 3, N_FILTERS, 2])
    
    conv = conv2d(inputs, W_conv)
    conv_bn = tf.layers.batch_normalization(conv, fused=True, training=training)
    conv_out = tf.nn.relu(conv_bn)

    flat = tf.reshape(conv_out, shape=[-1, 2*19*19])
    W_fc = weight_variable([2*19*19, 1])
    b_fc = bias_variable([1])
    fc = tf.matmul(flat, W_fc) + b_fc     # Linear activation because it's a regression
    
    output = tf.reshape(fc, shape=[-1, 1])
    
    return output
    
def neuralNetworkModel(x):
    x = tf.reshape(x, shape=[-1, 2, 19, 19])    # 2 layers of a 19 by 19 board, black stones and white stones
    xNorm = tf.layers.batch_normalization(x, fused=True, training=training)
    
    layer = convolutionalLayer(xNorm, 2, N_FILTERS)   # input layer
    print("Input layer parameters :", calcTotalParameters())
    
    for _ in range(N_LAYERS):  # hidden layers
        layer = convolutionalLayer(layer, N_FILTERS, N_FILTERS)
    
    output = outputLayer(layer)
    print("Total parameters in the model :", calcTotalParameters())
    
    return output

def calcTotalParameters():
    return np.sum([np.prod(v.shape) for v in tf.trainable_variables()])

def trainNetwork(batchMaker):
    print("Start Training")
    prediction = neuralNetworkModel(x)
    # score = prediction[0]
    # dev = prediction[1]

    # distance = tf.abs(tf.subtract(y, score))
    # expectedDev = distance
    # # winner = tf.sign(score)
    
    # lBound = tf.subtract(score, dev)
    # uBound = tf.add(score, dev)
    # bound = tf.minimum( tf.abs(tf.subtract(y, lBound)) , tf.abs(tf.subtract(y, uBound)) )  # Calculate the relevant bound
    
    # cost1 = tf.minimum(distance, dev)  # Linear cost within the confidence interval
    # distanceToBound = tf.maximum( tf.subtract(tf.abs(tf.subtract(y, bound)), dev), 0)
    # cost2 = tf.square(distanceToBound)  # Quadratic cost outside the confidence interval -> TODO add 1 to the distance / sub 1 to the deviation ?
    # cost3 = tf.square(tf.subtract(expectedDev, dev))  # Linear cost of the excessive deviation (the smaller the better)
    
    # cost = tf.reduce_mean(tf.add(cost1, tf.add(cost2, cost3)))
    
    cost = tf.reduce_mean(tf.square(tf.subtract(y, prediction)))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    
    saver = tf.train.Saver()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # Get the epoch from the tf_log, if any
    epoch = 1
    try:
        log = open(TF_LOG, "r").read().split("\n")
        epoch = int(log[-2])+1
        print("Starting at epoch", epoch)
    except:
        print("Starting training")
        
    debugLog = open(DEBUG_LOG, "w")
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        
        if epoch != 1:
            saver.restore(sess, NETWORK_CHECKPOINT)
        
        for ep in range(epoch, 11):
            start = time.time()
            epochStart = start
            epoch_loss = 0
            batchX, batchY = batchMaker.nextBatch(BATCH_SIZE)
            batchIndex = 1
            while batchY.shape[0] > 0:
                pred, _, c = sess.run([prediction, optimizer, cost], feed_dict={x: batchX, y: batchY, training:True})
                epoch_loss += c
                if batchIndex % 1000 == 0:
                    end = time.time()
                    print("Epoch %d - Batch %d - %.2fs" % (epoch, batchIndex, end - start))
                    start = end
                batchIndex += 1
                if random.random() < 0.001:
                    for i in range(batchY.shape[0]):
                        expected = batchY[i]
                        result = str(pred[i])
                        debugLog.write("%d : %s\n" % (expected, result))
                batchX, batchY = batchMaker.nextBatch(BATCH_SIZE)
                
            saver.save(sess, NETWORK_CHECKPOINT)
            with open(TF_LOG, "a") as f:
                f.write(str(epoch)+"\n")
            epochEnd = time.time()
            print("Epoch", epoch, "completed - %d batches in in %.2fs with loss: %.3f" % (batchIndex - 1, epochEnd - epochStart, epoch_loss))
            epoch += 1
            batchMaker.restart()
            
        print("Done training the model.")
        
def useNetwork(sgfPaths):
    from parseSgf import parseSgfFile
    
    positionsToEvaluate = []
    
    for sgfPath in sgfPaths:
        parsed = parseSgfFile(sgfPath)
        record = Record(parsed)
        positions = record.computePositions(range(50, 51), 100)
        positionsToEvaluate.append(positions[-1])
    
    print(runNetwork(positionsToEvaluate))
    
def runNetwork(inputs):
    prediction = neuralNetworkModel(x)
    saver = tf.train.Saver()
    
    result = None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, NETWORK_CHECKPOINT)
        result = sess.run([prediction], feed_dict={x:inputs, training:True})
    return result[0]

class Record:
    def __init__(self, line):
        winner, score, nMoves, moves = line.split(";")
        self.moves = [s.split(",") for s in moves.split(":")]
        self.score = int(winner) * float(score)
        self.nMoves = int(nMoves)
        
    def boardToArray(self, board):
        array = np.zeros([2, 19, 19])
        for group in board.groups.values():
            colorIndex = COLORS[group.color]
            for stone in group.moves:
                x, y = stone
                array[colorIndex, x, y] = 1.
        return array
    
    def computePositions(self, minimumMovesRange, movesBetweenTwoPositions):
        try:
            board = Board()
            color = COLOR_BLACK
            positions = []
            initMoves = random.choice(minimumMovesRange)
            for i in range(initMoves):
                move = self.moves[i]
                if move != "-1":
                    x, y = (int(m) for m in move)
                    board.addStone((x,y), color)
                color = otherColor(color)
            positions.append(self.boardToArray(board))
            moveNumber = i
            while moveNumber < self.nMoves:
                for _ in range(movesBetweenTwoPositions):
                    moveNumber += 1
                    if moveNumber >= self.nMoves: break
                    move = self.moves[moveNumber]
                    if move != "-1":
                        x, y = (int(m) for m in move)
                        board.addStone((x,y), color)
                    color = otherColor(color)
                positions.append(self.boardToArray(board))
            return positions
        except:  # Debugging
            trace("Error in processing record. %d moves ends by %f - %s" % (self.nMoves, self.score, [colRow2Goban([int(x) for x in m]) for m in self.moves]), 1)
            try:
                trace("At move number %d" % moveNumber or i, 1)
            except: pass
            raise
    
class BatchMaker:
    def __init__(self, dataFile):
        self.nRecords = lineCount(dataFile)
        self.data = open(dataFile, "r", buffering=5000)
        # self.table = {}
        self.boardPositions = None
        self.scores = None
        self.extra = []
        self.currentIndex = 0
        self.recordsDone = 0
        self.refillBuffers()
    
    def _resetBuffers(self):
        self.currentIndex = 0
        self.boardPositions = np.zeros([BUFFER_SIZE, 2, 19, 19])
        self.scores = np.zeros([BUFFER_SIZE, 1])
        for position, score in self.extra:
            self.boardPositions[self.currentIndex] = position
            self.scores[self.currentIndex] = score
            self.currentIndex += 1
        self.extra.clear()
    
    def _fillBuffers(self):
        filled = False
        end = False
        while not filled and not end:
            remaining = BUFFER_SIZE - self.currentIndex
            nRecords = 1 + remaining // 60  # Estimate of the number of positions from a single game
            end = self.parseRecords(nRecords)
            filled = len(self.extra) > 0
        if end:
            self.boardPositions = self.boardPositions[:self.currentIndex]
            self.scores = self.scores[:self.currentIndex]
        self.currentIndex = 0
        trace("Loaded %d of %d records" % (self.recordsDone, self.nRecords), 0)
        
    def refillBuffers(self):
        trace("Refilling data buffers", 1)
        self._resetBuffers()
        self._fillBuffers()
        self.shuffleRecords()
        
    def shuffleRecords(self):
        indices = np.arange(self.scores.shape[0])
        np.random.shuffle(indices)
        self.boardPositions = self.boardPositions[indices]
        self.scores = self.scores[indices]
        
    def parseRecords(self, nRecords):
        for i in range(nRecords):
            line = self.data.readline()
            if line == "": return True
            self.parseRecord(line)
        return False
        
    def parseRecord(self, line):
        record = Record(line)
        score = record.score
        try:
            positions = record.computePositions(MINIMUM_MOVES_RANGE, MOVES_INCREMENT)
        except:
            return
        for position in positions:
            if self.currentIndex >= BUFFER_SIZE:
                self.extra.append( (position, score) )
            else:
                self.boardPositions[self.currentIndex] = position
                self.scores[self.currentIndex] = score
                self.currentIndex += 1
        self.recordsDone += 1
    
    def nextBatch(self, batchSize): # FIXME make it more straightforward
        nPicks = self.currentIndex + batchSize
        batchX = np.zeros([batchSize, 2, 19, 19])
        batchY = np.zeros([batchSize, 1])
        total = self.scores.shape[0]
        firstIndex = 0
        lastIndex = batchSize
        if nPicks > total:
            remaining = total - self.currentIndex
            trace("Refilling: %d remaining, current index %d with %d total records" % (remaining, self.currentIndex, total), 2)
            batchX[:remaining] = self.boardPositions[self.currentIndex : total]
            batchY[:remaining] = self.scores[self.currentIndex : total]
            firstIndex = remaining
            self.refillBuffers()
            if self.scores.shape[0] < (batchSize - remaining):  # Last records available
                lastIndex = self.scores.shape[0]
            else:
                lastIndex = batchSize
            trace("Refill: only %d records. Current index %d - indices %d - %d" % (remaining, self.currentIndex, firstIndex, lastIndex), 2)
        if lastIndex != batchSize:
            batchX = batchX[:lastIndex]
            batchY = batchY[:lastIndex]
        remaining = lastIndex - firstIndex
        batchX[firstIndex : lastIndex] = self.boardPositions[self.currentIndex : self.currentIndex + remaining]
        batchY[firstIndex : lastIndex] = self.scores[self.currentIndex : self.currentIndex + remaining]
        self.currentIndex += lastIndex
        # print(self.currentIndex)
        return batchX, batchY
    
    def restart(self):
        self.recordsDone = 0
        self.data.seek(0)
        self.extra = []
        self.refillBuffers()
        
if __name__ == "__main__":
    random.seed(345)
    if len(sys.argv) > 1:
        # sgf = sys.argv[1]
        # useNetwork([sgf])
        
        BUFFER_SIZE = 2000
        batchMaker = BatchMaker("test")
        bx, by = batchMaker.nextBatch(32)
        result = runNetwork(bx)
        for i in range(by.shape[0]):
            print("Network %f  ---  Expected %f" % (result[i], by[i]))
    else:
        batchMaker = BatchMaker("test")
        trainNetwork(batchMaker)