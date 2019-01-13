import numpy as np
import keras
from keras.models import load_model
from game2048.game import Game
from game2048.agents import ExpectiMaxAgent


def match_table(arrnum):
    if arrnum == 0:
        return 0
    for i in range(1, 16):
        if arrnum == 2 ** i:
            return i


def onehot(arr):
    # match_table = {2**i: i for i in range(1,16)}
    # match_table[0] = 0
    ret = np.zeros((4, 4, 16), dtype=bool)
    for i in range(4):
        for j in range(4):
            x = arr[i, j]
            # print(x)
            # print(match_table(x))
            ret[i, j, match_table(x)] = 1

    return ret


NUM_EPOCHS = 3
NUM_CLASSES = 4
BATCH_SIZE = 1024

model_0_256 = load_model('myAgent_256_online_new.h5')
model_256_512 = load_model('myAgent_512_online_new.h5')
model_512_1024 = load_model('myAgent_1024_online_new.h5')
model_1024_2048 = load_model('myAgent_2048_online_new.h5')

boards_0_256 = []
boards_256_512 = []
boards_512_1024 = []
boards_1024_2048 = []
directions_0_256 = []
directions_256_512 = []
directions_512_1024 = []
directions_1024_2048 = []

for i in range(30000):
    game = Game(size=4,score_to_win=2048)
    expectiMaxAgent = ExpectiMaxAgent(game=game)
    while True: 
        rightDirection = expectiMaxAgent.step()
        if (game.end == True): 
            break
        maxNum = 0
        for p in range(4):
            for q in range(4):
                if game.board[p, q] > maxNum:
                    maxNum = game.board[p, q]
        if maxNum == 2048: 
            break
        input_board = np.zeros((1,4,4,16))
        inputboard = game.board
        inputboard = onehot(inputboard)
        input_board[0] = inputboard
        if maxNum < 256:
            boards_0_256.append(inputboard)
            directions_0_256.append(rightDirection)
            myDirection = model_0_256.predict(input_board).tolist()[0]
        elif maxNum == 256:
            boards_256_512.append(inputboard)
            directions_256_512.append(rightDirection)
            myDirection = model_256_512.predict(input_board).tolist()[0]
        elif maxNum == 512:
            boards_512_1024.append(inputboard)
            directions_512_1024.append(rightDirection)
            myDirection = model_512_1024.predict(input_board).tolist()[0]
        elif maxNum == 1024:
            boards_1024_2048.append(inputboard)
            directions_1024_2048.append(rightDirection)
            myDirection = model_1024_2048.predict(input_board).tolist()[0]

        game.move(myDirection.index(max(myDirection)))

    if len(boards_0_256) >= 400000:
        boards_0_256 = np.array(boards_0_256)
        directions_0_256 = np.array(directions_0_256)
        directions_0_256 = keras.utils.to_categorical(directions_0_256, num_classes=NUM_CLASSES)
        print("model_0_256_online")
        model_0_256.fit(boards_0_256, directions_0_256, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
        model_0_256.save('myAgent_256_online_new.h5')
        boards_0_256 = []
        directions_0_256 = []

    if len(boards_256_512) >= 600000:
        boards_256_512 = np.array(boards_256_512)
        directions_256_512 = np.array(directions_256_512)
        directions_256_512 = keras.utils.to_categorical(directions_256_512, num_classes=NUM_CLASSES)
        print("model_256_512_online")
        model_256_512.fit(boards_256_512, directions_256_512, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
        model_256_512.save('myAgent_512_online_new.h5')
        boards_256_512 = []
        directions_256_512 = []

    if len(boards_512_1024) >= 800000:
        boards_512_1024 = np.array(boards_512_1024)
        directions_512_1024 = np.array(directions_512_1024)
        directions_512_1024 = keras.utils.to_categorical(directions_512_1024, num_classes=NUM_CLASSES)
        print("model_512_1024_online")
        model_512_1024.fit(boards_512_1024, directions_512_1024, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
        model_512_1024.save('myAgent_1024_online_new.h5')
        boards_512_1024 = []
        directions_512_1024 = []

    if len(boards_1024_2048) >= 1000000:
        boards_1024_2048 = np.array(boards_1024_2048)
        directions_1024_2048 = np.array(directions_1024_2048)
        directions_1024_2048 = keras.utils.to_categorical(directions_1024_2048, num_classes=NUM_CLASSES)
        print("model_1024_2048_online")
        model_1024_2048.fit(boards_1024_2048, directions_1024_2048, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
        model_1024_2048.save('myAgent_2048_online_new.h5')
        boards_1024_2048 = []
        directions_1024_2048 = []
