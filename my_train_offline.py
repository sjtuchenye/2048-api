import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout

NUM_EPOCHS = 3
NUM_CLASSES = 4 
BATCH_SIZE = 512
INPUT_SHAPE = (4, 4, 16)

def match_table(arrnum):
    if arrnum==0:
        return 0
    for i in range(1,16):
        if arrnum == 2**i:
            return i


def onehot(arr):
    # match_table = {2**i: i for i in range(1,16)}
    # match_table[0] = 0
    ret = np.zeros((4,4,16), dtype=bool)
    for i in range(4):
        for j in range(4):
            x = arr[i,j]
            #print(x)
            #print(match_table(x))
            ret[i, j, match_table(x)]=1
    
    return ret

inputs = Input(shape=INPUT_SHAPE)
layer1_1 = Conv2D(512, (2, 1), strides=(1, 1), activation='relu')(inputs)
layer1_2 = Conv2D(512, (1, 2), strides=(1, 1), activation='relu')(inputs)
layer2_1 = Conv2D(256, (1, 2), strides=(1, 1), activation='relu')(layer1_1)
layer2_2 = Conv2D(256, (2, 1), strides=(1, 1), activation='relu')(layer1_2)
layer2_3 = Conv2D(256, (1, 2), strides=(1, 1), activation='relu')(layer1_1)
layer2_4 = Conv2D(256, (2, 1), strides=(1, 1), activation='relu')(layer1_2)
layer2_5 = Conv2D(256, (1, 2), strides=(1, 1), activation='relu')(layer1_1)
layer2_6 = Conv2D(256, (2, 1), strides=(1, 1), activation='relu')(layer1_2)
layer2 = keras.layers.concatenate([layer2_1, layer2_2, layer2_3, layer2_4, layer2_5, layer2_6])
layer3 = Flatten()(layer2)
layer4 = Dense(512, activation='relu')(layer3)
layer5 = Dense(64, activation='relu')(layer4)
outputs = Dense(NUM_CLASSES, activation='softmax')(layer5)
model = Model(inputs=inputs, outputs=outputs)


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#for epoch in range(NUM_EPOCHS):
f1 = open("dataset_256_new.txt")
# f2 = open("dataset_1024_3.txt")
# f3 = open("dataset_2048_3.txt")
#for k in range(5):
boards = []
directions = []
###从txt载入数据，前16行为棋盘，第17行为决策方向，17行为一组
for j in range(3000000):
    num = f1.readline()
    if not num:
        break
    num = float(num)
    board = np.zeros((4, 4))
    for p in range(4):
        for q in range(4):
            # if num == 0:
            #     board[p, q, 0] = 1
            # else:
            #     board[p, q, int(np.log2(num))] = 1
            # num = float(f1.readline())
            board[p,q] = num
            num = float(f1.readline())
    board = onehot(board)
    boards.append(board) 
    direction = int(num)
    directions.append(direction)


boards = np.array(boards)
directions= np.array(directions)
directions = keras.utils.to_categorical(directions, num_classes=NUM_CLASSES)
model.fit(boards, directions, epochs=2, batch_size=BATCH_SIZE, validation_split=0.3)
f1.close()
model.save('myAgent_256_new.h5')

