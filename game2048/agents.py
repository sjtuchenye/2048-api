import numpy as np
from keras.models import load_model


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
            ret[i, j, match_table(x)] = 1

    return ret


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent, object):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super(ExpectiMaxAgent, self).__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction


# class Myagent(Agent):
#     def step(self):
#         direction = 
class myAgent(Agent,object):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super(myAgent, self).__init__(game, display)
        self.game = game
        self.model_256 = load_model('myAgent_256_online_new.h5')
        self.model_512 = load_model('myAgent_512_online_new.h5')
        #self.model_512 = load_model('myAgent_512.h5')
        self.model_1024 = load_model('myAgent_1024_online_new.h5')
        self.model_2048 = load_model('myAgent_2048_online_new.h5')

    def step(self):
        #inputboard = np.zeros((1, 4, 4, 12))
        input_board = np.zeros((1,4,4,16))
        maxNum = 0
        for i in range(4):
            for j in range(4):
                num = self.game.board[i, j]
                if num > maxNum:
                    maxNum = num

        input_board[0] = onehot(self.game.board)
        #input_board = np.array(input_board)
                # if num == 0:
                #     inputboard[0, i, j, 0] = 1
                # else:
                #     inputboard[0, i, j, int(np.log2(num))] = 1
        # if maxNum <= 256:
        #     direction = self.model_256.predict(inputboard).tolist()[0]
        # elif maxNum == 512:
        #     direction = self.model_512.predict(inputboard).tolist()[0]
        # elif maxNum == 1024:
        #     direction = self.model_1024.predict(inputboard).tolist()[0]
        # return direction.index(max(direction))
        if maxNum < 256:
            direction = self.model_256.predict(input_board).tolist()[0]
            dir = direction.index(max(direction))
        #elif maxNum == 256:
        elif maxNum == 256:
            direction = self.model_512.predict(input_board).tolist()[0]
            dir = direction.index(max(direction))
        elif maxNum == 512:
            direction = self.model_1024.predict(input_board).tolist()[0]
            dir = direction.index(max(direction))
        elif maxNum == 1024:
            direction = self.model_2048.predict(input_board).tolist()[0]
            dir = direction.index(max(direction))
        else:
            dir = np.random.randint(0, 4)
        return dir
