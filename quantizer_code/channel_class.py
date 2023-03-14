import numpy as np

class Channel:
    def __init__(self, epsilon, bit_al_mat):
        self.epsilon = epsilon
        self.bit_allocation_matrix = bit_al_mat

    # Send encoded image through noisy (memoryless) channel
    def send_image(self, channel_in):
        height = int(channel_in.shape[0]/8)
        width = int(channel_in.shape[1]/8)
        channel_out = np.zeros(shape=(height*8, width*8))
        for i in range(height):
            for j in range(width):
                block = channel_in[i*8:(i+1)*8, j*8:(j+1)*8]
                channel_out[i*8:(i+1)*8, j*8:(j+1)*8] = self.__send_block(block)
        return channel_out

    def __send_block(self, block):
        for i in range(8):
            for j in range(8):
                bits = '{0:0{len}b}'.format(block[i][j], len=self.bit_allocation_matrix[i, j])
                newBits = ""
                for k in range(len(bits)):
                    if np.random.binomial(1, self.epsilon):
                        # flip bit
                        newBits = newBits + str((int(bits[k], 2) + 1) % 2)
                    else:
                        newBits = newBits + bits[k]
                block[i][j] = int(newBits, 2)
        return block


