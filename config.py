RANDOM_SEED = 11042004

DATA_DIR = "./data"

USED_DATA = "CIFAR10"
# USED_DATA = "MNIST"

NUM_LABELLED = 5000

DEVICE = "cuda:0"

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.01
SCHED = True

'''
CIFAR-10:
full label: epochs = 50, BATCH_SIZE = 512, LEARNING_RATE = 0.01, weight_decay = 0, grad_clip = None, optim = Adam, sched = True
other: BATCH_SIZE = 64

MNIST:
epochs = 20, LEARNING_RATE = 0.0004, threshold = 0.09925, weight_decay = 0, grad_clip = None, optim = Adam, sched = False

full label: BATCH_SIZE = 512
other: BATCH_SIZE = 32
'''



# GAN Config

GAN_BATCH_SIZE = 512