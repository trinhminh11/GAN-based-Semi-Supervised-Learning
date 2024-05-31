RANDOM_SEED = 11042004

DATA_DIR = "./data"

# USED_DATA = "CIFAR10"
# USED_DATA = "MNIST"
# USED_DATA = "EMNIST"
USED_DATA = "DOODLE"

NUM_LABELLED = -1

DEVICE = "cuda:0"

# GAN Config
GAN_BATCH_SIZE = 128

'''
CNN:
	CIFAR-10:
	epochs = 50, optim = Adam, sched = True

	full label: BATCH_SIZE = 512, LEARNING_RATE = 0.01
	other: BATCH_SIZE = 64, 128, 256, LEARNING_RATE = 0.001

	MNIST:
	LEARNING_RATE = 0.0002, optim = Adam, sched = False

	full label: BATCH_SIZE = 512, epochs=5,
	other: BATCH_SIZE = 32, epochs = 20
GANSSL:
	CIFAR-10:


	MNIST:
	epochs = 20, batch_size = 64, step_per_epoch = 100, lr = 0.00001, optim = RMSprop

'''

# Pygame CONST
WIDTH, HEIGHT = 820, 740
FPS = None

DRAW_WIDTH, DRAW_HEIGHT = (640, 640)

DATASETS = ["MNIST", "DOODLE", "EMNIST"]

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


BACKGROUND = (127, 127, 127)