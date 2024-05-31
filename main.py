import pygame
from config import *
import torch
import model
import torchvision.transforms as tt
import numpy as np

from typing import Type

from pygameutils import Genom, Line, Table, Button, GroupButton, Label

from PIL import Image

from model import Model
import utils

import pickle

toknn = tt.Compose([
	tt.Resize(28),
	tt.Normalize([-1], [2])
])


def resize640(image: torch.Tensor):
	image = tt.Resize(32)(image)
	image = image.unsqueeze(0)

	return torch.nn.functional.upsample(image, 640)[0]


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class PixelArr2Tensor:
	def __init__(self):
		pass

	def __call__(self, pixel_arr: np.ndarray) -> torch.Tensor:
		if pixel_arr.ndim == 2:
			pixel_arr = np.expand_dims(pixel_arr, 2)

		arr = np.swapaxes(pixel_arr, 0, 1)

		compose = tt.Compose([
			tt.ToPILImage(),
			tt.Grayscale(),
			tt.Resize(32),
			tt.ToTensor(),
			tt.Normalize([0.5], [0.5])
		])

		return compose(arr)
		

class MyApp:
	SCREEN_OFFSET = (WIDTH - DRAW_WIDTH, HEIGHT - DRAW_HEIGHT)
	def __init__(self) -> None:
		pygame.init()
		pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP])
		self.main_screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)

		self.draw_screen = pygame.Surface((DRAW_WIDTH, DRAW_HEIGHT))

		self.main_screen.set_alpha(None)
		self.draw_screen.set_alpha(None)

		self.result_table = Table(WIDTH-10, 10, height=25, pos = 'right')

		self.clock = pygame.time.Clock()

		self.genoms: list[Genom] = []

		self.to_Tensor = tt.Compose([
			tt.Resize(32),
			tt.Grayscale(),
			tt.ToTensor(),
			tt.Normalize([0.5], [0.5], inplace=True),
		])

		self.prev_pos = None

		self.models: list[dict[str, dict[str, model.Image_CLassification_Model]]] = []
		self.generatator: dict[str, model.Generator] = {}

		self.dataset = 'MNIST'

		self.n_sup = '50'

		self.n_sup_buttons = GroupButton()

		self.dataset_button = GroupButton()

		_y = 10
		_h = 75
		_step = 15

		self.generated_button = Button('Generate', (15, _y, 150, _h))
		self.clear_canvas = Button("Clear", (15 + 150 + 15 + 15, _y, 150, _h))
		_y += _h + _step



		self.lines = [(WHITE, (0, _y), (180, _y))]

		_y += _step - 5

		self.dataset_button.add_button("MNIST", (15, _y, 150, _h))
		self.dataset_button.clicked(0)
		_y += _h + _step
		self.dataset_button.add_button("DOODLE", (15, _y, 150, _h))
		_y += _h + _step
		self.dataset_button.add_button("EMNIST", (15, _y, 150, _h))
		_y += _h + _step

		self.lines.append((WHITE, (0, _y), (180, _y)))

		_y += _step - 5

		self.n_sup_buttons.add_button('50', (15, _y, 150, _h))
		self.n_sup_buttons.clicked(0)
		_y += _h + _step
		self.n_sup_buttons.add_button('100', (15, _y, 150, _h))
		_y += _h + _step
		self.n_sup_buttons.add_button('500', (15, _y, 150, _h))
		_y += _h + _step
		self.n_sup_buttons.add_button('full', (15, _y, 150, _h))
		_y += _h + _step

		self.click = True

		self.generated_screen = None

		l = ['helicopter', 'car', 'book', 'windmill', 'cat', 'umbrella', 'octopus', 'bird', 'hat', 'birthday cake']

		self.labels = [Label(l[i], WIDTH+10, 10+i*50) for i in range(10)]

	def add_model(self, name: str, m: dict[str, dict[str, model.Image_CLassification_Model]]):
		self.result_table.add_col([name, '0', '0%'])

		self.models.append(m)
	
	def add_gen(self, name: str, m: model.Generator):
		self.generatator[name] = m

	def pred(self, images): 
		for i in range(len(self.models)):
			if i == len(self.models)-1 and self.n_sup == 'full':
				self.result_table.table[1][i].text = "slow"
				self.result_table.table[2][i].text = "slow"
				continue
			try:
				m = self.models[i][self.dataset][self.n_sup]
				if i == len(self.models)-1:
					images = toknn(images)
				
				pred, res = m.evaluate(images)

				if self.dataset == 'DOODLE':
					d = {0: 'helicopter', 1: 'car', 2: 'book', 3: 'windmill', 4: 'cat', 5: 'umbrella', 6: 'octopus', 7: 'bird', 8: 'hat', 9: 'birthday cake'}
					pred = d[pred]
				
				if self.dataset == "EMNIST":
					d = {i: chr(65+i) for i in range(26)}
					pred = d[pred]
					
				self.result_table.table[1][i].text = str(pred)
				self.result_table.table[2][i].text = f'{res:.2f}%'
			except KeyError:
				self.result_table.table[1][i].text = "_"
				self.result_table.table[2][i].text = "_"
	
	def to_draw_screen(self, arr: np.ndarray):
		arr = arr.swapaxes(0, 1)
		pixel_arr = pygame.surfarray.pixels3d(self.draw_screen)
		pixel_arr[:, :, :] = arr[:, :, :]

	
	def draw(self):
		self.main_screen.fill(BACKGROUND)
		self.draw_screen.fill(BLACK)

		if pygame.mouse.get_pressed()[0]:
			mouse_pos = list(pygame.mouse.get_pos())
			
			if self.click:
				if self.generated_button.check(mouse_pos):
					z = torch.randn((100, 1, 1))
					img: torch.Tensor = self.generatator[self.dataset].forward(z)[0]
					img = img*0.5 + 0.5
					img *= 255

					gen_img: np.ndarray = torch.cat((img, img, img))
					print(gen_img.shape)

					gen_img = resize640(gen_img).permute(1, 2, 0).detach().numpy().astype(np.uint8)


					self.generated_screen = gen_img
				
				self.dataset_button.clicked(mouse_pos)

				self.n_sup_buttons.clicked(mouse_pos)
				
				
				if self.clear_canvas.check(mouse_pos):
					self.draw_screen.fill(BLACK)
					self.genoms = []
					self.generated_screen = None

				
				self.dataset = self.dataset_button.get_clicked()

				if self.dataset == "EMNIST":
					self.n_sup_buttons.buttons[0].text = "100"
					self.n_sup_buttons.buttons[1].text = "200"
					self.n_sup_buttons.buttons[2].text = "1000"


					if self.n_sup == '50':
						self.n_sup = '100'
					elif self.n_sup == '100':
						self.n_sup = '200'
					elif self.n_sup == '500':
						self.n_sup = '1000'

				else:
					self.n_sup_buttons.buttons[0].text = "50"
					self.n_sup_buttons.buttons[1].text = "100"
					self.n_sup_buttons.buttons[2].text = "500"
				
				self.n_sup = self.n_sup_buttons.get_clicked()
				

				self.click = False


			mouse_pos[0] -= self.SCREEN_OFFSET[0]
			mouse_pos[1] -= self.SCREEN_OFFSET[1]
			if self.prev_pos != None:
				self.genoms.append(Line((250, 250, 250), self.prev_pos, mouse_pos, 35))
			
			self.prev_pos = mouse_pos
		else:
			self.click = True
			self.prev_pos = None
		
		for line in self.lines:
			pygame.draw.line(self.main_screen, *line)

		
		self.generated_button.draw(self.main_screen)
		
		self.n_sup_buttons.draw(self.main_screen)

		self.dataset_button.draw(self.main_screen)

		self.clear_canvas.draw(self.main_screen)

		if self.generated_screen is not None:
			self.to_draw_screen(self.generated_screen)
		
		else:
			for genom in self.genoms:
				genom.draw(self.draw_screen)

		
		if self.dataset == "DOODLE":
			for l in self.labels:
			
				l.draw(self.main_screen)


		self.main_screen.blit(self.draw_screen, self.SCREEN_OFFSET)

		arr = pygame.surfarray.array3d(self.draw_screen)

		arr = arr.swapaxes(0, 1)

		img: Image.Image = Image.fromarray(arr)

		img = img.resize((32, 32))

		img.save('test.png')


		self.pred(self.to_Tensor(img))


		self.result_table.draw(self.main_screen)


		pygame.display.update()
	

	def run(self):
		run = True
		while run:
			self.clock.tick()

			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					run = False
					break
					
				if event.type == pygame.KEYDOWN:
					if event.key == pygame.K_SPACE:
						self.draw_screen.fill(BLACK)
						self.genoms = []
						self.generated_screen = None

			self.draw()
			

		pygame.quit()

def init_DL_model():
	models: dict[str, dict[str, dict[str, model.Image_CLassification_Model]]] = {'GANSSL': {}, 'CNN': {}}

	all_models: dict[str, Type[model.Image_CLassification_Model]] = {'GANSSL': {}, 'CNN': {}}

	for name, value in model.__dict__.items():
		if type(value) is type:
			if model.Image_CLassification_Model in value.__bases__:
				all_models[name] = value
				# models[name] = {}
	
	for name in models.keys():
		for dataset in DATASETS:
			n_classes = 10
			if dataset == "EMNIST":
				n_classes = 26
			models[name][dataset] = {}

			ns = ['50', '100', '500']

			if dataset == "EMNIST":
				ns = ['100', '200', '1000']
			
			for n in ns:
				file_state_dict = f'{dataset}/{name}/_{n}.pt'
				
				m = all_models[name](1, n_classes)

				# m.eval()
				m.load(file_state_dict)
				models[name][dataset][n] = m
			
			try:
				file_state_dict = f'{dataset}/{name}/_full.pt'
				m = all_models[name](1, n_classes)
				# m.eval()
				m.load(file_state_dict)
				models[name][dataset]['full'] = m
			except:
				pass 

	return models

def init_gen_model():
	models = {'DOODLE': model.Generator(100, 1), 'MNIST': model.Generator(100, 1), "EMNIST": model.Generator(100, 1)}
	models['DOODLE'].load_state_dict(torch.load('GAN/DOODLE/netG_epoch_009.pt', 'cpu'))
	models['MNIST'].load_state_dict(torch.load('GAN/MNIST/netG_epoch_009.pt', 'cpu'))
	models['EMNIST'].load_state_dict(torch.load('GAN/EMNIST/netG_epoch_009.pt', 'cpu'))

	models['DOODLE'].eval()
	models['MNIST'].eval()
	models['EMNIST'].eval()

	return models

def init_KNN():
	models = {'DOODLE': {}, 'MNIST': {}, "EMNIST": {}}

	

	for dataset in DATASETS:
		ns = ['50', '100', '500']

		if dataset == "EMNIST":
			ns = ['100', '200', '1000']

		for n in ns:
			with open(f"{dataset}/KNN/_{n}.pkl", 'rb') as f:
				models[dataset][str(n)] = pickle.load(f)
	
	return models

def main():
	models = init_DL_model()
	gens = init_gen_model()
	knn = init_KNN()


	app = MyApp()


	for name, ms in models.items():
		app.add_model(name, ms)

	app.add_model('KNN', knn)

	
	for name, m in gens.items():
		app.add_gen(name, m)

	app.run()

if __name__ == "__main__":
	main()

