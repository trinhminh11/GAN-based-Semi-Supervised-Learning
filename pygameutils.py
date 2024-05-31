import pygame
from pygame import gfxdraw
from typing import Literal

from config import *

from typing import overload

import pygame.freetype

def draw_circle(surface, color, pos, radius, ):
    gfxdraw.aacircle(surface, *pos, radius, color)
    gfxdraw.filled_circle(surface, *pos, radius, color)

pygame.freetype.init()

my_font = pygame.freetype.SysFont("Arial", 22)


class Genom:
	def draw(self, screen):
		raise NotImplementedError

class Label(Genom):
	def __init__(self, text, x, y, width = 150, height = 50, align = 'left', size = 22):
		self.text = text
		self.rect = (x, y, width, height)

		self.align = align

		self.size = size
	
	def set_align(self, align):
		self.align = align
	
	def draw(self, screen: pygame.Surface):
		text_surface, _ = my_font.render(self.text, WHITE, size=self.size)
		# text_rect = text_surface.get_rect()
		if self.align == 'left':
			rect = text_surface.get_rect(center = (self.rect[0] + text_surface.get_rect()[2]//2 + 5, self.rect[1] + self.rect[3]//2))
		if self.align == 'center':
			rect = text_surface.get_rect(center = (self.rect[0] + self.rect[2]//2, self.rect[1] + self.rect[3]//2))
		
		screen.blit(text_surface, rect)

		pygame.draw.rect(screen, WHITE, self.rect, 1)

class Circle(Genom):
	def __init__(self, color, pos, R):
		self.color = color
		self.pos = pos
		self.R = R
	
	def draw(self, screen):
		pygame.draw.circle(screen, self.color, self.pos, self.R)

class Line(Genom):
	def __init__(self, color, start, end, width = 2):
		self.color = color
		self.start = start
		self.end = end
		self.width = width
	
	def draw(self, screen):
		draw_circle(screen, self.color, self.start, self.width//2-2)
		pygame.draw.line(screen, self.color, self.start, self.end, self.width)
		draw_circle(screen, self.color, self.end, self.width//2-2)

class Table:
	def __init__(self, x, y, width = 150, height = 50, pos: Literal['left', 'right'] = 'left', align = 'center', size = 22):
		self.pos = pos

		self.table: list[list[Label]] = []


		self.i = 0
		self.j = 0

		self.x = x 
		if self.pos == 'right':
			self.x = x - width
		
		self.initial_y = y

		self.y = y

		self.w = width
		self.h = height

		self.align = align

		self.size = size
	
	def add_col(self, data):
		for d in data:
			try:
				self.table[self.i]
			except:
				self.table.append([])
			
			try:
				self.table[self.i][self.j]
			except:
				self.table[self.i].append(None)

			self.table[self.i][self.j] = Label(d, self.x, self.y, self.w, self.h, self.align, self.size)

			self.y += self.h
			self.i += 1
		
		self.j += 1
		self.i = 0
		self.y = self.initial_y

		if self.pos == 'left':
			self.x += self.w
		else:
			self.x -= self.w

	
	def draw(self, screen):
		for row in self.table: 
			for l in row:
				l.draw(screen)


class Button:
	def __init__(self, text, rect, size=22):
		self.text = text
		self.rect = rect
		self.color = (51, 51, 51)
		self.size = size

		self.active = False

	
	def reset_color(self):
		self.color = (51, 51, 51)
		self.active = False
	
	def clicked(self):
		self.color = BLACK
		self.active = True
	
	def check(self, pos):
		if pos[0] >= self.rect[0] and pos[1] >= self.rect[1] and pos[0] <= self.rect[0] + self.rect[2] and pos[1] <= self.rect[1] + self.rect[3]:
			return True
		
		return False
	
	
	def draw(self, screen):
		text_surface, _ = my_font.render(self.text, WHITE, size = self.size)

		rect = text_surface.get_rect(center = (self.rect[0] + self.rect[2]//2, self.rect[1] + self.rect[3]//2))
		
		pygame.draw.rect(screen, self.color, self.rect)
		screen.blit(text_surface, rect)
		pygame.draw.rect(screen, WHITE, self.rect, 1)



class GroupButton:
	def __init__(self) -> None:
		self.buttons: list[Button] = []
	
	def reset(self):
		for button in self.buttons:
			button.reset_color()

	@overload
	def clicked(self, mouse_pos: list): pass

	@overload
	def clicked(self, idx: int): pass

	def clicked(self, param):
		if type(param) == int:
			self.buttons[param].clicked()

		else:
			for button in self.buttons:
				if button.check(param):
					self.reset()
					button.clicked()
					return 
	
	def get_clicked(self):
		for button in self.buttons:
			if button.active:
				return button.text
	
	def add_button(self, text, rect, size = 22):
		self.buttons.append(Button(text, rect, size))
	
	def draw(self, screen):
		for button in self.buttons:
			button.draw(screen)