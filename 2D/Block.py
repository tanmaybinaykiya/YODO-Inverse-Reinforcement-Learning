import pygame

from constants import COLORS


class Block(pygame.sprite.Sprite):

    def __init__(self, color_index, grid_center, block_size=50):
        super(Block, self).__init__()
        self.block_size = block_size
        self.surf = pygame.Surface((block_size, block_size))
        self.surf.fill(COLORS[color_index])
        self.id = color_index
        self.rect = self.surf.get_rect(center=grid_center)
