import pygame

class VisualBar:
    def __init__(self, smoothing=0.9):
        pygame.init()
        self.screen = pygame.display.set_mode((500, 200))
        self.value = 0.5
        self.alpha = smoothing

    def update(self, target):
        self.value = self.alpha * self.value + (1 - self.alpha) * target
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(
            self.screen, (0, 255, 0),
            (0, 0, int(500 * self.value), 200)
        )
        pygame.display.flip()

