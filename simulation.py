import numpy as np
import time
from backend import step, count_mean
import pygame
from pygame.draw import rect
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

N = 100
dx = 1
g = 1
h = 1
dt = 0.02
n_eff = 2
block_size = 5
FPS = 1000
WHT = (255, 255, 255)
B = np.array([0, 0, 1])
S1 = np.repeat(np.array([0, 0, 1]), N, axis=-1).reshape((3, N)).T
S1 = np.repeat(S1, N, axis=0).reshape((N, N, 3))
S2 = np.random.random((N, N, 3))
S = S2
S = S / np.linalg.norm(S, axis=-1)[:, :, np.newaxis]


# print(S)


def draw_timer(S_curr, B, N, g, h, dx, dt, n_eff, Full_time=10):
    """

    Отрисовка решётки

    """
    C = []
    pygame.init()
    S_curr = S_curr / np.linalg.norm(S_curr, axis=-1)[:, :, np.newaxis]
    C0 = S_curr[:, :, 0].sum()
    screen = pygame.display.set_mode((int(N * block_size), int(N * block_size)))
    screen.fill(WHT)
    pygame.display.update()
    clock = pygame.time.Clock()
    timer = 0
    finished = False
    for i in range(N):
        for j in range(N):
            color = tuple((255.0 * (S_curr[i][j] * 0.5 + 0.5)).astype(int))
            # print(color)
            rect(screen, color, (i * block_size, j * block_size, block_size, block_size))

    for ind in tqdm(range(Full_time)):
        S_curr = S_curr / np.linalg.norm(S_curr, axis=-1)[:, :, np.newaxis]
        C.append(S_curr[:, :, 0].sum()/C0)
        clock.tick(FPS)
        st0 = time.time()
        screen.fill(WHT)
        for i in range(N):
            for j in range(N):
                color = tuple((255.0 * (S_curr[i][j] * 0.5 + 0.5)).astype(int))
                # print(color)
                rect(screen, color, (i * block_size, j * block_size, block_size, block_size))

        pygame.display.update()
        st1 = time.time()
        timer += 1
        if (timer <= Full_time):
            S_curr = step(S_curr, B, N, g, h, dx, dt, n_eff)
        else:
            finished = True
        st2 = time.time()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    finished = True
    pygame.quit()
    return C


B = np.array([0, 0, 1])
C = draw_timer(S2, B, N, g, h, dx, dt, n_eff, Full_time=10000)
C = count_mean(C, 100)
plt.plot(C)
plt.show()

# print(step1(S, B, N, g, h, dx, dt, n_eff))
