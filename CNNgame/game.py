import pygame
from enum import Enum
from collections import namedtuple
import random
import numpy as np
from collections import deque


pygame.init()
font = pygame.font.SysFont('arial', 15)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Block = namedtuple("Block", "x, y")

# Constants
WIN_WIDTH = 50
WIN_HEIGHT = 50
BLOCK_SIZE = 2
GAME_SPEED = 50

SNAKE_HEAD_COLOR = (0, 0, 255)
SNAKE_COLOR = (0, 255, 0)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)


class Snake:

    # x, y are the coordinates of the head of the snake, snakes starts going to the right
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.head = Block(self.x, self.y)
        self.snake_elements = [
            self.head,
            Block(self.head.x - BLOCK_SIZE, self.head.y),
            Block(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]

        # snake start going right
        self.current_direction = Direction.RIGHT

    
    def update(self, action):
        clock_wise_direction = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        i = clock_wise_direction.index(self.current_direction)
        
        # 3 possible actions
        # [1, 0, 0] -> go straight
        # [0, 1, 0] -> turn right
        # [0, 0, 1] -> turn left
        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise_direction[i]
        elif np.array_equal(action, [0, 1, 0]):
            new_idx = (i + 1) % 4
            new_direction = clock_wise_direction[new_idx]
        else:
            new_idx = (i - 1) % 4
            new_direction = clock_wise_direction[new_idx]

        # update direction of the snake
        self.current_direction = new_direction

        # if direction is right, then x coordinate of the head increases 
        # ---------> x
        #|
        #|
        #|
        #v y

        new_head_x = self.head.x
        new_head_y = self.head.y
        if self.current_direction == Direction.RIGHT:
            new_head_x += BLOCK_SIZE
        elif self.current_direction == Direction.LEFT:
            new_head_x -= BLOCK_SIZE
        elif self.current_direction == Direction.DOWN:
            new_head_y += BLOCK_SIZE
        elif self.current_direction == Direction.UP:
            new_head_y -= BLOCK_SIZE
        
        # update the head of the snake
        self.head = Block(new_head_x, new_head_y)
        self.snake_elements.insert(0, self.head)

    def draw(self, win):
        pygame.draw.rect(win, SNAKE_HEAD_COLOR, pygame.Rect(
            self.snake_elements[0].x, self.snake_elements[0].y, BLOCK_SIZE, BLOCK_SIZE))
        for block in self.snake_elements[1:]:
            pygame.draw.rect(win, SNAKE_COLOR, pygame.Rect(
                block.x, block.y, BLOCK_SIZE, BLOCK_SIZE))

    def is_collision(self, block=None):
        if block == None:
            block = self.head

        # hits itself
        if block in self.snake_elements[1:]:
            return True

        # hits boundary
        if block.x > WIN_WIDTH - BLOCK_SIZE or block.x < 0 or block.y > WIN_HEIGHT - BLOCK_SIZE or block.y < 0:
            return True

        return False

    def get_distance_from_apple(self, apple):
        # i will use manhattan distance
        return abs(self.head.x - apple.x) + abs(self.head.y - apple.y)


class Apple:

    def __init__(self, snake):
        self.x, self.y = self.spawn_apple(snake)

    def spawn_apple(self, snake):
        x = random.randint(0, (WIN_WIDTH - BLOCK_SIZE) //
                           BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (WIN_HEIGHT - BLOCK_SIZE) //
                           BLOCK_SIZE) * BLOCK_SIZE
        
        # recursive call if apple spawns on the snake
        for block in snake.snake_elements:
            if block.x == x and block.y == y:
                return self.spawn_apple(snake)
    
        return x, y

    def change_pos(self, snake):
        self.x, self.y = self.spawn_apple(snake)

    def draw(self, win):
        pygame.draw.rect(win, RED, pygame.Rect(
            self.x, self.y, BLOCK_SIZE, BLOCK_SIZE))


class SnakeGameAI:

    def __init__(self):
        self.width = WIN_WIDTH
        self.height = WIN_HEIGHT
        self.display = pygame.display.set_mode((self.width, self.height))
        self.frame = 0
        self.state_queue = deque(maxlen=4)
        pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.snake = Snake(self.width / 2, self.height / 2)
        self.apple = Apple(self.snake)
        self.game_over = False
        self.score = 0
        self.frame = 0

    def play_step(self, action):
        self.frame += 1
        old_distance = self.snake.get_distance_from_apple(self.apple)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
           
        self.snake.update(action)

        reward = 0
        if self.snake.is_collision(): 
            self.game_over = True
            reward -= 100
            return self._get_state(), reward, self.game_over, self.score

        if self.snake.head.x == self.apple.x and self.snake.head.y == self.apple.y:
            self.score += 1
            reward += 50
            self.apple.change_pos(self.snake)
        else:
            self.snake.snake_elements.pop()

        if self.snake.get_distance_from_apple(self.apple) < old_distance:
            reward += 10
        else:
            reward -= 10

        self.draw()
        self.clock.tick(GAME_SPEED)

        return self._get_state(), reward, self.game_over, self.score

    def draw(self):
        self.display.fill(BLACK)
        self.snake.draw(self.display)
        self.apple.draw(self.display)
        text = font.render(str(self.score), True, WHITE)
        #self.display.blit(text, [5, 5])
        pygame.display.flip()

    def _get_state(self):
        game_frame = pygame.surfarray.array3d(pygame.display.get_surface())
        game_frame_gray = np.mean(game_frame, axis=2).astype(np.uint8)

        self.state_queue.append(game_frame_gray)

        while len(self.state_queue) < 4:
            self.state_queue.append(game_frame_gray)

        state = np.stack(list(self.state_queue), axis=2)
        
        return state