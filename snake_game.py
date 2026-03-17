import pygame # type: ignore
import random
import sys

# Configuration
GRID_SIZE = 20
GRID_W    = 30
GRID_H    = 30
WINDOW_W  = GRID_SIZE * GRID_W
WINDOW_H  = GRID_SIZE * GRID_H
FPS       = 10

# Colors
BLACK      = (  0,   0,   0)
WHITE      = (255, 255, 255)
GREEN      = ( 34, 197,  94)
DARK_GREEN = ( 21, 128,  61)
RED        = (220,  38,  38)
GRAY       = ( 30,  30,  30)
LIGHT_GRAY = ( 50,  50,  50)

# Directions (dx, dy)
UP    = ( 0, -1)
DOWN  = ( 0,  1)
LEFT  = (-1,  0)
RIGHT = ( 1,  0)

# Snake class
class Snake:
    def __init__(self):
        start_x = GRID_W // 2
        start_y = GRID_H // 2
        self.body      = [(start_x, start_y),
                          (start_x - 1, start_y),
                          (start_x - 2, start_y)]
        self.direction = RIGHT
        self.alive     = True
        self.score     = 0
        self.steps_since_food = 0

    def set_direction(self, new_dir):
        opposite = (-self.direction[0], -self.direction[1])
        if new_dir != opposite:
            self.direction = new_dir

    def look_in_direction(self, direction, food_pos):
        dist     = 0
        pos      = list(self.head)
        see_food = 0
        see_body = 0

        while True:
            pos[0] += direction[0]
            pos[1] += direction[1]
            dist   += 1

            if not (0 <= pos[0] < GRID_W and 0 <= pos[1] < GRID_H):
                break
            if tuple(pos) == food_pos and see_food == 0:
                see_food = 1
            if tuple(pos) in self.body and see_body == 0:
                see_body = 1

        wall_dist = 1.0 / dist
        return [wall_dist, see_food, see_body]

    def get_inputs(self, food_pos):
        directions = [UP, DOWN, LEFT, RIGHT,
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]
        inputs = []
        for d in directions:
            inputs.extend(self.look_in_direction(d, food_pos))
        return inputs

    def move(self):
        if not self.alive:
            return False

        head_x, head_y = self.body[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        # Wall collision
        if not (0 <= new_head[0] < GRID_W and 0 <= new_head[1] < GRID_H):
            self.alive = False
            return False

        # Self collision
        if new_head in self.body:
            self.alive = False
            return False

        # Move forward
        self.body.insert(0, new_head)
        self.body.pop()

        self.steps_since_food += 1
        return True
        
    def grow(self):
        self.body.append(self.body[-1])
        self.score += 1
        self.steps_since_food = 0

    @property
    def head(self):
        return self.body[0]

    def draw(self, surface):
        for i, (x, y) in enumerate(self.body):
            rect = pygame.Rect(x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
            color = GREEN if i == 0 else DARK_GREEN
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, GRAY, rect, 1)

# Food
def spawn_food(snake_body):
    while True:
        pos = (random.randint(0, GRID_W - 1), random.randint(0, GRID_H - 1))
        if pos not in snake_body:
            return pos

def draw_food(surface, pos):
    x, y = pos
    rect = pygame.Rect(x * GRID_SIZE + 2, y * GRID_SIZE + 2,
                       GRID_SIZE - 4, GRID_SIZE - 4)
    pygame.draw.rect(surface, RED, rect, border_radius=4)


# HUD
def draw_hud(surface, font, score):
    text = font.render(f"Score: {score}", True, WHITE)
    surface.blit(text, (8, 8))

# Game over screen
def game_over_screen(surface, font_big, font_small, score):
    overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 160))
    surface.blit(overlay, (0, 0))

    msg1 = font_big.render("GAME OVER", True, RED)
    msg2 = font_small.render(f"Score: {score}", True, WHITE)
    msg3 = font_small.render("Press R to restart  |  Q to quit", True, LIGHT_GRAY)

    surface.blit(msg1, msg1.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 - 40)))
    surface.blit(msg2, msg2.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 + 10)))
    surface.blit(msg3, msg3.get_rect(center=(WINDOW_W // 2, WINDOW_H // 2 + 50)))

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return True
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit(); sys.exit()

# Main loop
def main():
    pygame.init()
    screen  = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Snake  —  Phase 1")
    clock   = pygame.time.Clock()
    font_sm = pygame.font.SysFont("monospace", 18)
    font_lg = pygame.font.SysFont("monospace", 48, bold=True)

    while True:
        snake = Snake()
        food  = spawn_food(snake.body)

        running = True
        while running:
            clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_UP,    pygame.K_w): snake.set_direction(UP)
                    if event.key in (pygame.K_DOWN,  pygame.K_s): snake.set_direction(DOWN)
                    if event.key in (pygame.K_LEFT,  pygame.K_a): snake.set_direction(LEFT)
                    if event.key in (pygame.K_RIGHT, pygame.K_d): snake.set_direction(RIGHT)
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit()

            snake.move()

            if not snake.alive:
                running = False
                continue

            if snake.head == food:
                snake.grow()
                food = spawn_food(snake.body)

            screen.fill(GRAY)
            for gx in range(0, WINDOW_W, GRID_SIZE):
                pygame.draw.line(screen, LIGHT_GRAY, (gx, 0), (gx, WINDOW_H))
            for gy in range(0, WINDOW_H, GRID_SIZE):
                pygame.draw.line(screen, LIGHT_GRAY, (0, gy), (WINDOW_W, gy))

            draw_food(screen, food)
            snake.draw(screen)
            draw_hud(screen, font_sm, snake.score)
            pygame.display.flip()

        game_over_screen(screen, font_lg, font_sm, snake.score)

if __name__ == "__main__":
    main()