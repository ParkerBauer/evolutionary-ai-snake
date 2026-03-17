import pygame
import neat
import os
import sys
import pickle
from snake_game import Snake, spawn_food, draw_food, draw_hud
from snake_game import GRID_SIZE, GRID_W, GRID_H, WINDOW_W, WINDOW_H
from snake_game import UP, DOWN, LEFT, RIGHT
from snake_game import GRAY, LIGHT_GRAY, WHITE, GREEN, RED

# Training configuration
MAX_STEPS     = 150   # max steps a snake can take without eating before being killed
GENERATIONS   = 500    # how many generations to train for
FPS_TRAINING  = 60    # speed when dev view is on (higher = faster)

# Dev view colors
SENSOR_WALL  = (255, 100, 100)   # red rays = wall distance
SENSOR_FOOD  = (100, 255, 100)   # green rays = food
SENSOR_BODY  = (100, 100, 255)   # blue rays = body

def draw_sensors(surface, snake, food_pos):
    directions = [UP, DOWN, LEFT, RIGHT,
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for direction in directions:
        pos  = list(snake.head)
        dist = 0

        while True:
            pos[0] += direction[0]
            pos[1] += direction[1]
            dist   += 1

            if not (0 <= pos[0] < GRID_W and 0 <= pos[1] < GRID_H):
                # Draw wall ray
                end_x = pos[0] * GRID_SIZE + GRID_SIZE // 2
                end_y = pos[1] * GRID_SIZE + GRID_SIZE // 2
                start_x = snake.head[0] * GRID_SIZE + GRID_SIZE // 2
                start_y = snake.head[1] * GRID_SIZE + GRID_SIZE // 2
                pygame.draw.line(surface, SENSOR_WALL,
                                 (start_x, start_y), (end_x, end_y), 1)
                break

            if tuple(pos) == food_pos:
                # Draw food ray
                end_x = pos[0] * GRID_SIZE + GRID_SIZE // 2
                end_y = pos[1] * GRID_SIZE + GRID_SIZE // 2
                start_x = snake.head[0] * GRID_SIZE + GRID_SIZE // 2
                start_y = snake.head[1] * GRID_SIZE + GRID_SIZE // 2
                pygame.draw.line(surface, SENSOR_FOOD,
                                 (start_x, start_y), (end_x, end_y), 1)
                break

            if tuple(pos) in snake.body:
                # Draw body ray
                end_x = pos[0] * GRID_SIZE + GRID_SIZE // 2
                end_y = pos[1] * GRID_SIZE + GRID_SIZE // 2
                start_x = snake.head[0] * GRID_SIZE + GRID_SIZE // 2
                start_y = snake.head[1] * GRID_SIZE + GRID_SIZE // 2
                pygame.draw.line(surface, SENSOR_BODY,
                                 (start_x, start_y), (end_x, end_y), 1)
                break

def draw_sidebar(surface, font, generation, score, fitness, alive_count, all_time_best):
    sidebar_x = 10
    items = [
        f"Generation : {generation}",
        f"Best Ever  : {all_time_best}",
        f"Score      : {score}",
        f"Fitness    : {fitness:.1f}",
        f"Alive      : {alive_count}",
        f"",
        f"D = toggle dev view",
        f"Q = quit",
    ]

    panel = pygame.Surface((200, len(items) * 22 + 16))
    panel.set_alpha(180)
    panel.fill((0, 0, 0))
    surface.blit(panel, (4, 4))

    for i, item in enumerate(items):
        text = font.render(item, True, WHITE)
        surface.blit(text, (sidebar_x, 12 + i * 22))

def eval_genomes(genomes, config, screen, clock, font, dev_view, all_time_best):    
    snakes    = []
    nets      = []
    ge        = []
    foods     = []

    # Create a snake, network, and food for each genome
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        snakes.append(Snake())
        nets.append(net)
        ge.append(genome)
        foods.append(spawn_food([]))

    generation_best = 0

    while True:
        clock.tick(FPS_TRAINING)

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_d:
                    dev_view[0] = not dev_view[0]

        # Update all snakes
        for i, snake in enumerate(snakes):
            if not snake.alive:
                continue

            # Kill snakes that are looping without eating
            if snake.steps_since_food > MAX_STEPS:
                snake.alive = False
                ge[i].fitness -= 50
                continue

            # Gradually punish the longer they go without eating
            if snake.steps_since_food > 100:
                ge[i].fitness -= 0.5

            # Get neural network inputs and activate
            inputs  = snake.get_inputs(foods[i])
            output  = nets[i].activate(inputs)
            move    = [UP, DOWN, LEFT, RIGHT][output.index(max(output))]
            snake.set_direction(move)
            snake.move()

            # Only reward/punish food distance if snake can see the food
            head = snake.head
            food = foods[i]
            inputs = snake.get_inputs(food)
            can_see_food = any(inputs[j] for j in range(1, 24, 3))

            if can_see_food:
                dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
                if not hasattr(snake, 'last_dist'):
                    snake.last_dist = dist
                if dist < snake.last_dist:
                    ge[i].fitness += 1.0
                else:
                    ge[i].fitness -= 0.5
                snake.last_dist = dist

            # Reward eating food
            if snake.head == foods[i]:
                snake.grow()
                foods[i] = spawn_food(snake.body)
                ge[i].fitness += 20

            if snake.score > generation_best:
                generation_best = snake.score
            if snake.score > all_time_best[0]:
                all_time_best[0] = snake.score

        # Check if all snakes are dead
        alive = [s for s in snakes if s.alive]
        if not alive:
            break

        # Drawing
        screen.fill(GRAY)
        for gx in range(0, WINDOW_W, GRID_SIZE):
            pygame.draw.line(screen, LIGHT_GRAY, (gx, 0), (gx, WINDOW_H))
        for gy in range(0, WINDOW_H, GRID_SIZE):
            pygame.draw.line(screen, LIGHT_GRAY, (0, gy), (WINDOW_W, gy))

        if dev_view[0]:
            # Find the best alive snake to highlight
            best = max(alive, key=lambda s: ge[snakes.index(s)].fitness)
            draw_sensors(screen, best, foods[snakes.index(best)])
            best.draw(screen)
            draw_food(screen, foods[snakes.index(best)])
            draw_sidebar(screen, font,
                         getattr(eval_genomes, 'generation', 0),
                         best.score,
                         ge[snakes.index(best)].fitness,
                         len(alive),
                         all_time_best[0])
        else:
            # Draw all snakes faintly when dev view is off
            for snake in alive:
                snake.draw(screen)
            draw_food(screen, foods[0])
            hud = font.render(f"Gen: {getattr(eval_genomes, 'generation', 0)}  Alive: {len(alive)}  Best Ever: {all_time_best[0]}  This Gen: {generation_best}", True, WHITE)
            screen.blit(hud, (8, 8))

        pygame.display.flip()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Snake AI — NEAT Training")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 16)
    all_time_best = [0]

    dev_view = [False]  # wrapped in list so eval_genomes can modify it

    # Load NEAT config
    config_path = os.path.join(os.path.dirname(__file__), "config.txt")
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Create population
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    # Track generation number for sidebar
    def run_generation(genomes, config):
        run_generation.generation += 1
        eval_genomes.generation    = run_generation.generation
        eval_genomes(genomes, config, screen, clock, font, dev_view, all_time_best)

        # Save the best genome after every generation
        best = max(population.population.values(), key=lambda g: g.fitness if g.fitness is not None else 0)
        with open("best_genome.pkl", "wb") as f:
            pickle.dump(best, f)
        print(f"  Saved best genome (fitness: {best.fitness:.1f})")
    run_generation.generation = 0

    # Load existing brain if available
    checkpoint_path = "best_genome.pkl"
    if os.path.exists(checkpoint_path):
        print("Found saved brain — loading checkpoint...")
        with open(checkpoint_path, "rb") as f:
            winner = pickle.load(f)

    # Run training
    winner = population.run(run_generation, GENERATIONS)

    # Save the best brain
    with open(checkpoint_path, "wb") as f:
        pickle.dump(winner, f)
    print(f"\nBest genome saved to {checkpoint_path}")

if __name__ == "__main__":
    main()