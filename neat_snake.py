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
FPS_TRAINING  = [120]    # speed when dev view is on (higher = faster)

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
        f"+/- change FPS: {FPS_TRAINING[0]}",
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

def draw_graph(surface, font, history_best, history_avg):
    panel_x = WINDOW_W
    panel_w = 300
    panel_h = WINDOW_H

    # Background
    pygame.draw.rect(surface, (15, 15, 15), (panel_x, 0, panel_w, panel_h))
    pygame.draw.line(surface, LIGHT_GRAY, (panel_x, 0), (panel_x, panel_h), 2)

    title = font.render("Training Progress", True, WHITE)
    surface.blit(title, (panel_x + 10, 10))

    if len(history_best) < 2:
        msg = font.render("Waiting for data...", True, LIGHT_GRAY)
        surface.blit(msg, (panel_x + 10, 40))
        return

    graph_x      = panel_x + 20
    graph_y      = 60
    graph_w      = panel_w - 40
    graph_h      = panel_h - 180

    # Draw graph border
    pygame.draw.rect(surface, LIGHT_GRAY, (graph_x, graph_y, graph_w, graph_h), 1)

    max_val = max(max(history_best), 1)

    def to_screen(idx, val):
        x = graph_x + int(idx / max(len(history_best) - 1, 1) * graph_w)
        y = graph_y + graph_h - int(val / max_val * graph_h)
        return (x, y)

    # Draw avg fitness line (yellow)
    if len(history_avg) >= 2:
        avg_normalized = [max(v, 0) for v in history_avg]
        for j in range(1, len(avg_normalized)):
            pygame.draw.line(surface, (255, 220, 50),
                             to_screen(j - 1, avg_normalized[j - 1]),
                             to_screen(j, avg_normalized[j]), 1)

    # Draw best fitness line (green)
    for j in range(1, len(history_best)):
        pygame.draw.line(surface, GREEN,
                         to_screen(j - 1, history_best[j - 1]),
                         to_screen(j, history_best[j]), 2)

    # Labels
    latest_best = history_best[-1]
    latest_avg  = history_avg[-1] if history_avg else 0
    gen         = len(history_best)

    stats = [
        f"Gen      : {gen}",
        f"Best fit : {latest_best:.1f}",
        f"Avg fit  : {latest_avg:.1f}",
        f"",
        f"— Best fitness",
        f"— Avg fitness",
    ]

    for k, line in enumerate(stats):
        color = WHITE
        if line.startswith("—") and "Best" in line:
            color = GREEN
        elif line.startswith("—") and "Avg" in line:
            color = (255, 220, 50)
        text = font.render(line, True, color)
        surface.blit(text, (graph_x, graph_y + graph_h + 15 + k * 20))

def eval_genomes(genomes, config, screen, clock, font, dev_view, all_time_best, history_best, history_avg):
    snakes    = []
    nets      = []
    ge        = []
    foods     = []

    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        snakes.append(Snake())
        nets.append(net)
        ge.append(genome)
        foods.append(spawn_food([]))

    generation_best = 0

    while True:
        clock.tick(FPS_TRAINING[0])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_d:
                    dev_view[0] = not dev_view[0]
                if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    FPS_TRAINING[0] = min(FPS_TRAINING[0] + 10, 300)
                    print(f"FPS: {FPS_TRAINING[0]}")
                if event.key == pygame.K_MINUS:
                    FPS_TRAINING[0] = max(FPS_TRAINING[0] - 10, 5)
                    print(f"FPS: {FPS_TRAINING[0]}")

        for i, snake in enumerate(snakes):
            if not snake.alive:
                continue

            # Longer snakes get more steps since navigation is harder
            allowed_steps = MAX_STEPS + (snake.score * 15)
            if snake.steps_since_food > allowed_steps:
                snake.alive = False
                continue

            # Get inputs and move
            inputs = snake.get_inputs(foods[i])
            output = nets[i].activate(inputs)
            move   = [UP, DOWN, LEFT, RIGHT][output.index(max(output))]
            snake.set_direction(move)
            snake.move()

            # Punish death
            if not snake.alive:
                ge[i].fitness -= 100
                continue

            # Big reward for eating
            if snake.head == foods[i]:
                snake.grow()
                foods[i] = spawn_food(snake.body)
                ge[i].fitness += 200

            if snake.score > generation_best:
                generation_best = snake.score
            if snake.score > all_time_best[0]:
                all_time_best[0] = snake.score

        alive = [s for s in snakes if s.alive]
        if not alive:
            break

        screen.fill(GRAY)
        for gx in range(0, WINDOW_W, GRID_SIZE):
            pygame.draw.line(screen, LIGHT_GRAY, (gx, 0), (gx, WINDOW_H))
        for gy in range(0, WINDOW_H, GRID_SIZE):
            pygame.draw.line(screen, LIGHT_GRAY, (0, gy), (WINDOW_W, gy))

        if dev_view[0]:
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
            for snake in alive:
                snake.draw(screen)
            draw_food(screen, foods[0])
            hud = font.render(f"Gen: {getattr(eval_genomes, 'generation', 0)}  Alive: {len(alive)}  Best Ever: {all_time_best[0]}  This Gen: {generation_best}", True, WHITE)
            screen.blit(hud, (8, 8))

        draw_graph(screen, font, history_best, history_avg)
        pygame.display.flip()


def main():
    pygame.init()
    PANEL_W = 300
    screen = pygame.display.set_mode((WINDOW_W + PANEL_W, WINDOW_H))    
    pygame.display.set_caption("Snake AI — NEAT Training")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 16)
    all_time_best = [0]
    history_best = []
    history_avg  = []

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
    # Create or restore population
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("neat-checkpoint-")]
    if checkpoints:
        latest = max(checkpoints, key=lambda f: int(f.split("-")[-1]))
        print(f"Restoring from checkpoint: {latest}")
        population = neat.Checkpointer.restore_checkpoint(os.path.join(checkpoint_dir, latest))
    else:
        print("No checkpoint found, starting fresh...")
        population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(neat.Checkpointer(5, filename_prefix=f"{checkpoint_dir}/neat-checkpoint-"))

    # Track generation number for sidebar
    def run_generation(genomes, config):
        eval_genomes.generation = population.generation
        eval_genomes(genomes, config, screen, clock, font, dev_view, all_time_best, history_best, history_avg)

        best = max(population.population.values(), key=lambda g: g.fitness if g.fitness is not None else 0)
        fitnesses = [g.fitness for g in population.population.values() if g.fitness is not None]
        history_best.append(best.fitness)
        history_avg.append(sum(fitnesses) / len(fitnesses))

        with open("best_genome.pkl", "wb") as f:
            pickle.dump(best, f)
        print(f"  Saved best genome (fitness: {best.fitness:.1f})")

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