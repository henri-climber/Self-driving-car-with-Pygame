import pygame
import neat
import os
import math
import pickle

pygame.init()

WIDTH, HEIGHT = 700, 900

BG = pygame.image.load("Images/bg_better.png")
CAR_IMG = pygame.image.load("../Real Car/self_driving-car/Images/car.png")
CAR_IMG = pygame.transform.scale(CAR_IMG, (210, 200))

# lines
LINE_HOR = pygame.image.load("../Real Car/self_driving-car/Images/line.png")
LINE_VERT = pygame.image.load("../Real Car/self_driving-car/Images/line_vert.png")


class Collision:

    def init(self):
        pass

    def collide(self, x, y, x2, y2, mask, mask2):
        x_offset = x2 - x
        y_offset = y2 - y
        return mask.overlap(mask2, (x_offset, y_offset))

    def get_background_mask(self):
        return pygame.mask.from_surface(BG)

    def get_player_mask(self, img):
        return pygame.mask.from_surface(img)


class Background:
    def __init__(self, win):
        self.win = win
        self.y = -9101

    def draw(self, img):
        self.win.blit(img, (0, self.y))


class Car:
    def __init__(self, x, y, img, win, line_vert, line_hor):
        self.x = x
        self.y = y
        self.center = [self.x, self.y]
        self.img = img
        self.win = win
        self.line_vert = line_vert
        self.line_hor = line_hor

        self.vel = 6
        self.vel_ang = 1
        self.tilt = 0
        self.rotated_image = img

        self.x_col = x
        self.y_col = y

        self.radars = []

    def move_left(self):
        if self.tilt + self.vel_ang > 90:
            return
        self.tilt += self.vel_ang

    def move_right(self):
        if self.tilt - self.vel_ang < -90:
            return
        self.tilt -= self.vel_ang

    def move_up(self):
        x = self.tilt * 100 / 90 / 100

        self.x -= self.vel * x

        if x < 0:
            x *= -1

        x = 1 - x
        return self.vel * x

    def check_radar(self, degree, map):
        self.center = [self.x + 105, self.y + 100]
        len = 1
        x = int(self.center[0] + math.cos(math.radians(360 - (self.tilt + degree + 90))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.tilt + degree + 90))) * len)
        try:
            while not map.get_at((x, y)) == (90, 90, 90, 255) and len < 500:

                len += 1
                x = int(self.center[0] + math.cos(math.radians(360 - (self.tilt + degree + 90))) * len)
                y = int(self.center[1] + math.sin(math.radians(360 - (self.tilt + degree + 90))) * len)
                if x < 0:
                    x *= -1
                if y < 0:
                    y *= -1

            dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
            self.radars.append([(x, y), dist])

        except Exception as e:

            self.radars.append([(0, 0), 2000])

    def update_radar(self, win):
        self.radars.clear()
        for d in range(-90, 120, 45):
            self.check_radar(d, win)

    def draw_radar(self, screen):
        for r in self.radars:
            pos, dist = r
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    def get_data(self):
        radars = self.radars
        ret = [0, 0, 0, 0, 0]
        vis = []
        for i in radars:
            vis.append(i[1] / 10)
        print(vis)

        for i, r in enumerate(radars):
            ret[i] = int(r[1] / 10)

        return ret

    def draw(self):
        # update lines
        """xlleft = self.x - 35
        xltop = self.x + 100
        xlright = self.x + 145

        ylleft = self.y + 100
        yltop = self.y - 280
        ylright = self.y + 100"""

        self.rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = self.rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        self.x_col, self.y_col = new_rect.topleft

        self.win.blit(self.rotated_image, (self.x_col, self.y_col))
        # self.win.blit(self.img, (self.x, self.y))
        pygame.display.update()



def redraw_game_window(win, bg, bg_class, car):
    win.fill((250, 250, 250))
    bg_class.draw(bg)

    car.draw()
    car.draw_radar(win)

    pygame.display.update()


GEN = 0
MOVES = 80


def main(net):
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("car")

    clock = pygame.time.Clock()

    run = True

    # neat variables
    global GEN
    global MOVES
    GEN += 1

    car = Car(150, 550, CAR_IMG, WIN, LINE_VERT, LINE_HOR)
    background = Background(WIN)

    while run:

        clock.tick(0)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        car.update_radar(WIN)

        output = net.activate(car.get_data())

        background.y += car.move_up()
        if output[0] > 0.5:
            car.move_left()
        elif output[1] > 0.5:
            car.move_right()

        if background.y > 650:
            break

        if Collision().collide(int(car.x_col), int(car.y_col), 0, int(background.y),
                               Collision().get_player_mask(car.rotated_image),
                               Collision().get_background_mask()):
            break

        redraw_game_window(WIN, BG, background, car)

    pygame.quit()


def run_main(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                                neat.DefaultStagnation, config_path)

    # Save the winner.
    with open('trained_car_model1', 'rb') as f:
        genomes = pickle.load(f)
        f.close()

    net = neat.nn.FeedForwardNetwork.create(genomes, config)
    main(net)


local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "config-feedforward")
run_main(config_path)
