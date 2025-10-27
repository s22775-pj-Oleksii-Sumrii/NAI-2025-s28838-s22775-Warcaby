"""
   Projekt: Autonomiczne autko

AUTORZY:
Oleksii Sumrii (s22775),
Oskar Szyszko (s28838),

INSTRUKCJA PRZYGOTOWANIA ŚRODOWISKA:
Zainstaluj Python 3.8 lub nowszy z https://python.org/,
Zainstaluj Visual Studio Code (VS Code) z https://code.visualstudio.com/,
Zainstaluj PyGame w terminale wpisz: pip install pygame (ale po instalowanym Pythonie)
Zainstaluj rozszerzenie „Python” w VS Code:
Otwórz VS Code
Kliknij ikonę rozszerzeń po lewej stronie albo Ctrl+Shift+X
Wyszukaj Python i zainstaluj rozszerzenie od Microsoft,
,
,
,
Otworz folder w którym znajduje się plik fuzzy_maze.py,
Uruchom gre:
Otwórz plik fuzzy_maze.py
Kliknij przycisk Run Python File albo Ctrl+F5
"""

import math
import pygame

WIN_W, WIN_H = 1000, 700
FPS = 60
RAY_DEGS = [-50, 0, 50]
RAY_LEN = 260
MAX_STEER_DEG = 30
BASE_SPEED = 2.5
SPEED_SCALE_ON_TURN = 0.6
TURN_LIMIT_DEG = 100
DIST_NEAR = (0, 25, 45)
DIST_MID  = (30, 60, 90)
DIST_FAR  = (70, 120, 150)
DIFF_LEFT  = (-200, -100, -20)
DIFF_ZERO  = (-40, 0, 40)
DIFF_RIGHT = (20, 100, 200)
STEER_SETS = {
    "HardL":   (-30, -30, -15),
    "SoftL":   (-20, -10, 0),
    "Straight":(-3, 0, 3),
    "SoftR":   (0, 10, 20),
    "HardR":   (15, 30, 30)
}
SPEED_SETS = {
    "Slow": (0.0, 0.0, 0.35),
    "Med":  (0.2, 0.55, 0.80),
    "Fast": (0.6, 1.0, 1.0)
}

def clamp(v, a, b): return max(a, min(b, v))
def tri_mu(x, a, b, c):
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    return (x - a) / (b - a) if x < b else (c - x) / (c - b)
def tri_clip(x, a, b, c, alpha): return min(tri_mu(x, a, b, c), alpha)
def centroid_aggregate(domain, contribs):
    num = 0.0; den = 0.0
    for x in domain:
        mu = 0.0
        for fn in contribs: mu = max(mu, fn(x))
        num += x * mu; den += mu
    return (num / den) if den > 1e-9 else 0.0
def dist_memberships(d):
    return {"Near": tri_mu(d, *DIST_NEAR), "Mid": tri_mu(d, *DIST_MID), "Far": tri_mu(d, *DIST_FAR)}
def diff_memberships(delta):
    return {"Left": tri_mu(delta, *DIFF_LEFT), "Zero": tri_mu(delta, *DIFF_ZERO), "Right": tri_mu(delta, *DIFF_RIGHT)}

MAZE = [
"######################",
"#                    #",
"#   S                #",
"#                    #",
"#   ##############   #",
"#   ##############   #",
"#   ##############   #",
"#   ###              #",
"#   ###              #",
"#   ###              #",
"#   ###   ############",
"#   ###   ############",
"#   ###   ############",
"#   ###              #",
"#   ###              #",
"#   ###              #",
"#   ##############   #",
"#   ##############   #",
"#   ##############   #",
"#                    #",
"#                    #",
"#                   E#",
"######################",
]

CELL = 28
MAZE_ROWS = len(MAZE)
MAZE_COLS = len(MAZE[0])
MAZE_W = MAZE_COLS * CELL
MAZE_H = MAZE_ROWS * CELL
OFFSET_X = (WIN_W - MAZE_W) // 2
OFFSET_Y = (WIN_H - MAZE_H) // 2

def grid_to_px(c, r):
    x = OFFSET_X + c * CELL + CELL // 2
    y = OFFSET_Y + r * CELL + CELL // 2
    return x, y
def draw_labyrinth(surface):
    surface.fill((255, 255, 255))
    black = (0, 0, 0)
    for r, row in enumerate(MAZE):
        for c, ch in enumerate(row):
            if ch == '#':
                x = OFFSET_X + c * CELL
                y = OFFSET_Y + r * CELL
                pygame.draw.rect(surface, black, (x, y, CELL, CELL))
def find_start():
    for r, row in enumerate(MAZE):
        for c, ch in enumerate(row):
            if ch == 'S': return grid_to_px(c, r)
    return WIN_W//2, WIN_H//2

def ray_distance(dist_surf, start, ang_deg, max_len=RAY_LEN):
    x0, y0 = start
    a = math.radians(ang_deg)
    dx, dy = math.cos(a), math.sin(a)
    for d in range(0, max_len, 2):
        x = int(x0 + dx * d); y = int(y0 + dy * d)
        if x < 0 or y < 0 or x >= dist_surf.get_width() or y >= dist_surf.get_height(): return d
        if dist_surf.get_at((x, y))[0] < 20: return d
    return max_len
def collide_point(dist_surf, pos):
    x, y = int(pos[0]), int(pos[1])
    if x < 0 or y < 0 or x >= dist_surf.get_width() or y >= dist_surf.get_height(): return True
    return dist_surf.get_at((x, y))[0] < 20

def fuzzy_controller(L, F, R):
    delta = L - R
    Fm = dist_memberships(F)
    Dm = diff_memberships(delta)
    steer_contribs, speed_contribs = [], []
    def add_steer(label, alpha):
        a,b,c = STEER_SETS[label]
        steer_contribs.append(lambda x, A=a,B=b,C=c,al=alpha: tri_clip(x,A,B,C,al))
    def add_speed(label, alpha):
        a,b,c = SPEED_SETS[label]
        speed_contribs.append(lambda x, A=a,B=b,C=c,al=alpha: tri_clip(x,A,B,C,al))
    add_steer("SoftL", Dm["Right"])
    add_steer("SoftR", Dm["Left"])
    add_steer("Straight", Dm["Zero"])
    add_steer("HardL", min(Fm["Near"], Dm["Right"]))
    add_steer("HardR", min(Fm["Near"], Dm["Left"]))
    add_speed("Slow", Fm["Near"])
    add_speed("Med",  Fm["Mid"])
    add_speed("Fast", Fm["Far"])
    steer_domain = [x for x in range(-MAX_STEER_DEG, MAX_STEER_DEG+1)]
    speed_domain = [i/100 for i in range(0, 101)]
    steer_deg = centroid_aggregate(steer_domain, steer_contribs)
    speed_u   = centroid_aggregate(speed_domain, speed_contribs)
    return steer_deg, speed_u

class Car:
    def __init__(self, x, y, heading_deg=0.0):
        self.x, self.y = x, y
        self.heading = heading_deg
        self.trace = []
        self.trace_on = True
        self.turn_accum = 0.0
        self.turn_dir = 0
    def sense(self, map_surf):
        dists = []
        for deg in RAY_DEGS:
            ang = self.heading + deg
            dists.append(ray_distance(map_surf, (self.x, self.y), ang, RAY_LEN))
        return dists
    def step(self, map_surf):
        L, F, R = self.sense(map_surf)
        steer_deg, speed_u = fuzzy_controller(L, F, R)
        speed = BASE_SPEED * (0.4 + 0.6*speed_u)
        speed *= (1.0 - (abs(steer_deg)/MAX_STEER_DEG)*(1.0 - SPEED_SCALE_ON_TURN))
        speed = clamp(speed, 0.2, 5.0)
        desired_delta = steer_deg * 0.15
        sgn = 1 if desired_delta > 0 else (-1 if desired_delta < 0 else 0)
        if abs(steer_deg) < 2 or sgn == 0:
            self.turn_accum = 0.0; self.turn_dir = 0; applied_delta = desired_delta
        else:
            if self.turn_dir == 0 or sgn != self.turn_dir:
                self.turn_dir = sgn; self.turn_accum = 0.0
            remaining = TURN_LIMIT_DEG - abs(self.turn_accum)
            applied_delta = 0.0 if remaining <= 0 else clamp(abs(desired_delta), 0, remaining) * sgn
            self.turn_accum += applied_delta
        self.heading += applied_delta
        a = math.radians(self.heading)
        self.x += math.cos(a) * speed
        self.y += math.sin(a) * speed
        if collide_point(map_surf, (self.x, self.y)):
            self.x -= math.cos(a) * (speed*2)
            self.y -= math.sin(a) * (speed*2)
            self.heading += 30
            self.turn_accum = 0.0
            self.turn_dir = 0
        if self.trace_on:
            if not self.trace or (abs(self.trace[-1][0]-self.x)+abs(self.trace[-1][1]-self.y))>3:
                self.trace.append((self.x, self.y))
        return (L, F, R), steer_deg, speed
    def draw(self, screen, map_surf):
        cx, cy = int(self.x), int(self.y)
        for i, deg in enumerate(RAY_DEGS):
            a = self.heading + deg
            d = ray_distance(map_surf, (self.x, self.y), a, RAY_LEN)
            x2 = int(cx + math.cos(math.radians(a))*d)
            y2 = int(cy + math.sin(math.radians(a))*d)
            color = (120,120,255) if i != 1 else (80,180,80)
            pygame.draw.line(screen, color, (cx,cy), (x2,y2), 2)
        pygame.draw.circle(screen, (30,144,255), (cx, cy), 7)
        nx = int(cx + math.cos(math.radians(self.heading))*12)
        ny = int(cy + math.sin(math.radians(self.heading))*12)
        pygame.draw.line(screen, (255,80,80), (cx,cy), (nx,ny), 3)
        if self.trace_on and len(self.trace) > 1:
            pygame.draw.lines(screen, (80,80,255), False, [(int(x),int(y)) for x,y in self.trace], 2)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Fuzzy Auto — Labirynt")
    clock = pygame.time.Clock()
    map_surf = pygame.Surface((WIN_W, WIN_H))
    draw_labyrinth(map_surf)
    start = find_start()
    car = Car(*start, heading_deg=0)
    font = pygame.font.SysFont("consolas", 20, bold=True)
    small = pygame.font.SysFont("consolas", 16)
    paused = False
    while True:
        clock.tick(FPS)
        for e in pygame.event.get():
            if e.type == pygame.QUIT: return
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE: paused = not paused
                elif e.key == pygame.K_r: car = Car(*start, heading_deg=0)
                elif e.key == pygame.K_t: car.trace_on = not car.trace_on; car.trace.clear()
        if not paused:
            dists, steer_deg, speed = car.step(map_surf)
        else:
            dists, steer_deg, speed = car.sense(map_surf), 0.0, 0.0
        screen.blit(map_surf, (0,0))
        car.draw(screen, map_surf)
        hud = pygame.Surface((WIN_W, 36), pygame.SRCALPHA)
        hud.fill((255,255,255,210))
        screen.blit(hud, (0,0))
        L, F, R = dists
        txt = font.render(f"L:{int(L)}   F:{int(F)}   R:{int(R)}   Δ={int(L-R):+}", True, (0,0,0))
        screen.blit(txt, (20,7))
        help_txt = small.render("[SPACE] pauza   [R] reset   [T] ślad", True, (0,0,0))
        screen.blit(help_txt, (WIN_W-350, 10))
        pygame.display.flip()

if __name__ == "__main__":
    main()
