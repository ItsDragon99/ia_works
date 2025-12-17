import pygame

pygame.init()

ANCHO_VENTANA = 800
VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ANCHO_VENTANA))
pygame.display.set_caption("Visualización de Nodos · A* (LA/LC)")

THEMES = {
    "claro": {
        "default": (255, 255, 255),
        "wall": (30, 30, 30),
        "grid": (200, 200, 200),
        "open": (102, 204, 0),
        "closed": (0, 102, 204),
        "start": (255, 140, 0),
        "end": (154, 0, 154),
        "path": (30, 144, 255),
        "background": (245, 245, 245),
        "text": (25, 25, 25),
        "panel_bg": (255, 255, 255, 215)
    },
    "oscuro": {
        "default": (40, 40, 40),
        "wall": (95, 95, 95),
        "grid": (70, 70, 70),
        "open": (0, 180, 115),
        "closed": (0, 120, 210),
        "start": (255, 140, 0),
        "end": (170, 0, 170),
        "path": (0, 153, 255),
        "background": (25, 25, 25),
        "text": (230, 230, 230),
        "panel_bg": (50, 50, 50, 180)
    },
    "pastel": {
        "default": (250, 248, 239),
        "wall": (120, 110, 100),
        "grid": (215, 210, 205),
        "open": (170, 217, 166),
        "closed": (160, 190, 240),
        "start": (255, 201, 120),
        "end": (198, 143, 198),
        "path": (143, 190, 255),
        "background": (250, 248, 239),
        "text": (90, 80, 70),
        "panel_bg": (255, 255, 255, 230)
    }
}
active_theme_names = list(THEMES.keys())
active_theme_index = 0
def theme():
    return THEMES[active_theme_names[active_theme_index]]

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = col * ancho
        self.y = fila * ancho
        self.ancho = ancho
        self.total_filas = total_filas
        self.estado = 'default'
        self.color = theme()['default']

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.estado == 'wall'

    def es_inicio(self):
        return self.estado == 'start'

    def es_fin(self):
        return self.estado == 'end'

    def restablecer(self):
        self.estado = 'default'
        self.color = theme()['default']

    def hacer_inicio(self):
        self.estado = 'start'
        self.color = theme()['start']

    def hacer_pared(self):
        self.estado = 'wall'
        self.color = theme()['wall']

    def hacer_fin(self):
        self.estado = 'end'
        self.color = theme()['end']
    
    def hacer_la(self):
        self.estado = 'open'
        self.color = theme()['open']
        
    def hacer_lc(self):
        self.estado = 'closed'
        self.color = theme()['closed']
        
    def hacer_camino(self):
        self.estado = 'path'
        self.color = theme()['path']

    def dibujar(self, ventana):
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho))
        if self.estado == 'path':
            pygame.draw.rect(ventana, theme()['grid'], (self.x, self.y, self.ancho, self.ancho), 1)

class NodoAStar(Nodo):
    def __init__(self, fila, col, ancho, total_filas, padre=None, distancia=None, h=None):
        super().__init__(fila, col, ancho, total_filas)
        if distancia is not None and h is not None:
            self.distancia = distancia
            self.h = h
            self.total = distancia + h
            self.padre = padre
        else:
            self.distancia = float("inf")
            self.h = float("inf")
            self.total = float("inf")
            self.padre = None
        self.vecinos = []

    def actualizar_vecinos(self, grid, fin):
        self.vecinos = []
        if self.fila < self.total_filas - 1 and not grid[self.fila + 1][self.col].es_pared():
            self.vecinos.append(NodoAStar(grid[self.fila + 1][self.col].fila, 
                                          grid[self.fila + 1][self.col].col, 
                                          self.ancho, self.total_filas, 
                                          self, 
                                          self.distancia + 10, 
                                          abs(fin.fila - (self.fila + 1)) * 10 + abs(fin.col - self.col) * 10))
        if self.fila > 0 and not grid[self.fila - 1][self.col].es_pared():
            self.vecinos.append(NodoAStar(grid[self.fila - 1][self.col].fila, 
                                          grid[self.fila - 1][self.col].col, 
                                          self.ancho, self.total_filas, self, 
                                          self.distancia + 10, 
                                          abs(fin.fila - (self.fila - 1)) * 10 + abs(fin.col - self.col) * 10))
        if self.col < self.total_filas - 1 and not grid[self.fila][self.col + 1].es_pared():
            self.vecinos.append(NodoAStar(grid[self.fila][self.col + 1].fila, 
                                          grid[self.fila][self.col + 1].col, 
                                          self.ancho, 
                                          self.total_filas, 
                                          self, 
                                          self.distancia + 10, 
                                          abs(fin.fila - self.fila) * 10 + abs(fin.col - (self.col + 1)) * 10))
        if self.col > 0 and not grid[self.fila][self.col - 1].es_pared():
            self.vecinos.append(NodoAStar(grid[self.fila][self.col - 1].fila, 
                                          grid[self.fila][self.col - 1].col, 
                                          self.ancho, 
                                          self.total_filas, 
                                          self, 
                                          self.distancia + 10, 
                                          abs(fin.fila - self.fila) * 10 + abs(fin.col - (self.col - 1)) * 10))
        if self.fila < self.total_filas - 1 and self.col < self.total_filas - 1 and not grid[self.fila + 1][self.col + 1].es_pared():
            self.vecinos.append(NodoAStar(grid[self.fila + 1][self.col + 1].fila, 
                                          grid[self.fila + 1][self.col + 1].col, 
                                          self.ancho, 
                                          self.total_filas, 
                                          self, 
                                          self.distancia + 15, 
                                          abs(fin.fila - (self.fila + 1)) * 10 + abs(fin.col - (self.col + 1)) * 10))
        if self.fila < self.total_filas - 1 and self.col > 0 and not grid[self.fila + 1][self.col - 1].es_pared():
            self.vecinos.append(NodoAStar(grid[self.fila + 1][self.col - 1].fila, 
                                          grid[self.fila + 1][self.col - 1].col, 
                                          self.ancho, 
                                          self.total_filas, 
                                          self, 
                                          self.distancia + 15, 
                                          abs(fin.fila - (self.fila + 1)) * 10 + abs(fin.col - (self.col - 1)) * 10))
        if self.fila > 0 and self.col < self.total_filas - 1 and not grid[self.fila - 1][self.col + 1].es_pared():
            self.vecinos.append(NodoAStar(grid[self.fila - 1][self.col + 1].fila, 
                                          grid[self.fila - 1][self.col + 1].col, 
                                          self.ancho, 
                                          self.total_filas, 
                                          self, 
                                          self.distancia + 15, 
                                          abs(fin.fila - (self.fila - 1)) * 10 + abs(fin.col - (self.col + 1)) * 10))
        if self.fila > 0 and self.col > 0 and not grid[self.fila - 1][self.col - 1].es_pared():
            self.vecinos.append(NodoAStar(grid[self.fila - 1][self.col - 1].fila, 
                                          grid[self.fila - 1][self.col - 1].col, 
                                          self.ancho, 
                                          self.total_filas, 
                                          self, 
                                          self.distancia + 15, 
                                          abs(fin.fila - (self.fila - 1)) * 10 + abs(fin.col - (self.col - 1)) * 10))

def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana, filas, ancho):
    color_grid = theme()['grid']
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, color_grid, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, color_grid, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def dibujar(ventana, grid, filas, ancho):
    ventana.fill(theme()['background'])
    for fila in grid:
        for nodo in fila:
            nodo.dibujar(ventana)
    dibujar_grid(ventana, filas, ancho)
    pygame.display.update()

def obtener_click_pos(pos, filas, ancho):
    ancho_nodo = ancho // filas
    x, y = pos
    fila = y // ancho_nodo
    col = x // ancho_nodo
    return fila, col

def algoritmo_aStar(fin, lc, la, grid, actual, inicio):
    while(actual.fila != fin.fila or actual.col != fin.col):
        actual.actualizar_vecinos(grid, fin)
        for vecino in actual.vecinos:
            if la.get((vecino.fila, vecino.col)) is None and lc.get((vecino.fila, vecino.col)) is None:
                la[(vecino.fila, vecino.col)] = vecino
                if(vecino.fila != fin.fila or vecino.col != fin.col):
                    grid[vecino.fila][vecino.col].hacer_la()
            elif la.get((vecino.fila, vecino.col)) is not None and vecino.total < la[(vecino.fila, vecino.col)].total:
                la[(vecino.fila, vecino.col)] = vecino
        minValue = min(la.values(), key=lambda x: x.total).total if la else float("inf")
        actual = None
        for key in la:
            if la[key].total == minValue:
                actual = la[key]
                break
        if actual is None:
            return
        lc[(actual.fila, actual.col)] = actual
        la.pop((actual.fila, actual.col))
        if actual.fila != fin.fila or actual.col != fin.col:
            grid[actual.fila][actual.col].hacer_lc()
      
    if(actual.fila == fin.fila and actual.col == fin.col):
        while actual.padre:
            actual = actual.padre
            if (actual.fila != fin.fila or actual.col != fin.col) and (actual.fila != inicio.fila or actual.col != inicio.col):
                grid[actual.fila][actual.col].hacer_camino()
        for key in la:
            nodo = la[key]
        for key in lc:
            nodo = lc[key]

def main(ventana, ancho):
    FILAS = 11
    grid = crear_grid(FILAS, ancho)
    inicio = None
    fin = None
    corriendo = True

    while corriendo:
        dibujar(ventana, grid, FILAS, ancho)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                if not inicio and nodo != fin:
                    inicio = nodo
                    inicio.hacer_inicio()
                elif not fin and nodo != inicio:
                    fin = nodo
                    fin.hacer_fin()
                elif nodo != fin and nodo != inicio:
                    if not nodo.es_pared():
                        nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                fila, col = obtener_click_pos(pos, FILAS, ancho)
                nodo = grid[fila][col]
                nodo.restablecer()
                if nodo == inicio:
                    inicio = None
                elif nodo == fin:
                    fin = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    if(inicio and fin):
                        actual = NodoAStar(inicio.fila, inicio.col, inicio.ancho, inicio.total_filas, None, 0, 0)
                        lc = {}
                        la = {}
                        lc[(actual.fila, actual.col)] = actual
                        algoritmo_aStar(fin, lc, la, grid, actual, inicio)
                if event.key in (pygame.K_c, pygame.K_r):
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, ancho)
                if event.key == pygame.K_t:
                    global active_theme_index
                    active_theme_index = (active_theme_index + 1) % len(active_theme_names)
                    for fila in grid:
                        for nodo in fila:
                            estado = nodo.estado
                            metodo = {
                                'default': nodo.restablecer,
                                'wall': nodo.hacer_pared,
                                'open': nodo.hacer_la,
                                'closed': nodo.hacer_lc,
                                'startclear': nodo.hacer_inicio,
                                'end': nodo.hacer_fin,
                                'path': nodo.hacer_camino
                            }[estado]
                            metodo()

    pygame.quit()

if __name__ == '__main__':
    main(VENTANA, ANCHO_VENTANA)