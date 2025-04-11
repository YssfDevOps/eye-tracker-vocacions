import pygame

class Button():
    def __init__(self, x, y, image, scale):
        width = image.get_width()
        height = image.get_height()
        self.image = pygame.transform.scale(image, (int(width * scale), int(height * scale)))
        self.rect = self.image.get_rect(center=(x, y))  # CENTRAR en (x, y)
        self.clicked = False

    def draw(self, surface):
        action = False
        pos = pygame.mouse.get_pos()

        if self.rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1 and not self.clicked:
                self.clicked = True
                action = True

        if pygame.mouse.get_pressed()[0] == 0:
            self.clicked = False

        surface.blit(self.image, self.rect)
        return action

pygame.init()

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Main Menu")

font = pygame.font.SysFont("arialblack", 40)
TEXT_COL = (255, 255, 255)

# Cargar imágenes con transparencia
resume_img = pygame.image.load("src/menu/calibration_mode.png").convert_alpha()
options_img = pygame.image.load("src/menu/collection_mode.png").convert_alpha()
quit_img = pygame.image.load("src/menu/tracking_mode.png").convert_alpha()

# Escala aumentada para que se vean más grandes
scale = 1.5

# Centrar los botones en pantalla
resume_button = Button(SCREEN_WIDTH // 2, 150, resume_img, scale)
options_button = Button(SCREEN_WIDTH // 2, 275, options_img, scale)
quit_button = Button(SCREEN_WIDTH // 2, 400, quit_img, scale)

def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))

game_paused = False
menu_state = "main"
run = True

while run:
    screen.fill((52, 78, 91))

    if game_paused:
        if menu_state == "main":
            if resume_button.draw(screen):
                game_paused = False
            if options_button.draw(screen):
                menu_state = "options"
            if quit_button.draw(screen):
                run = False
    else:
        draw_text("Press SPACE to pause", font, TEXT_COL, 160, 250)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                game_paused = True

    pygame.display.update()

pygame.quit()
