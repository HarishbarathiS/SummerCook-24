import sys
import pygame
import time


pygame.init()

WINDOW_SIZE = (400,400)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Tic Tac Toe")


WHITE = (255,255,255)
GRAY = (64,64,64)
RED = (255,0,0)
GREEN = (0,255,0)
BLACK = (0,0,0)

WIDTH = 400
HEIGHT = 400
LINE_WIDTH = 5
BOARD_ROWS = 3
BOARD_COLS = 3
SQUARE_SIZE = WIDTH // BOARD_COLS
CIRCLE_RADIUS = SQUARE_SIZE // BOARD_ROWS
CIRCLE_WIDTH = 10
CROSS_WIDTH = 15

board = [[0 for i in range(3)]for j in range(3)]


def mark_cell(row,col,player):
        board[row][col] = player

def is_cell_available(row,col):
    return board[row][col] == 0

def is_board_full(check_board=board):
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if check_board[row][col] == 0:
                return False
    return True

def check_win(player,check_board=board):
    # check cols
    for col in range(BOARD_COLS):
        if check_board[0][col] == player and check_board[1][col] == player and check_board[2][col] == player:
            return 1
    # check rows
    for row in range(BOARD_ROWS):
        if check_board[row][0] == player and check_board[row][1] == player and check_board[row][2] == player:
            return 2
    # check negative slope 
    if check_board[0][0] == player and check_board[1][1] == player and check_board[2][2] == player:
            return 3
    # check positive slope
    if check_board[2][0] == player and check_board[1][1] == player and check_board[0][2] == player:
            return 4
    
    return False

def Minimax(minimax_board, depth,is_maximizing,alpha,beta):
     if check_win(2,minimax_board):
          return 1
     elif check_win(1,minimax_board):
          return -1
     elif is_board_full(minimax_board):
          return 0
     
     if is_maximizing:
        # AI player
        best_reward = float('-inf')
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if minimax_board[row][col] == 0:
                        minimax_board[row][col] = 2
                        reward = Minimax(minimax_board,depth + 1,False,alpha,beta)
                        minimax_board[row][col] = 0
                        best_reward = max(best_reward,reward)
                        alpha = max(alpha,best_reward)
                        # Alpha beta pruning
                        if beta <= alpha:
                            break;
        return best_reward
        
     else:
        # Human player
        best_reward = float('inf')
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if minimax_board[row][col] == 0:
                        minimax_board[row][col] = 1
                        reward = Minimax(minimax_board,depth + 1,True,alpha,beta)
                        minimax_board[row][col] = 0
                        best_reward = min(best_reward,reward)
                        beta = min(beta,best_reward)
                        # Alpha beta pruning
                        if beta <= alpha:
                            break;
        return best_reward
     
def best_move():
    best_reward = float('-inf')
    move = (-1,-1)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
             if board[row][col] == 0:
                  board[row][col] = 2
                  reward = Minimax(board,0,False,float('-inf'),float('inf'))
                  board[row][col] = 0
                  if reward > best_reward:
                       best_reward = reward
                       move = (row,col)

    if move != (-1,-1):
        mark_cell(move[0],move[1],2)
        return True
    return False

def draw_o(color,row,col):
    pygame.draw.circle(screen, color,(col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), CIRCLE_RADIUS, CIRCLE_WIDTH)


def draw_x(color,row,col):
    pygame.draw.line(screen, color,(col * SQUARE_SIZE + SQUARE_SIZE // 4, row * SQUARE_SIZE + SQUARE_SIZE // 4), (col * SQUARE_SIZE + 3 * SQUARE_SIZE // 4, row * SQUARE_SIZE + 3 * SQUARE_SIZE // 4), CROSS_WIDTH)
    pygame.draw.line(screen, color,(col * SQUARE_SIZE + SQUARE_SIZE // 4, row * SQUARE_SIZE + 3 * SQUARE_SIZE // 4), (col * SQUARE_SIZE + 3 * SQUARE_SIZE // 4, row * SQUARE_SIZE + SQUARE_SIZE // 4), CROSS_WIDTH)




def draw_figures(color=WHITE,trio=0,player=0):
    if player == 2:
        color = RED
    elif player == 1:
        color = GREEN
    if trio == 1:
        for col in range(BOARD_COLS):
            if board[0][col] == player and board[1][col] == player and board[2][col] == player and player == 1:
                draw_x(color,0,col)
                draw_x(color,1,col)
                draw_x(color,2,col)
            elif board[0][col] == player and board[1][col] == player and board[2][col] == player and player == 2:
                draw_o(color,0,col)
                draw_o(color,1,col)
                draw_o(color,2,col)
    if trio == 2:
        for row in range(BOARD_ROWS):
            if board[row][0] == player and board[row][1] == player and board[row][2] == player and player == 1:
                draw_x(color,row,0)
                draw_x(color,row,1)
                draw_x(color,row,2)
            elif board[row][0] == player and board[row][1] == player and board[row][2] == player and player == 2:
                draw_o(color,row,0)
                draw_o(color,row,1)
                draw_o(color,row,2)
    if trio == 3:
        if board[0][0] == player and board[1][1] == player and board[2][2] == player and player == 1:
            draw_x(color,0,0)
            draw_x(color,1,1)
            draw_x(color,2,2)
        elif board[0][0] == player and board[1][1] == player and board[2][2] == player and player == 2:
            draw_o(color,0,0)
            draw_o(color,1,1)
            draw_o(color,2,2)
    if trio == 4:
        if board[2][0] == player and board[1][1] == player and board[0][2] == player and player == 1:
            draw_x(color,2,0)
            draw_x(color,1,1)
            draw_x(color,0,2)
        elif board[2][0] == player and board[1][1] == player and board[0][2] == player and player == 2:
            draw_o(color,2,0)
            draw_o(color,1,1)
            draw_o(color,0,2)

    color = WHITE

    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            if board[row][col] == 2:
                if player != 2:
                    draw_o(color,row,col)
            elif board[row][col] == 1:
                if player != 1:
                    draw_x(color,row,col)
               

def draw_board(color=WHITE):
    for i in range(1,BOARD_ROWS):
        #horizontal line
        pygame.draw.line(screen, color,(0, SQUARE_SIZE * i),(WIDTH,SQUARE_SIZE * i),LINE_WIDTH)
        #vertical line
        pygame.draw.line(screen, color,(SQUARE_SIZE * i, 0),(SQUARE_SIZE * i, HEIGHT),LINE_WIDTH)



def restart_game():
    screen.fill(BLACK)
    draw_board()
    board = [[0 for i in range(3)] for j in range(3)]

draw_board()

player = 1
game_over = False


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
           sys.exit()

        elif pygame.mouse.get_pressed()[0]:
            mouse_pos = pygame.mouse.get_pos()
            print(mouse_pos)
            x = mouse_pos[0] // SQUARE_SIZE
            y = mouse_pos[1] // SQUARE_SIZE
            
            if is_cell_available(y,x):
                mark_cell(y,x,player)
                if check_win(player):
                   game_over = True
                player = player % 2 + 1

                if not game_over:
                    if best_move():
                        if check_win(2):
                            game_over = True
                        player = player % 2 + 1

                if not game_over:
                    if is_board_full():
                        game_over = True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                restart_game()
                player = 1
                game_over = False
        
    if not game_over:
        draw_figures()
    else:
        trio = check_win(1)
        if trio:
            draw_board()
            draw_figures(trio=trio,player=1)
        else:
            trio = check_win(2)
            if trio:
                draw_board()
                draw_figures(trio=trio,player=2)
            else:
                draw_board(GRAY)
                draw_figures(GRAY)

    pygame.display.update()