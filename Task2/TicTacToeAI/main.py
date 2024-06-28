# import sys
# import pygame
# import numpy as np
import time
# pygame.init()

# White = (255,255,255)
# Gray = (180,180,180)
# Red = (255,0,0)
# Green = (0,255,0)
# Black = (0,0,0)

# width = 300
# height = 300
# line_width = 5
BOARD_ROWS = 3
BOARD_COLS = 3
# square_size = width // cols
# circle_szie = square_size // rows
# circle_width = 15
# cross_width = 25

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
            return True
    # check rows
    for row in range(BOARD_ROWS):
        if check_board[row][0] == player and check_board[row][1] == player and check_board[row][2] == player:
            return True
    # check negative slope 
    if check_board[0][0] == player and check_board[1][1] == player and check_board[2][2] == player:
            return True
    # check positive slope
    if check_board[2][0] == player and check_board[1][1] == player and check_board[0][2] == player:
            return True
    
    return False

def Minimax(minimax_board, depth,is_maximizing):
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
                        reward = Minimax(minimax_board,depth + 1,False)
                        minimax_board[row][col] = 0
                        best_reward = max(best_reward,reward)
        return best_reward
        
     else:
        # Human player
        best_reward = float('inf')
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if minimax_board[row][col] == 0:
                        minimax_board[row][col] = 1
                        reward = Minimax(minimax_board,depth + 1,True)
                        minimax_board[row][col] = 0
                        best_reward = min(best_reward,reward)
        return best_reward
     
def best_move():
    best_reward = float('-inf')
    move = (-1,-1)
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
             if board[row][col] == 0:
                  board[row][col] = 2
                  reward = Minimax(board,0,False)
                  board[row][col] = 0
                  if reward > best_reward:
                       best_reward = reward
                       move = (row,col)

    if move != (-1,-1):
        mark_cell(move[0],move[1],2)
        return True
    return False

def display_board(board,player):
    if player == 1:
        print("BOARD AFTER YOUR MOVE")
    elif player == 2 :
        print("BOARD AFTER AI's MOVE")
    else : 
        print("STARING GAME")
    print("--- --- ---")  
    for row in range(BOARD_ROWS):
        print(" {} | {} | {}".format(board[row][0],board[row][1],board[row][2]))
        print("--- --- ---")


display_board(board,0)

player = 1
game_over = False

while True:
    x,y = map(int,input("Enter move (x,y) : ").split(','))
    if is_cell_available(x,y):
         mark_cell(x,y,player)
    display_board(board,player)
    if check_win(player):
        game_over = True
        print("Player 1 won the game")
        break
    player = player % 2 + 1
        
    if not game_over:
        time.sleep(2)
        if best_move():
            display_board(board,player)
            if check_win(2):
                game_over = True
                print("Player 2 won the game")
                break
            player = player % 2 + 1
    
    if not game_over:
        if is_board_full(board):
            game_over = True
            print("Game draw")
            break


    