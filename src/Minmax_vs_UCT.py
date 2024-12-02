import pygame
import chess
from time import sleep
from Board import GUI_Board
from encoder_decoder import decode_action
from alpha_net import ChessNet
import torch.multiprocessing as mp
from chess_board import board as c_board
import pickle
import torch
import os
from MCTS_chess import UCT_search, do_decode_n_move_pieces
import Bot


def draw(display):
    display.fill('white')
    board.draw(display)
    pygame.display.update()

WINDOW_SIZE = (600, 600)

if __name__=='__main__':
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    board = GUI_Board(*WINDOW_SIZE)
    current_board = c_board()
    Bot.initialize_openings()
    sequence = []
    opening = True

    net_to_play="current_net_trained_iter3.pth.tar"
    mp.set_start_method("spawn",force=True)
    net = ChessNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    net.share_memory()
    net.eval()
    print("hi")
    #torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/",\
    #                                "current_net.pth.tar"))
    
    current_net_filename = os.path.join("./model_data/",\
                                    net_to_play)
    checkpoint = torch.load(current_net_filename)
    net.load_state_dict(checkpoint['state_dict'])

    running = True
    while running:
        if board.turn==chess.BLACK:
            best_move,_ = UCT_search(current_board, 100, net)
            current_board = do_decode_n_move_pieces(current_board, best_move)
            ipos, fpos, prom = decode_action(current_board, best_move)
            ipos = (lambda x,y: (y.item(), x.item()))(*ipos[0])
            fpos = (lambda x,y: (y.item(), x.item()))(*fpos[0])
            print(ipos, fpos)
            piece = board.get_piece_from_pos(ipos)
            dst = board.get_square_from_pos(fpos)
            piece.move(dst)
            if current_board.check_status() == True and current_board.in_check_possible_moves() == []: # checkmate
                running = False
                break
        else:
            move_made = None
            if opening and len(sequence) < 6:
                move_made = Bot.opening_search(board, sequence)
            
            if move_made is None:
                move_made = Bot.minimax_search(board, 5, board.turn == chess.BLACK)
                opening = False
            status = (move_made is not None)
            if opening:
                sequence.append(move_made)

            if not status:
                print(board.is_end_game())
                sleep(5)
                break
            
        if board.is_end_game() is not False:
            print(board.is_end_game())  
            sleep(5)
            break
        draw(screen)
        sleep(0.01)