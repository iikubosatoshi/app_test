from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import streamlit as st

# -----------------------------------------------------------------------------
# Streamlit page config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Othello (Reversi) - Streamlit", page_icon="♟️", layout="centered")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
EMPTY = 0
BLACK = 1   # Black goes first
WHITE = -1
DIRS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),          (0, 1),
    (1, -1),  (1, 0), (1, 1),
]

# Positional weights (classic heuristic)
WEIGHTS = [
    [120, -20,  20,  5,  5, 20, -20, 120],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [ 20,  -5, 15,  3,  3, 15,  -5,  20],
    [  5,  -5,  3,  3,  3,  3,  -5,   5],
    [  5,  -5,  3,  3,  3,  3,  -5,   5],
    [ 20,  -5, 15,  3,  3, 15,  -5,  20],
    [-20, -40, -5, -5, -5, -5, -40, -20],
    [120, -20, 20,  5,  5, 20, -20, 120],
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
Board = List[List[int]]
Move = Tuple[int, int]


def new_board() -> Board:
    b = [[EMPTY for _ in range(8)] for _ in range(8)]
    b[3][3] = WHITE
    b[3][4] = BLACK
    b[4][3] = BLACK
    b[4][4] = WHITE
    return b


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


def opponent(p: int) -> int:
    return -p


def find_flips(board: Board, r: int, c: int, p: int) -> List[Move]:
    """Return list of discs to flip if (r,c) is played by p; empty if invalid."""
    if board[r][c] != EMPTY:
        return []
    flips: List[Move] = []
    for dr, dc in DIRS:
        path: List[Move] = []
        rr, cc = r + dr, c + dc
        while in_bounds(rr, cc) and board[rr][cc] == opponent(p):
            path.append((rr, cc))
            rr += dr
            cc += dc
        if in_bounds(rr, cc) and board[rr][cc] == p and path:
            flips.extend(path)
    return flips


def valid_moves(board: Board, p: int) -> List[Move]:
    moves: List[Move] = []
    for r in range(8):
        for c in range(8):
            if find_flips(board, r, c, p):
                moves.append((r, c))
    return moves


def apply_move(board: Board, move: Move, p: int) -> Board:
    r, c = move
    flips = find_flips(board, r, c, p)
    if not flips:
        return board
    nb = [row[:] for row in board]
    nb[r][c] = p
    for rr, cc in flips:
        nb[rr][cc] = p
    return nb


def score(board: Board) -> Tuple[int, int]:
    b = sum(cell == BLACK for row in board for cell in row)
    w = sum(cell == WHITE for row in board for cell in row)
    return b, w


def game_over(board: Board) -> bool:
    return not valid_moves(board, BLACK) and not valid_moves(board, WHITE)


# -----------------------------------------------------------------------------
# Simple AI
# -----------------------------------------------------------------------------

def evaluate_move(board: Board, move: Move, p: int) -> int:
    # combine flips count and positional weight
    r, c = move
    flips = len(find_flips(board, r, c, p))
    posw = WEIGHTS[r][c]
    return flips * 10 + posw


def ai_pick_move(board: Board, p: int, level: str = "Normal") -> Optional[Move]:
    moves = valid_moves(board, p)
    if not moves:
        return None
    if level == "Easy":
        return random.choice(moves)
    # Normal/Hard: pick by heuristic; Hard gives extra bias to safe corners/edges
    scored = []
    for m in moves:
        val = evaluate_move(board, m, p)
        if level == "Hard":
            r, c = m
            # Encourage corners a lot, avoid X-squares next to corners
            if (r, c) in [(0, 0), (0, 7), (7, 0), (7, 7)]:
                val += 200
            if (r, c) in [(1, 1), (1, 6), (6, 1), (6, 6)]:
                val -= 50
        scored.append((val, m))
    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[0][1]


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
if "board" not in st.session_state:
    st.session_state.board = new_board()
    st.session_state.player = BLACK
    st.session_state.history: List[Tuple[Board, int]] = []  # (board, player)
    st.session_state.human_color = BLACK  # by default human plays black
    st.session_state.mode = "Human vs CPU"
    st.session_state.level = "Normal"


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.title("Othello / Reversi")
st.sidebar.caption("Streamlit example game. Black moves first.")

st.session_state.mode = st.sidebar.selectbox("対戦モード", ["Human vs CPU", "Human vs Human"], index=0)
if st.session_state.mode == "Human vs CPU":
    st.session_state.human_color = BLACK if st.sidebar.radio("あなたの色", ["Black", "White"], index=0) == "Black" else WHITE
    st.session_state.level = st.sidebar.select_slider("CPUレベル", options=["Easy", "Normal", "Hard"], value=st.session_state.level)

col_a, col_b, col_c = st.sidebar.columns(3)
if col_a.button("⟲ Undo"):
    if st.session_state.history:
        st.session_state.board, st.session_state.player = st.session_state.history.pop()

if col_b.button("⟲ 2手戻す"):
    for _ in range(2):
        if st.session_state.history:
            st.session_state.board, st.session_state.player = st.session_state.history.pop()

if col_c.button("🆕 New Game"):
    st.session_state.board = new_board()
    st.session_state.player = BLACK
    st.session_state.history.clear()

show_valid = st.sidebar.checkbox("合法手のヒントを表示", value=True)

# -----------------------------------------------------------------------------
# Header & status
# -----------------------------------------------------------------------------
st.title("♟️ Othello (Reversi)")

b_cnt, w_cnt = score(st.session_state.board)
turn_str = "Black" if st.session_state.player == BLACK else "White"
st.write(f"**Turn: {turn_str}**  |  ⚫ {b_cnt} - ⚪ {w_cnt}")

# -----------------------------------------------------------------------------
# Board rendering
# -----------------------------------------------------------------------------

def piece_emoji(cell: int) -> str:
    if cell == BLACK:
        return "⚫"
    if cell == WHITE:
        return "⚪"
    return "🟩"  # empty square


valid = set(valid_moves(st.session_state.board, st.session_state.player))


def make_move(r: int, c: int):
    board = st.session_state.board
    p = st.session_state.player
    flips = find_flips(board, r, c, p)
    if not flips:
        return
    # push history
    st.session_state.history.append(([row[:] for row in board], p))
    st.session_state.board = apply_move(board, (r, c), p)
    st.session_state.player = opponent(p)


# Grid (8x8)
for r in range(8):
    cols = st.columns(8, gap="small")
    for c in range(8):
        label = piece_emoji(st.session_state.board[r][c])
        if show_valid and (r, c) in valid and st.session_state.board[r][c] == EMPTY:
            label = "🟢"  # hint for valid move
        # Each square is a button; clicking attempts a move for current player
        if cols[c].button(label, key=f"sq_{r}_{c}"):
            make_move(r, c)

# -----------------------------------------------------------------------------
# Turn management & CPU move
# -----------------------------------------------------------------------------
cur = st.session_state.player
if game_over(st.session_state.board):
    b_cnt, w_cnt = score(st.session_state.board)
    if b_cnt > w_cnt:
        st.success(f"Game Over! Winner: Black  ({b_cnt} - {w_cnt})")
    elif w_cnt > b_cnt:
        st.success(f"Game Over! Winner: White  ({w_cnt} - {b_cnt})")
    else:
        st.info(f"Game Over! Draw  ({b_cnt} - {w_cnt})")
else:
    # If no legal moves, pass
    if not valid_moves(st.session_state.board, cur):
        st.info(f"{ 'Black' if cur == BLACK else 'White' } has no legal moves. Pass.")
        st.session_state.player = opponent(cur)

# CPU turn (after handling pass)
cur = st.session_state.player
if (
    st.session_state.mode == "Human vs CPU"
    and not game_over(st.session_state.board)
    and ((st.session_state.human_color == BLACK and cur == WHITE) or (st.session_state.human_color == WHITE and cur == BLACK))
):
    moves = valid_moves(st.session_state.board, cur)
    if moves:
        mv = ai_pick_move(st.session_state.board, cur, level=st.session_state.level)
        if mv is not None:
            # save history
            st.session_state.history.append(([row[:] for row in st.session_state.board], cur))
            st.session_state.board = apply_move(st.session_state.board, mv, cur)
            st.session_state.player = opponent(cur)
            st.rerun()  # immediately reflect CPU move

# Footer
st.caption("Built with Streamlit. Othello rules: capture lines by enclosing opponent discs.")
