from __future__ import annotations
from typing import List, Tuple, Optional
import streamlit as st
import random

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Othello (Reversi) - Streamlit", page_icon="♟️", layout="centered")

# ─────────────────────────────────────────────────────────────
# CSS（固定8×8・隙間ゼロ・直角マス）
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container{padding-left:.5rem!important;padding-right:.5rem!important}
.board-scroll{overflow-x:auto}
.board-wrap{width:max-content;margin:0 auto}
.board{display:grid;grid-template-columns:repeat(8,56px);grid-auto-rows:56px;gap:0}
.cell{width:56px;height:56px;background:#1c7c2d;border:.25px solid #004400;border-radius:0;
      box-shadow:inset 0 0 8px rgba(0,0,0,.25);display:flex;align-items:center;justify-content:center}
.piece{border-radius:50%;width:54px;height:54px}
.piece.black{background:#000;box-shadow:inset 0 0 6px rgba(255,255,255,.2)}
.piece.white{background:#fff;border:1px solid #222;box-shadow:inset 0 0 6px rgba(0,0,0,.2)}

form.mv{margin:0;padding:0}
.mvbtn{width:56px;height:56px;margin:0;padding:0;cursor:pointer;background:#1c7c2d;
       border:.25px solid #004400;border-radius:0;box-shadow:inset 0 0 8px rgba(0,0,0,.25);
       line-height:1;font-size:32px;color:yellow;font-weight:bold}
.mvbtn:focus,.mvbtn:focus-visible,.mvbtn:active{outline:none;background:#1c7c2d;border:.25px solid #004400;
       box-shadow:inset 0 0 8px rgba(0,0,0,.25)}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# 盤ロジック
# ─────────────────────────────────────────────────────────────
EMPTY, BLACK, WHITE = 0, 1, -1
DIRS = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
Board = list[list[int]]
Move  = tuple[int,int]

def new_board()->Board:
    b=[[EMPTY]*8 for _ in range(8)]
    b[3][3]=WHITE; b[3][4]=BLACK; b[4][3]=BLACK; b[4][4]=WHITE
    return b

def in_bounds(r:int,c:int)->bool: return 0<=r<8 and 0<=c<8
def opponent(p:int)->int: return -p

def find_flips(board:Board,r:int,c:int,p:int)->list[Move]:
    if board[r][c]!=EMPTY: return []
    flips=[]
    for dr,dc in DIRS:
        path=[]; rr,cc=r+dr,c+dc
        while in_bounds(rr,cc) and board[rr][cc]==opponent(p):
            path.append((rr,cc)); rr+=dr; cc+=dc
        if in_bounds(rr,cc) and board[rr][cc]==p and path: flips+=path
    return flips

def valid_moves(board:Board,p:int)->list[Move]:
    return [(r,c) for r in range(8) for c in range(8) if find_flips(board,r,c,p)]

def apply_move(board:Board, mv:Move, p:int)->Board:
    r,c=mv; flips=find_flips(board,r,c,p)
    if not flips: return board
    nb=[row[:] for row in board]; nb[r][c]=p
    for rr,cc in flips: nb[rr][cc]=p
    return nb

def score(board:Board)->tuple[int,int]:
    b=sum(cell==BLACK for row in board for cell in row)
    w=sum(cell==WHITE for row in board for cell in row)
    return b,w

def game_over(board:Board)->bool:
    return not valid_moves(board,BLACK) and not valid_moves(board,WHITE)

# ─────────────────────────────────────────────────────────────
# URLエンコード（セッション切れても継続）
# ─────────────────────────────────────────────────────────────
def encode_board(b:Board)->str:
    m={EMPTY:'.', BLACK:'B', WHITE:'W'}
    return ''.join(m[b[r][c]] for r in range(8) for c in range(8))

def decode_board(s:str)->Board:
    m={'.':EMPTY,'B':BLACK,'W':WHITE}
    out=[[EMPTY]*8 for _ in range(8)]
    if not s or len(s)!=64: return new_board()
    k=0
    for r in range(8):
        for c in range(8):
            out[r][c]=m.get(s[k],EMPTY); k+=1
    return out

def read_query():
    try:
        q=st.query_params
        s=q.get("s",None); p=q.get("p",None); mv=q.get("mv",None)
        s=s if isinstance(s,str) else (s[0] if s else None)
        p=p if isinstance(p,str) else (p[0] if p else None)
        mv=mv if isinstance(mv,str) else (mv[0] if mv else None)
        return s,p,mv
    except Exception:
        q=st.experimental_get_query_params()
        return q.get("s",[None])[0], q.get("p",[None])[0], q.get("mv",[None])[0]

def set_query(s:Optional[str]=None, p:Optional[int]=None):
    try:
        st.query_params.clear()
        params={}
        if s is not None: params["s"]=s
        if p is not None: params["p"]=str(p)
        if params: st.query_params.update(params)
    except Exception:
        if s is None and p is None: st.experimental_set_query_params()
        else:
            args={}
            if s is not None: args["s"]=s
            if p is not None: args["p"]=str(p)
            st.experimental_set_query_params(**args)

# ─────────────────────────────────────────────────────────────
# CPU 思考
# ─────────────────────────────────────────────────────────────
# Hard 用の位置評価
WEIGHTS = [
    [120,-20, 20,  5,  5, 20,-20,120],
    [-20,-40, -5, -5, -5, -5,-40,-20],
    [ 20, -5, 15,  3,  3, 15, -5, 20],
    [  5, -5,  3,  3,  3,  3, -5,  5],
    [  5, -5,  3,  3,  3,  3, -5,  5],
    [ 20, -5, 15,  3,  3, 15, -5, 20],
    [-20,-40, -5, -5, -5, -5,-40,-20],
    [120,-20, 20,  5,  5, 20,-20,120],
]

def eval_board(b:Board, p:int)->int:
    # 自石: +、相手: -
    s=0
    for r in range(8):
        for c in range(8):
            if b[r][c]==p: s+=WEIGHTS[r][c]
            elif b[r][c]==opponent(p): s-=WEIGHTS[r][c]
    return s

def cpu_pick(b:Board, p:int, level:str)->Optional[Move]:
    moves=valid_moves(b,p)
    if not moves: return None
    if level=="Easy":
        return random.choice(moves)
    if level=="Medium":
        # 返す枚数が最大の手
        best=None; best_n=-1
        for mv in moves:
            n=len(find_flips(b,mv[0],mv[1],p))
            if n>best_n: best_n=n; best=mv
        return best
    # Hard: 位置評価 + 着手後の eval を最大化
    best=None; best_v=-10**9
    for mv in moves:
        nb=apply_move(b,mv,p)
        v=eval_board(nb,p)
        # 角優先の微調整（角なら大きく加点）
        if mv in [(0,0),(0,7),(7,0),(7,7)]: v+=500
        if v>best_v: best_v=v; best=mv
    return best

def cpu_auto_play():
    """現在の設定でCPUの番なら、連続してCPUの手を適用（パス処理込み）。"""
    b=st.session_state.board
    p=st.session_state.player
    human = st.session_state.human_side
    level = st.session_state.cpu_level
    changed=False
    guard=0
    while p != human and not game_over(b) and guard<4:  # 連続2パスで終局する想定で控えめループ
        mv=cpu_pick(b,p,level)
        if mv is None:
            p=opponent(p)  # パス
            if not valid_moves(b,p): break  # 連続パスで終局
        else:
            b=apply_move(b,mv,p)
            p=opponent(p)
            # 相手（人/CPU）がパスなら手番を戻す
            if not valid_moves(b,p) and not game_over(b):
                p=opponent(p)
        changed=True; guard+=1
    if changed:
        st.session_state.board=b
        st.session_state.player=p
        set_query(encode_board(b), p)

# ─────────────────────────────────────────────────────────────
# 初期化（URL→復元 or 新規）
# ─────────────────────────────────────────────────────────────
if "board" not in st.session_state:
    st.session_state.board=new_board()
    st.session_state.player=BLACK
if "human_side" not in st.session_state:
    st.session_state.human_side=BLACK
if "cpu_level" not in st.session_state:
    st.session_state.cpu_level="Hard"

s_q,p_q,mv_q=read_query()
if s_q and p_q:
    st.session_state.board=decode_board(s_q)
    try: st.session_state.player=int(p_q)
    except: st.session_state.player=BLACK
else:
    set_query(encode_board(st.session_state.board), st.session_state.player)

# ─────────────────────────────────────────────────────────────
# 設定UI（上部）
# ─────────────────────────────────────────────────────────────
left, right = st.columns([1,1])
with left:
    side = st.radio("あなたの手番", options=["黒(先手)","白(後手)"], index=0 if st.session_state.human_side==BLACK else 1, horizontal=True)
with right:
    level = st.selectbox("CPUの強さ", options=["Easy","Medium","Hard"], index=["Easy","Medium","Hard"].index(st.session_state.cpu_level))

if (side.startswith("黒") and st.session_state.human_side!=BLACK) or (side.startswith("白") and st.session_state.human_side!=WHITE) or (level!=st.session_state.cpu_level):
    st.session_state.human_side = BLACK if side.startswith("黒") else WHITE
    st.session_state.cpu_level = level
    # 盤面を初期化して設定反映
    st.session_state.board = new_board()
    st.session_state.player = BLACK
    set_query(encode_board(st.session_state.board), st.session_state.player)

# ─────────────────────────────────────────────────────────────
# クリック（mv）が来ていたら適用 → 必要ならCPUを自動実行
# ─────────────────────────────────────────────────────────────
if mv_q:
    try:
        r,c = map(int, mv_q.split("_"))
        b=st.session_state.board; p=st.session_state.player
        if (r,c) in valid_moves(b,p):
            b2=apply_move(b,(r,c),p); p2=opponent(p)
            if not valid_moves(b2,p2) and not game_over(b2):  # 相手がパス
                p2=opponent(p2)
            st.session_state.board=b2; st.session_state.player=p2
            set_query(encode_board(b2), p2)
            # ユーザーの手の後にCPU番なら自動思考
            cpu_auto_play()
    except Exception:
        set_query(encode_board(st.session_state.board), st.session_state.player)

# ユーザーが後手を選んでいて、ゲーム開始直後にCPU番の場合（ページロード時）
if st.session_state.player != st.session_state.human_side and not game_over(st.session_state.board):
    cpu_auto_play()

# ─────────────────────────────────────────────────────────────
# UI（スコアと盤）
# ─────────────────────────────────────────────────────────────
b_cnt,w_cnt=score(st.session_state.board)
st.write(f"⚫ {b_cnt} - ⚪ {w_cnt}")
st.write(f"Turn: {'Black' if st.session_state.player==BLACK else 'White'}")

board=st.session_state.board
valid=set(valid_moves(board, st.session_state.player))
human=st.session_state.human_side

# 盤HTML（人の番の合法手=フォーム、その他は静的div）
cells=[]
s_now=encode_board(board); p_now=st.session_state.player
for r in range(8):
    for c in range(8):
        if p_now==human and (r,c) in valid and board[r][c]==EMPTY:
            cells.append(
                f"<form class='mv' method='get' target='_self'>"
                f"  <input type='hidden' name='mv' value='{r}_{c}'/>"
                f"  <input type='hidden' name='s'  value='{s_now}'/>"
                f"  <input type='hidden' name='p'  value='{p_now}'/>"
                f"  <button class='mvbtn' type='submit' title='Move {r+1},{c+1}'>◎</button>"
                f"</form>"
            )
        else:
            v=board[r][c]
            if v==BLACK: cells.append("<div class='cell'><div class='piece black'></div></div>")
            elif v==WHITE: cells.append("<div class='cell'><div class='piece white'></div></div>")
            else: cells.append("<div class='cell'></div>")

html = "<div class='board-scroll'><div class='board-wrap'><div class='board'>" + "".join(cells) + "</div></div></div>"
st.markdown(html, unsafe_allow_html=True)

# 終局・操作
if game_over(st.session_state.board):
    b_cnt,w_cnt=score(st.session_state.board)
    if b_cnt>w_cnt: st.success(f"Game Over! Winner: Black ({b_cnt}-{w_cnt})")
    elif w_cnt>b_cnt: st.success(f"Game Over! Winner: White ({w_cnt}-{b_cnt})")
    else: st.info(f"Game Over! Draw ({b_cnt}-{w_cnt})")

if st.button("New Game"):
    st.session_state.board=new_board()
    st.session_state.player=BLACK
    set_query(encode_board(st.session_state.board), st.session_state.player)
