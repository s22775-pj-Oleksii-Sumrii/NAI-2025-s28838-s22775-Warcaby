"""
   Projekt: Warcaby z trybem PvP i AI
ZASADY GRY:
- celem jest zbicie wszystkich pionków przeciwnika. 
- piony poruszają się do przodu tylko o jedno pole po przekątnej. 
- zbicie odbywa się po przekątnej przeskakując przez pionek przeciwnika na puste pole. 
- podczas jednego ruchu tym samym pionkiem można zbić więcej niż jeden pion.

AUTORZY:
- Oleksii Sumrii
- Oskar Szyszko

INSTRUKCJA PRZYGOTOWANIA ŚRODOWISKA:
1. Zainstaluj Python 3.8 lub nowszy z https://python.org  
2. Zainstaluj Visual Studio Code (VS Code) z https://code.visualstudio.com/
3. Zainstaluj rozszerzenie „Python” w VS Code:
	- Otwórz VS Code
	- Kliknij ikonę rozszerzeń po lewej stronie albo Ctrl+Shift+X
	- Wyszukaj Python i zainstaluj rozszerzenie od Microsoft
"""

import tkinter as tk
import copy

BOARD_SIZE = 8
TILE_SIZE = 80

class CheckersGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Warcaby 🟤⚪")

        self.mode = None  # "ai" lub "pvp"
        self.create_menu()

    # === MENU STARTOWE ===
    def create_menu(self):
        self.menu_frame = tk.Frame(self.root, bg="#EEE")
        self.menu_frame.pack(fill="both", expand=True)

        title = tk.Label(self.menu_frame, text="Warcaby 🟤⚪", font=("Arial", 32, "bold"), bg="#EEE")
        title.pack(pady=50)

        btn_pvp = tk.Button(self.menu_frame, text="🧍‍♂️ vs 🧍‍♀️  (2 graczy)", font=("Arial", 20),
                            width=30, command=lambda: self.start_game("pvp"))
        btn_pvp.pack(pady=20)

        btn_ai = tk.Button(self.menu_frame, text="🧍‍♂️ vs 🤖  (z komputerem)", font=("Arial", 20),
                           width=30, command=lambda: self.start_game("ai"))
        btn_ai.pack(pady=20)

    def start_game(self, mode):
        self.mode = mode
        self.menu_frame.destroy()

        self.canvas = tk.Canvas(self.root, width=BOARD_SIZE*TILE_SIZE, height=BOARD_SIZE*TILE_SIZE)
        self.canvas.pack()

        self.board = self.init_board()
        self.selected = None
        self.turn = 'w'  # 'w' - biały, 'b' - czarny
        self.draw_board()
        self.canvas.bind("<Button-1>", self.on_click)

    # === LOGIKA GRY ===
    def init_board(self):
        board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for r in range(3):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1:
                    board[r][c] = 'b'
        for r in range(5, 8):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 == 1:
                    board[r][c] = 'w'
        return board

    def draw_board(self):
        self.canvas.delete("all")
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                color = "#D18B47" if (r + c) % 2 == 1 else "#FFCE9E"
                self.canvas.create_rectangle(
                    c*TILE_SIZE, r*TILE_SIZE, (c+1)*TILE_SIZE, (r+1)*TILE_SIZE,
                    fill=color, outline="black"
                )

                piece = self.board[r][c]
                if piece != ' ':
                    fill = "white" if piece.lower() == 'w' else "black"
                    x, y = c*TILE_SIZE + TILE_SIZE//2, r*TILE_SIZE + TILE_SIZE//2
                    self.canvas.create_oval(
                        x-30, y-30, x+30, y+30, fill=fill, outline="gray", width=2
                    )
                    if piece.isupper():  # damka
                        self.canvas.create_text(x, y, text="👑", font=("Arial", 20))

        if self.selected:
            r, c = self.selected
            self.canvas.create_rectangle(
                c*TILE_SIZE, r*TILE_SIZE, (c+1)*TILE_SIZE, (r+1)*TILE_SIZE,
                outline="yellow", width=4
            )

    def on_click(self, event):
        col = event.x // TILE_SIZE
        row = event.y // TILE_SIZE
        if row >= BOARD_SIZE or col >= BOARD_SIZE:
            return

        # Ruch tylko, gdy tura gracza (w AI tylko biały może klikać)
        if self.mode == "ai" and self.turn != "w":
            return

        if self.selected:
            if self.try_move(self.selected, (row, col)):
                self.next_turn()
            self.selected = None
        else:
            piece = self.board[row][col]
            if piece != ' ' and piece.lower() == self.turn:
                self.selected = (row, col)
        self.draw_board()

    def next_turn(self):
        self.turn = 'b' if self.turn == 'w' else 'w'
        self.draw_board()

        if self.mode == "ai" and self.turn == 'b':
            self.root.after(600, self.ai_move)

    def try_move(self, start, end):
        sr, sc = start
        er, ec = end
        piece = self.board[sr][sc]
        if self.board[er][ec] != ' ':
            return False
        dr, dc = er - sr, ec - sc
        if abs(dc) != abs(dr):
            return False

        # zwykły ruch
        if abs(dr) == 1:
            if piece == 'w' and dr == -1:
                self.move_piece(sr, sc, er, ec)
                return True
            elif piece == 'b' and dr == 1:
                self.move_piece(sr, sc, er, ec)
                return True
            elif piece.isupper():
                self.move_piece(sr, sc, er, ec)
                return True

        # bicie
        if abs(dr) == 2:
            mid_r, mid_c = (sr + er)//2, (sc + ec)//2
            mid_piece = self.board[mid_r][mid_c]
            if mid_piece != ' ' and mid_piece.lower() != piece.lower():
                self.board[mid_r][mid_c] = ' '
                self.move_piece(sr, sc, er, ec)
                return True

        return False

    def move_piece(self, sr, sc, er, ec):
        piece = self.board[sr][sc]
        self.board[sr][sc] = ' '
        self.board[er][ec] = piece
        # promocja na damkę
        if piece == 'w' and er == 0:
            self.board[er][ec] = 'W'
        elif piece == 'b' and er == 7:
            self.board[er][ec] = 'B'

    # === AI ===
    def ai_move(self):
        moves = self.get_all_moves('b')
        if not moves:
            self.end_game("🏁 Wygrał gracz! 🎉")
            return

        # wybierz najlepszy ruch według heurystyki
        best_score = float('-inf')
        best_move = None

        for move in moves:
            simulated = copy.deepcopy(self.board)
            self.simulate_move(simulated, *move)
            score = self.evaluate_board(simulated)
            if score > best_score:
                best_score = score
                best_move = move

        if best_move:
            self.try_move(*best_move)

        self.turn = 'w'
        self.draw_board()

        if not self.get_all_moves('w'):
            self.end_game("🏁 Wygrał komputer 🤖")

    def get_all_moves(self, color):
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.board[r][c]
                if piece.lower() == color:
                    for dr in [-1, 1, -2, 2]:
                        for dc in [-1, 1, -2, 2]:
                            er, ec = r + dr, c + dc
                            if 0 <= er < BOARD_SIZE and 0 <= ec < BOARD_SIZE:
                                temp_game = [row.copy() for row in self.board]
                                if self.try_move_sim(temp_game, (r, c), (er, ec)):
                                    moves.append(((r, c), (er, ec)))
        return moves

    def simulate_move(self, board, start, end):
        sr, sc = start
        er, ec = end
        piece = board[sr][sc]
        board[sr][sc] = ' '
        # bicie
        if abs(er - sr) == 2:
            mid_r, mid_c = (sr + er)//2, (sc + ec)//2
            board[mid_r][mid_c] = ' '
        board[er][ec] = piece
        # promocja
        if piece == 'b' and er == 7:
            board[er][ec] = 'B'

    def try_move_sim(self, board, start, end):
        sr, sc = start
        er, ec = end
        piece = board[sr][sc]
        if board[er][ec] != ' ':
            return False
        dr, dc = er - sr, ec - sc
        if abs(dc) != abs(dr):
            return False
        if abs(dr) == 1:
            if piece == 'w' and dr == -1:
                return True
            elif piece == 'b' and dr == 1:
                return True
            elif piece.isupper():
                return True
        if abs(dr) == 2:
            mid_r, mid_c = (sr + er)//2, (sc + ec)//2
            mid_piece = board[mid_r][mid_c]
            if mid_piece != ' ' and mid_piece.lower() != piece.lower():
                return True
        return False

    def evaluate_board(self, board):
        """Prosta heurystyka: punkty za pionki i damki"""
        score = 0
        for row in board:
            for p in row:
                if p == 'b': score += 3
                if p == 'B': score += 5
                if p == 'w': score -= 3
                if p == 'W': score -= 5
        return score

    def end_game(self, message):
        self.canvas.create_text(
            BOARD_SIZE*TILE_SIZE//2,
            BOARD_SIZE*TILE_SIZE//2,
            text=message,
            fill="red",
            font=("Arial", 32, "bold")
        )
        self.canvas.unbind("<Button-1>")

# === START PROGRAMU ===
if __name__ == "__main__":
    root = tk.Tk()
    game = CheckersGame(root)
    root.mainloop()