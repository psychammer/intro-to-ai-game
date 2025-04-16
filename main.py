import pygame
import sys
import math
import copy

# ----- Constants & Board Configuration -----
WIDTH, HEIGHT = 600, 600         # Window dimensions
TOLERANCE = 10                   # Pixel tolerance for clicking on an edge
DEPTH = 3                        # Depth for the minimax search
BOARD_RADIUS = 2                 # Board "radius" for hex board: cells with max(|q|,|r|,|s|)<=3
HEX_SIZE = 40                    # Size of each hexagon
MARGIN = 50                       # Margin from window edge to the board
SPACING = 100                     # Distance between adjacent dots

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)   # Human color
BLUE  = (0, 0, 255)   # AI color
GRAY  = (200, 200, 200)


# ----- Hex Geometry Helper Functions -----
def axial_to_pixel(q, r, offset_x=0, offset_y=0):
    """
    Convert axial hex coordinates (q, r) to pixel coordinates
    using the pointy-topped layout, and add an additional offset.
    """
    x = HEX_SIZE * math.sqrt(3) * (q + r/2) + offset_x
    y = HEX_SIZE * 3/2 * r + offset_y
    return (x, y)

def polygon_vertices(center, size):
    """
    Compute the 6 vertices for a pointy-topped hexagon given its center.
    """
    cx, cy = center
    vertices = []
    for i in range(6):
        angle_deg = 60 * i - 30  # so the top is a point
        angle_rad = math.radians(angle_deg)
        vx = cx + size * math.cos(angle_rad)
        vy = cy + size * math.sin(angle_rad)
        vertices.append((round(vx), round(vy)))
    return vertices

def normalize_edge(v1, v2):
    """
    Order the two endpoints of an edge so that every edge has a unique representation.
    """
    return tuple(sorted([v1, v2]))

# ----- Game State Setup for Hexagonal Board -----
def init_state():
    """
    Build the hexagonal board state using axial coordinates.
    This version computes a bounding box for the board (with no offset) and then
    calculates an additional offset to center the board in the window.
    """
    # First pass: compute vertices for each valid hex cell with no offset.
    temp_vertices = {}
    valid_cells = []
    for q in range(-BOARD_RADIUS, BOARD_RADIUS+1):
        for r in range(-BOARD_RADIUS, BOARD_RADIUS+1):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) <= BOARD_RADIUS:
                valid_cells.append((q, r))
                center = axial_to_pixel(q, r, 0, 0)
                vertices = polygon_vertices(center, HEX_SIZE)
                temp_vertices[(q, r)] = vertices

    # Compute bounding box (min/max x and y) from all vertices.
    all_x = []
    all_y = []
    for vertices in temp_vertices.values():
        for (x, y) in vertices:
            all_x.append(x)
            all_y.append(y)
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    board_width = max_x - min_x
    board_height = max_y - min_y

    # Compute offsets to center the board in the window.
    offset_x = (WIDTH - board_width) / 2 - min_x
    offset_y = (HEIGHT - board_height) / 2 - min_y

    # Build final state using the computed offset.
    state = {}
    state['cells'] = {}
    state['edges'] = {}
    state['cell_edges'] = {}
    state['edge_cells'] = {}
    state['cell_vertices'] = {}

    for cell in valid_cells:
        q, r = cell
        center = axial_to_pixel(q, r, offset_x, offset_y)
        vertices = polygon_vertices(center, HEX_SIZE)
        state['cells'][(q, r)] = -1  # Unclaimed
        state['cell_vertices'][(q, r)] = vertices
        cell_edge_list = []
        for i in range(6):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % 6]
            edge = normalize_edge(v1, v2)
            cell_edge_list.append(edge)
            # Register the edge if not present and link the cell to the edge.
            if edge not in state['edges']:
                state['edges'][edge] = -1
                state['edge_cells'][edge] = []
            if (q, r) not in state['edge_cells'][edge]:
                state['edge_cells'][edge].append((q, r))
        state['cell_edges'][(q, r)] = cell_edge_list

    state['turn'] = 0  # Human starts
    state['score'] = [0, 0]
    return state

# ----- Move Utilities -----
def get_possible_moves(state):
    """Return a list of all undrawn edges."""
    moves = []
    for edge, owner in state['edges'].items():
        if owner == -1:
            moves.append(edge)
    return moves

def apply_move(state, move, player):
    """
    Apply the move (drawing an edge) by the given player.
    For each cell adjacent to the move, check if all 6 edges have been drawn.
    If so, mark the cell with the player's number and update score.
    Returns the new state and a flag for extra turn.
    """
    new_state = copy.deepcopy(state)
    extra_turn = False
    new_state['edges'][move] = player
    # Check each cell that uses this edge.
    for cell in new_state['edge_cells'][move]:
        if new_state['cells'][cell] == -1:  # still unclaimed
            completed = True
            for edge in new_state['cell_edges'][cell]:
                if new_state['edges'][edge] == -1:
                    completed = False
                    break
            if completed:
                new_state['cells'][cell] = player
                new_state['score'][player] += 1
                extra_turn = True
    if not extra_turn:
        new_state['turn'] = 1 - player
    return new_state, extra_turn

def is_terminal(state):
    """The game is over if there are no moves left."""
    return len(get_possible_moves(state)) == 0

def evaluate(state):
    """
    Evaluation function for minimax.
    Returns (AI score - Human score).
    """
    return state['score'][1] - state['score'][0]

# ----- Minimax with Alpha–Beta Pruning -----
def minimax(state, depth, alpha, beta, maximizingPlayer):
    if depth == 0 or is_terminal(state):
        return evaluate(state), None

    possible_moves = get_possible_moves(state)
    best_move = None
    if maximizingPlayer:
        maxEval = -math.inf
        for move in possible_moves:
            new_state, extra_turn = apply_move(state, move, 1)  # AI is player 1
            if extra_turn:
                eval_score, _ = minimax(new_state, depth - 1, alpha, beta, True)
            else:
                eval_score, _ = minimax(new_state, depth - 1, alpha, beta, False)
            if eval_score > maxEval:
                maxEval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return maxEval, best_move
    else:
        minEval = math.inf
        for move in possible_moves:
            new_state, extra_turn = apply_move(state, move, 0)  # Human is player 0
            if extra_turn:
                eval_score, _ = minimax(new_state, depth - 1, alpha, beta, False)
            else:
                eval_score, _ = minimax(new_state, depth - 1, alpha, beta, True)
            if eval_score < minEval:
                minEval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return minEval, best_move

# ----- Pygame Drawing & User Input Helpers -----
def draw_board(screen, state, font):
    """
    Render the board:
      - Fill completed hexagonal cells with the player's color.
      - Draw each edge (thicker colored lines for drawn edges, thin grey for undrawn).
      - Draw vertices as small circles.
      - Display current scores.
    """
    screen.fill(WHITE)
    # Fill claimed cells
    for cell, owner in state['cells'].items():
        if owner != -1:
            vertices = state['cell_vertices'][cell]
            color = RED if owner == 0 else BLUE
            pygame.draw.polygon(screen, color, vertices)
    # Draw edges
    for edge, owner in state['edges'].items():
        a, b = edge
        if owner != -1:
            color = RED if owner == 0 else BLUE
            width = 4
        else:
            color = GRAY
            width = 1
        pygame.draw.line(screen, color, a, b, width)
    # Draw vertices (avoid redrawing duplicates)
    drawn_vertices = set()
    for vertices in state['cell_vertices'].values():
        for vertex in vertices:
            if vertex not in drawn_vertices:
                pygame.draw.circle(screen, BLACK, vertex, 4)
                drawn_vertices.add(vertex)
    # Draw score text at the bottom
    score_text = font.render(f"Human: {state['score'][0]}  AI: {state['score'][1]}", True, BLACK)
    screen.blit(score_text, (MARGIN, HEIGHT - MARGIN))
    pygame.display.flip()

def point_line_distance(p, a, b):
    """
    Compute the minimum distance from point p to the line segment defined by endpoints a and b.
    """
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)

def get_clicked_edge(pos, state):
    """
    Given the mouse position, determine if it is close to an undrawn edge.
    Returns the edge if found, otherwise None.
    """
    x, y = pos
    for edge, owner in state['edges'].items():
        if owner == -1:
            a, b = edge
            if point_line_distance((x, y), a, b) < TOLERANCE:
                return edge
    return None

# ----- Main Game Loop -----
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Hexagonal Dots and Boxes with Minimax AI")
    font = pygame.font.SysFont(None, 36)
    clock = pygame.time.Clock()
    state = init_state()
    running = True

    while running:
        # Handle events (e.g. closing window, clicks)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Only allow moves if it is the human player's turn.
                if state['turn'] == 0:
                    move = get_clicked_edge(pygame.mouse.get_pos(), state)
                    if move is not None:
                        new_state, extra_turn = apply_move(state, move, 0)
                        state = new_state
                        draw_board(screen, state, font)

        # AI turn
        if state['turn'] == 1:
            pygame.display.set_caption("AI is thinking...")
            pygame.time.delay(500)  # Small delay to show update
            _, move = minimax(state, DEPTH, -math.inf, math.inf, True)
            if move is not None:
                new_state, extra_turn = apply_move(state, move, 1)
                state = new_state
            pygame.display.set_caption("Hexagonal Dots and Boxes with Minimax AI")

        draw_board(screen, state, font)
        if is_terminal(state):
            running = False
        clock.tick(30)

    # Final drawing and delay before quitting
    draw_board(screen, state, font)
    final_text = font.render("Game Over!", True, BLACK)
    screen.blit(final_text, (WIDTH // 2 - 70, HEIGHT // 2))
    pygame.display.flip()
    pygame.time.delay(3000)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
