import pygame
import sys
import math
import copy
import time  # Add this import for time measurement

# ----- Constants & Board Configuration -----
DEFAULT_WIDTH, DEFAULT_HEIGHT = 800, 800  # Increased default size
CURRENT_WIDTH, CURRENT_HEIGHT = DEFAULT_WIDTH, DEFAULT_HEIGHT  # Track current window size
TOLERANCE = 10                   # Pixel tolerance for clicking on an edge
DEPTH = 3                        # Depth for the minimax search
BOARD_RADIUS = 2                 # Board "radius" for hex board: cells with max(|q|,|r|,|s|)<=3
HEX_SIZE = 40                    # Size of each hexagon
MARGIN = 50                      # Margin from window edge to the board
SPACING = 100                    # Distance between adjacent dots

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)   # Human color
BLUE  = (0, 0, 255)   # AI color
GRAY  = (200, 200, 200)
LIGHT_GREEN = (200, 255, 200)  # For cache hits
LIGHT_YELLOW = (255, 255, 200) # For newly evaluated positions
YELLOW = (255, 255, 0)  # Bright yellow for last move highlight

# Add stats tracking for visualization
AI_STATS = {
    'cache_hits': 0,
    'positions_evaluated': 0,
    'thinking_time': 0.0,
    'visualization_edges': {},  # To track which edges to highlight
    'total_cache_size': 0
}

# ----- Scale Calculation Functions -----
def get_scale_factor():
    """Calculate the scale factor based on current window size vs default size"""
    x_scale = CURRENT_WIDTH / DEFAULT_WIDTH
    y_scale = CURRENT_HEIGHT / DEFAULT_HEIGHT
    # Use the smaller scale to ensure everything fits
    return min(x_scale, y_scale)

def scale_point(point, scale_factor=None):
    """Scale a point based on the current window size"""
    if scale_factor is None:
        scale_factor = get_scale_factor()
    x, y = point
    scaled_x = x * scale_factor
    scaled_y = y * scale_factor
    return (scaled_x, scaled_y)

def scale_hex_size():
    """Scale the hexagon size based on the window size"""
    return HEX_SIZE * get_scale_factor()

# ----- Hex Geometry Helper Functions (Updated for scaling) -----
def axial_to_pixel(q, r, offset_x=0, offset_y=0):
    """
    Convert axial hex coordinates (q, r) to pixel coordinates
    using the pointy-topped layout, and add an additional offset.
    """
    scale = get_scale_factor()
    x = scale_hex_size() * math.sqrt(3) * (q + r/2) + offset_x * scale
    y = scale_hex_size() * 3/2 * r + offset_y * scale
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
    WIDTH = CURRENT_WIDTH - 2 * MARGIN
    HEIGHT = CURRENT_HEIGHT - 2 * MARGIN
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
    state['last_move'] = None  # Initialize last_move to None
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
    
    # Store the last move in the state
    new_state['last_move'] = move
    
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

# Add this function to create a unique hash for a game state
def hash_state(state):
    """Create a hash representation of the board state for the transposition table"""
    # Use a tuple of sorted (edge, owner) pairs as the hash
    edge_tuples = tuple(sorted((edge, owner) for edge, owner in state['edges'].items()))
    return hash(edge_tuples)

# Modified minimax with transposition table
def minimax(state, depth, alpha, beta, maximizingPlayer, transposition_table=None):
    """Minimax algorithm with alpha-beta pruning and transposition table"""
    if transposition_table is None:
        transposition_table = {}
    
    # Generate a hash for the current state
    state_hash = hash_state(state)
    
    # Check if this state is already in our table at sufficient depth
    if state_hash in transposition_table and transposition_table[state_hash]['depth'] >= depth:
        AI_STATS['cache_hits'] += 1
        return transposition_table[state_hash]['value'], transposition_table[state_hash]['move']
    
    AI_STATS['positions_evaluated'] += 1
    
    if depth == 0 or is_terminal(state):
        value = evaluate(state)
        transposition_table[state_hash] = {'value': value, 'move': None, 'depth': depth}
        AI_STATS['total_cache_size'] = len(transposition_table)
        return value, None

    possible_moves = get_possible_moves(state)
    ordered_moves = order_moves(state, possible_moves, maximizingPlayer)
    
    best_move = None
    if maximizingPlayer:
        maxEval = -math.inf
        for move in ordered_moves:
            new_state, extra_turn = apply_move(state, move, 1)
            if extra_turn:
                eval_score, _ = minimax(new_state, depth - 1, alpha, beta, True, transposition_table)
            else:
                eval_score, _ = minimax(new_state, depth - 1, alpha, beta, False, transposition_table)
            if eval_score > maxEval:
                maxEval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        transposition_table[state_hash] = {'value': maxEval, 'move': best_move, 'depth': depth}
        AI_STATS['total_cache_size'] = len(transposition_table)
        return maxEval, best_move
    else:
        minEval = math.inf
        for move in ordered_moves:
            new_state, extra_turn = apply_move(state, move, 0)
            if extra_turn:
                eval_score, _ = minimax(new_state, depth - 1, alpha, beta, False, transposition_table)
            else:
                eval_score, _ = minimax(new_state, depth - 1, alpha, beta, True, transposition_table)
            if eval_score < minEval:
                minEval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        transposition_table[state_hash] = {'value': minEval, 'move': best_move, 'depth': depth}
        AI_STATS['total_cache_size'] = len(transposition_table)
        return minEval, best_move

# Helper function to order moves - prioritize moves that complete boxes
def order_moves(state, moves, maximizing_player):
    """Order moves to improve alpha-beta pruning efficiency"""
    move_scores = []
    player = 1 if maximizing_player else 0
    
    for move in moves:
        score = 0
        # Check how many edges are already drawn for each affected cell
        for cell in state['edge_cells'][move]:
            if state['cells'][cell] == -1:  # unclaimed cell
                edges_drawn = 0
                for edge in state['cell_edges'][cell]:
                    if edge == move or state['edges'][edge] != -1:
                        edges_drawn += 1
                # Prioritize moves that complete or nearly complete cells
                if edges_drawn == 6:  # Would complete a cell
                    score += 100
                elif edges_drawn == 5:  # One away from completion
                    score += 50
        
        move_scores.append((move, score))
    
    # Sort moves by score (descending for maximizing, ascending for minimizing)
    if maximizing_player:
        return [move for move, score in sorted(move_scores, key=lambda x: -x[1])]
    else:
        return [move for move, score in sorted(move_scores, key=lambda x: x[1])]

# ----- Updated Drawing Function -----
def draw_board(screen, state, font):
    """
    Render the board with visualization of the AI's thinking process.
    """
    screen.fill(WHITE)
    scale = get_scale_factor()
    scaled_margin = MARGIN * scale
    
    # Fill claimed cells
    for cell, owner in state['cells'].items():
        if owner != -1:
            # Scale the vertices
            vertices = [scale_point(v) for v in state['cell_vertices'][cell]]
            color = RED if owner == 0 else BLUE
            pygame.draw.polygon(screen, color, vertices)
    
    # Draw edges with visualization highlights
    for edge, owner in state['edges'].items():
        # Scale the edge endpoints
        a, b = edge
        scaled_a = scale_point(a)
        scaled_b = scale_point(b)
        
        # Check if this is the last move
        is_last_move = (edge == state.get('last_move'))
        
        if owner != -1:
            # If this is the last move, highlight it in yellow
            if is_last_move:
                color = YELLOW
                width = int(6 * scale)  # Make it slightly wider for emphasis
            else:
                color = RED if owner == 0 else BLUE
                width = int(4 * scale)
        else:
            # Check if this edge is in our visualization dictionary
            if edge in AI_STATS['visualization_edges']:
                if AI_STATS['visualization_edges'][edge] == 'cache_hit':
                    color = LIGHT_GREEN  # Cache hit
                    width = int(2 * scale)
                else:
                    color = LIGHT_YELLOW  # Newly evaluated
                    width = int(2 * scale)
            else:
                color = GRAY
                width = max(1, int(1 * scale))
        
        pygame.draw.line(screen, color, scaled_a, scaled_b, width)
    
    # Draw vertices
    drawn_vertices = set()
    for vertices in state['cell_vertices'].values():
        for vertex in vertices:
            scaled_vertex = scale_point(vertex)
            if scaled_vertex not in drawn_vertices:
                pygame.draw.circle(screen, BLACK, scaled_vertex, int(4 * scale))
                drawn_vertices.add(scaled_vertex)
    
    # Draw score text at the bottom
    score_text = font.render(f"Human: {state['score'][0]}  AI: {state['score'][1]}", True, BLACK)
    screen.blit(score_text, (scaled_margin, CURRENT_HEIGHT - scaled_margin))
    
    # Draw AI stats - position based on current window size
    stats_x = CURRENT_WIDTH - 200 * scale
    stats_y = scaled_margin
    line_height = 30 * scale
    
    stats_text = font.render(f"Time: {AI_STATS['thinking_time']:.3f}s", True, BLUE)
    screen.blit(stats_text, (stats_x, stats_y))
    
    cache_text = font.render(f"Cache: {AI_STATS['total_cache_size']} positions", True, BLUE)
    screen.blit(cache_text, (stats_x, stats_y + line_height))
    
    hits_text = font.render(f"Hits: {AI_STATS['cache_hits']}", True, LIGHT_GREEN)
    screen.blit(hits_text, (stats_x, stats_y + line_height * 2))
    
    evals_text = font.render(f"Evals: {AI_STATS['positions_evaluated']}", True, LIGHT_YELLOW)
    screen.blit(evals_text, (stats_x, stats_y + line_height * 3))
    
    # If we have both hits and evaluations, show efficiency
    if AI_STATS['positions_evaluated'] + AI_STATS['cache_hits'] > 0:
        efficiency = AI_STATS['cache_hits'] / (AI_STATS['positions_evaluated'] + AI_STATS['cache_hits']) * 100
        eff_text = font.render(f"Efficiency: {efficiency:.1f}%", True, BLUE)
        screen.blit(eff_text, (stats_x, stats_y + line_height * 4))
    
    # Add legend
    pygame.draw.rect(screen, LIGHT_GREEN, (stats_x - 20 * scale, stats_y + line_height * 5, 15 * scale, 15 * scale))
    legend1 = font.render("Cache hit", True, BLACK)
    screen.blit(legend1, (stats_x, stats_y + line_height * 5))
    
    pygame.draw.rect(screen, LIGHT_YELLOW, (stats_x - 20 * scale, stats_y + line_height * 6, 15 * scale, 15 * scale))
    legend2 = font.render("New evaluation", True, BLACK)
    screen.blit(legend2, (stats_x, stats_y + line_height * 6))
    
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

# Updated click detection for resized windows
def get_clicked_edge(pos, state):
    """
    Given the mouse position, determine if it is close to an undrawn edge.
    Returns the edge if found, otherwise None.
    """
    x, y = pos
    scale = get_scale_factor()
    inverse_scale = 1 / scale
    
    # Convert mouse position to original coordinate system
    original_x, original_y = x * inverse_scale, y * inverse_scale
    
    for edge, owner in state['edges'].items():
        if owner == -1:
            a, b = edge
            if point_line_distance((original_x, original_y), a, b) < TOLERANCE:
                return edge
    return None

# ----- Main Game Loop with resize handling -----
def main():
    global CURRENT_WIDTH, CURRENT_HEIGHT
    
    pygame.init()
    screen = pygame.display.set_mode((DEFAULT_WIDTH, DEFAULT_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Hexagonal Dots and Boxes with Minimax AI")
    font = pygame.font.SysFont(None, 28)  # Smaller font to fit all stats
    clock = pygame.time.Clock()
    state = init_state()
    running = True

    # Create a transposition table that persists between moves
    transposition_table = {}
    
    while running:
        # Handle events (e.g. closing window, clicks, resize)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                # Update current window size
                CURRENT_WIDTH, CURRENT_HEIGHT = event.size
                screen = pygame.display.set_mode((CURRENT_WIDTH, CURRENT_HEIGHT), pygame.RESIZABLE)
                # Recalculate font size based on new window size
                font_size = int(28 * get_scale_factor())
                font = pygame.font.SysFont(None, max(12, font_size))
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Only allow moves if it is the human player's turn.
                if state['turn'] == 0:
                    move = get_clicked_edge(pygame.mouse.get_pos(), state)
                    if move is not None:
                        new_state, extra_turn = apply_move(state, move, 0)
                        state = new_state
                        draw_board(screen, state, font)
                        # Reset visualization edges after human move
                        AI_STATS['visualization_edges'] = {}

        # AI turn
        if state['turn'] == 1:
            pygame.display.set_caption("AI is thinking...")
            
            # Reset counters for this turn
            AI_STATS['cache_hits'] = 0
            AI_STATS['positions_evaluated'] = 0
            AI_STATS['visualization_edges'] = {}
            
            # Small delay to show update and redraw the board
            draw_board(screen, state, font)
            pygame.time.delay(200)
            
            # Record start time for AI thinking
            start_time = time.time()
            
            # AI makes its move
            _, move = minimax(state, DEPTH, -math.inf, math.inf, True, transposition_table)
            
            # Calculate and store thinking time
            AI_STATS['thinking_time'] = time.time() - start_time
            
            # Visualize the move chosen by the AI
            if move is not None:
                # Generate visualization data showing which edges were evaluated
                possible_moves = get_possible_moves(state)
                for edge in possible_moves:
                    edge_hash = hash_state(apply_move(state, edge, 1)[0])
                    if edge_hash in transposition_table:
                        if edge == move:
                            # Highlight the chosen move differently
                            AI_STATS['visualization_edges'][edge] = 'chosen'
                        else:
                            AI_STATS['visualization_edges'][edge] = 'cache_hit'
                
                # Show visualization for a moment before making the move
                draw_board(screen, state, font)
                pygame.display.flip()
                pygame.time.delay(1000)  # Pause to show the visualization
                
                # Apply the AI's move
                new_state, extra_turn = apply_move(state, move, 1)
                state = new_state
                
            pygame.display.set_caption("Hexagonal Dots and Boxes with Minimax AI")

        draw_board(screen, state, font)
        if is_terminal(state):
            running = False
        clock.tick(30)

    # Final drawing and delay before quitting
    draw_board(screen, state, font)
    
    # Determine winner
    if state['score'][0] > state['score'][1]:
        result = "You Win!"
    elif state['score'][0] < state['score'][1]:
        result = "AI Wins!"
    else:
        result = "It's a Tie!"
    
    # Scale text position for game over message    
    final_text = font.render(f"Game Over! {result}", True, BLACK)
    screen.blit(final_text, (CURRENT_WIDTH // 2 - 120 * get_scale_factor(), CURRENT_HEIGHT // 2))
    pygame.display.flip()
    pygame.time.delay(3000)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()