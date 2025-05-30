import sys
sys.path.append('../')
from utils import get_centre,measure_distace

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70*70  # Maximum distance to consider a player as the ball holder

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_centre(ball_bbox)
        
        min_distance = float('inf')
        closest_player_id = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            dist_left = measure_distace(ball_position, (player_bbox[0],player_bbox[-1]))
            dist_right = measure_distace(ball_position, (player_bbox[2],player_bbox[-1]))
            distance=min(dist_left,dist_right)

            if distance < self.max_player_ball_distance and distance < min_distance:
                min_distance = distance
                closest_player_id = player_id
        return closest_player_id
            
