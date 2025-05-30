from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assignment import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from tqdm import tqdm
import numpy as np
def main():
    video_frames = read_video('video/08fd33_4.mp4')
    # video_frames = video_frames[::10]  # Downsample the video frames
    # video_frames = video_frames[:10]  # Limit to 100 frames for testing
    # Initialize the tracker with the model path 
    tracker=Tracker('models/best.pt')

    tracks= tracker.get_object_tracks(video_frames,read_from_stub=True,
                                      stub_path='stubs/track_stubs.pkl') 
    #interpolate ball
    tracks['ball'] = tracker.interpolate_ball(tracks['ball'])
    team_assigner=TeamAssigner()

    team_assigner.assign_teams(video_frames[0],tracks['player'][0]) 

    for frame_num,player_tracks in tqdm(enumerate(tracks['player'])):
        for player_id,track in player_tracks.items():
            team=team_assigner.get_team(video_frames[frame_num],track['bbox'],player_id)

            tracks['player'][frame_num][player_id]['team']=int(team)
            tracks['player'][frame_num][player_id]['team_colour']=team_assigner.team_colours[int(team)]


    # Assign ball to player
    player_ball_assigner=PlayerBallAssigner()
    team_ball_control=[]
    for frame_num,player_tracks in tqdm(enumerate(tracks['player'])):
        ball_bbox=tracks['ball'][frame_num][1]['bbox']
        assigned_player=player_ball_assigner.assign_ball_to_player(player_tracks,ball_bbox)

        if assigned_player != -1:
            tracks['player'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['player'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(-1)
    team_ball_control=np.array(team_ball_control)
    output_video=tracker.draw_annotations(video_frames,tracks,team_ball_control)
    #save video
    save_video(output_video, 'output_video/output.avi')

if __name__ == "__main__":
    main()