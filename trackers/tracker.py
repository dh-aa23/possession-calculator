from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import sys
import cv2
import pandas as pd
from team_assignment import TeamAssigner
sys.path.append('../')
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_centre,get_width
class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker=sv.ByteTrack()

    def detect_frames(self,frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_detections = self.model.predict(batch_frames,conf=0.1)
            detections= detections + batch_detections
            # print(batch_detections)
        return detections

    def get_object_tracks(self,frames):


        detections=self.detect_frames(frames)

        tracks={
            "player": [],
            "ball": [],
            "referee": [],
        }
        for frame_num,detections in enumerate(detections):
            cls_names = detections.names
            cls_names_inv={v:k for k,v in cls_names.items()}

            #convert detections to supervision format
            detections_supervision = sv.Detections.from_ultralytics(detections)  

            # print (detections_supervision.class_id)
            #convert goalkeeper to player
            for object_ind,class_id in enumerate(detections_supervision.class_id):
                if cls_names[class_id] == 'goalkeeper':
                    detections_supervision.class_id[object_ind]=cls_names_inv['player']

            #track objects
            detections_with_tracks=self.tracker.update_with_detections(detections_supervision)
            tracks["player"].append({})
            tracks["ball"].append({})
            tracks["referee"].append({})

            for from_detection in detections_with_tracks:
                bbox = from_detection[0].tolist()
                cls_id = from_detection[3]
                track_id = from_detection[4]
                
                if cls_id==cls_names_inv['player']:
                    tracks["player"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id==cls_names_inv['referee']:
                    tracks["referee"][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detections_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}


        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2=int(bbox[3])
        x_centre,_=get_centre(bbox)
        width=get_width(bbox)

        cv2.ellipse( 
            frame,
            center=(x_centre,y2),
            axes = (int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )

        rectangle_width=40
        rectangle_height=20
        x1_rectangle=x_centre-rectangle_width//2
        x2_rectangle=x_centre+rectangle_width//2
        y1_rectangle=(y2-rectangle_height//2)+15
        y2_rectangle=(y2+rectangle_height//2)+15

        if track_id is not None:
            cv2.rectangle(frame, 
                         (int(x1_rectangle), int(y1_rectangle)),
                         (int(x2_rectangle), int(y2_rectangle)),
                         color,
                         cv2.FILLED)
            x1_text=x1_rectangle+12
            if x1_text>=100:
                x1_text-=10

            cv2.putText(
                frame,
                str(track_id),
                (int(x1_text), int(y1_rectangle+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0,0),
                2,
                
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_centre(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        #draw graphic
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255, 255, 255), -1)
        alpha = 0.4  # Transparency factor
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame=team_ball_control[:frame_num+1]
        team1 = np.sum(team_ball_control_till_frame == 1) / len(team_ball_control_till_frame)
        team2 = np.sum(team_ball_control_till_frame == 2) / len(team_ball_control_till_frame)

        cv2.putText(frame, f"Team 1 ball control: {team1*100:.2f}", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 ball control: {team2*100:.2f}", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        return frame  

    def draw_annotations(self,video_frames,tracks,team_ball_control):
        output_video = []
        prev=-1
        for i in range(len(team_ball_control)):
            if team_ball_control[i]!=-1:
                prev=team_ball_control[i]
            else:
                team_ball_control[i]=prev

        for frame_num,frame in enumerate(video_frames):
            frame = frame.copy()
            player_dict= tracks["player"][frame_num]
            referee_dict= tracks["referee"][frame_num]
            ball_dict= tracks["ball"][frame_num]

            #draw players
            for track_id,player in player_dict.items():
                color=player.get("team_colour",(0,0,255))
                frame=self.draw_ellipse(frame,player["bbox"],color,track_id)

                if player.get("has_ball",False):
                    frame=self.draw_traingle(frame,player["bbox"],(0,0,255))
                    
            #draw referees
            for _,referee in referee_dict.items():
                frame=self.draw_ellipse(frame,referee["bbox"],(255,0,0))
                
            #draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))

            #write team ball control
            frame=self.draw_team_ball_control(frame,frame_num,team_ball_control)

            output_video.append(frame)  

        return output_video
    
    def interpolate_ball(self,ball_positions):
        ball_positions=[x.get(1,{}).get("bbox",[]) for x in ball_positions]
        df_ball=pd.DataFrame(ball_positions,columns=["x1","y1","x2","y2"])

        #Interpolate missing values in the ball positions
        df_ball=df_ball.interpolate()
        df_ball=df_ball.bfill()

        ball_positions=[{1:{'bbox':x}}for x in df_ball.to_numpy().tolist()]
        return ball_positions
    
    def team_assigner(self,tracks,frames):
        tracks['ball'] = self.interpolate_ball(tracks['ball'])
        team_assigner=TeamAssigner()
        team_assigner.assign_teams(frames[0],tracks['player'][0]) 

        for frame_num,player_tracks in enumerate(tracks['player']):
            for player_id,track in player_tracks.items():
                team=team_assigner.get_team(frames[frame_num],track['bbox'],player_id)

                tracks['player'][frame_num][player_id]['team']=int(team)
                tracks['player'][frame_num][player_id]['team_colour']=team_assigner.team_colours[int(team)]
        return tracks
