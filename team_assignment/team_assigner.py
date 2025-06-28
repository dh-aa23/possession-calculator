from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colours={}
        self.kmeans=None
        self.player_teams={}

    def get_model(self,image):
        image_2d=image.reshape((-1,3))
        kmeans=KMeans(n_clusters=2,init='k-means++',n_init=1).fit(image_2d)
        return kmeans

    def get_player_colour(self,frame,bbox):
        image=frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        top_half_image=image[0:int(image.shape[0]/2),:]
        #get kmeans model for top half of image
        k_means=self.get_model(top_half_image)
        #get the cluster labels
        labels=k_means.labels_
        #reshape the labels to the original image shape
        clustered_img=labels.reshape(top_half_image.shape[0],top_half_image.shape[1])
        #get the PLAYER CLUSTER 
        corner_cluster=[clustered_img[0,0],clustered_img[0,-1],clustered_img[-1,0],clustered_img[-1,-1]] 
        non_player_cluster=max(set(corner_cluster), key=corner_cluster.count)
        player_cluster=1-non_player_cluster
        return k_means.cluster_centers_[player_cluster]



    def assign_teams(self, frame,player_detections):
        player_colours=[]
        for _,player_detection in  player_detections.items():
            bbox=player_detection['bbox']
            player_colour=self.get_player_colour(frame,bbox)
            player_colours.append(player_colour)
        kmeans=KMeans(n_clusters=2,init='k-means++',n_init=1).fit(player_colours)

        self.kmeans=kmeans

        self.team_colours[1]=kmeans.cluster_centers_[0]
        self.team_colours[2]=kmeans.cluster_centers_[1]


    def get_team(self,frame,player_bbox,player_id):
        if player_id in self.player_teams:
            return self.player_teams[player_id]
        player_colour=self.get_player_colour(frame,player_bbox)
        team_id=self.kmeans.predict(player_colour.reshape(1,-1))[0]
        team_id+=1 # team id starts from 0
        self.player_teams[player_id]=team_id

        return team_id
    
    def get_team_colour(self,tracks,video_frames):
        for frame_num,player_tracks in enumerate(tracks['player']):
            for player_id,track in player_tracks.items():
                team=self.get_team(video_frames[frame_num],track['bbox'],player_id)

                tracks['player'][frame_num][player_id]['team']=int(team)
                tracks['player'][frame_num][player_id]['team_colour']=self.team_colours[int(team)]
        return tracks


