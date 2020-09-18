# -*- coding: utf-8 -*-

import cv2
import pytesseract as pt
import pandas as pd
import json
from datetime import date
import re
import requests
import os
import numpy as np
import sys
import datetime
from datetime import datetime
import time
import os
from moviepy.editor import *
import operator
import librosa
import requests
import jwt
from difflib import SequenceMatcher
from client import ZoomClient
from zoom import ZoomSessionClient
from pathlib import Path
from resemblyzer import VoiceEncoder
from audio import preprocess_wav
from resemblyzer.hparams import *
from image_transforms import get_grayscale,erode,binarization,crop_img_left_buttom,crop_img_centre

class ActiveSpeaker(object):
    def __init__(self, zoom):
        self.zoom = zoom

    def process_meetings(self, email, password, save_dir, api_key, api_secret):
        meetings = self.zoom.get_meetings()
        meetings = self.zoom.filter_meetings(meetings)
        
        self.outJson = []

        for i,meeting in enumerate(meetings):
            print("**Meeting id # :" + str(i) + "**************************YURII**********************") # new code
            self._process_meeting(meeting, save_dir, api_key, api_secret, email, password)
            
        out_data = '2020-07-02_speaker.json'
        
        with open(out_data, 'w') as outfile:
            json.dump(self.outJson, outfile)
            
    def _process_meeting(self, meeting, save_dir, api_key, api_secret, email, password):
        # Get 'Speaker View' version of video
        recording_files = filter(
            lambda x: x.get("recording_type") == "shared_screen_with_speaker_view",
            meeting.get('recording_files', [])
        )
        #print("******recording filed"+str(recording_files))
        for i, video_data in enumerate(recording_files):
            
            meeting_start = video_data.get("recording_start")
            meeting_end = video_data.get("recording_end")
            meeting_id = video_data.get("meeting_id")
            rid = video_data.get('id')
            
            prefix = '_speaker_'+rid[0:5]
            filename = self.zoom._get_output_filename(meeting, prefix)
            save_path = self.zoom._get_output_path(filename, save_dir)

            self.client = ZoomClient(api_key, api_secret)
            query_params = {'meeting_id': meeting_id}
    
            participant_resp = json.loads(self.client.report.get_meeting_participant_report(**query_params).content)
        
            self.attendees = self._extract_values(participant_resp,'name')
            self.attendees_email = self._extract_values(participant_resp,'user_email')
            self.attendees_join_time = self._extract_values(participant_resp,'join_time')
            self.attendees_leave_time = self._extract_values(participant_resp,'leave_time')
            
            # Create a list of regex expressions for each attendee
            self.attendee_list = [re.compile(att+'*') for att in self.attendees]
          
            self.session_client = ZoomSessionClient(api_key, api_secret)

            session, response = self.session_client.session(email, password)
            if response.status_code != 200:
                session.close()
                return

            if self.zoom._real_download_file(session, video_data.get('download_url'),save_path):
                print('Downloaded the file: {}'.format(save_path))
            else:
                print("Error while downloading: {}".format(video_data.get('download_url')))
            
            
            
            #Website Optimization_speaker_4a001
            print("meeting id   ",str(rid))
            # OCR for Active Speaker
            activeSpeakerDF = self._process_video_frames(save_path)
            
            # Get Utterances for Speakers
            utter = self._get_utterances(activeSpeakerDF)
            fpath_ocr = '/Users/sapnasharma/Documents/Speaker-diarization-master/vid/OCR' + str(rid[0:5])+".csv"
            activeSpeakerDF.to_csv(fpath_ocr)
            print("After OCR ") # new code
            #print(activeSpeakerDF['Speaker'].value_counts()) # new code
            samples = len(activeSpeakerDF) # new code 
            print ("Total samples",samples)
            try:
                speakers_unknown_OCR = activeSpeakerDF['Speaker'].value_counts().to_dict()["<unknown>"] # new code
            except:
                speakers_unknown_OCR = 0
            print("Total Samples :" + str(samples)) # new code
            print("Total unknowns " +str(speakers_unknown_OCR) )
            OCR_accuracy = (samples - speakers_unknown_OCR)/samples # new code
            print("OCR Accuracy :" + str(OCR_accuracy)) # new code
            
            
             
            
            # Convert Video to Audio
            audio_data = save_path.replace('.mp4', '.mp3')
            videoclip = VideoFileClip(save_path)
            videoclip.audio.write_audiofile(audio_data)
            
            # Remove video file
            #os.remove(save_path)
                   
          # Prep for embedding
            slices = self._get_longest_utterance(utter)
            segments = []
            speaker_names = []

            for slice in slices:
                seg = utter[slice]
                end = seg['EndTime'] / 1000
                
                # pass the last 10 seconds of the longest utterance to the embedding generator
                if seg['Duration'] / 1000 > 10.0:
                    beg = end - 10.0
                else:
                    beg = seg['StartTime'] / 1000


                speaker_names.append(slice.rsplit('_', 1)[0])
                segments.append([beg, end])
                
            wav = preprocess_wav(audio_data)

            speaker_wavs = [wav[int(s[0] * sampling_rate):int(s[1]) * sampling_rate] for s in segments]
            encoder = VoiceEncoder("cpu")
            print("Running the continuous embedding on cpu, this might take a while...")
            _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=0.625)
            # Get the continuous similarity for every speaker. It amounts to a dot product between the 
            # embedding of the speaker and the continuous embedding of the interview
            speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]

            similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in 
                               zip(speaker_names, speaker_embeds)} 
            
            # Remove the audio file
            os.remove(audio_data)
            
            allSimilarity = np.stack([value for key, value in similarity_dict.items()])
           
            # For unknown speaker time slices, update the speaker if the similarity GE 0.65
            for i, row in activeSpeakerDF.iterrows():
                if row['Speaker'] == '<unknown>':
                    sp = allSimilarity[:,i]
 
                    am = np.argmax(sp, axis=0)
                    if sp[am] >= .65:
                        activeSpeakerDF.at[i,'Speaker'] = speaker_names[am]
                        
            # Calculate speaking time in seconds for each speaker
            grouped = activeSpeakerDF.groupby('Speaker').count().reset_index()
            fpath_sim = '/Users/sapnasharma/Documents/Speaker-diarization-master/vid/SIM' + str(rid[0:5])+".csv"
            activeSpeakerDF.to_csv(fpath_sim)
            if speakers_unknown_OCR > 0: # new code
                print("After all similarity") # new code
                #print(activeSpeakerDF['Speaker'].value_counts()) # new code
                speakers_unknown_sim = activeSpeakerDF['Speaker'].value_counts().to_dict()["<unknown>"] # new code
                SIM_accuracy = (speakers_unknown_OCR - speakers_unknown_sim)/speakers_unknown_OCR # new code
                print("SIM Accuracy :" + str(SIM_accuracy)) # new code
                Overall_accuracy = (samples - speakers_unknown_sim)/samples # new code
                print("Overall Accuracy : " + str(Overall_accuracy)) # new code
            else: # new code
                print(" None ") # new code

            speaker_time = []

            for spkr in speaker_names:
                for i, row in grouped.iterrows():
                    if row['Speaker'] == spkr:
                        secs =  time.strftime('%H:%M:%S', time.gmtime(row['Millisecond'] * 1.6 ))
                        speaker_time.append(secs)
           
            # Time Late
            self.attendee_late = []
            meeting_start_t = datetime.strptime(meeting_start,"%Y-%m-%dT%H:%M:%SZ")
            for st in self.attendees_join_time:
                jt = datetime.strptime(st,"%Y-%m-%dT%H:%M:%SZ")
                late_secs = max(0,(jt - meeting_start_t).total_seconds())
                secs =  time.strftime('%H:%M:%S', time.gmtime(late_secs))
                self.attendee_late.append(secs)
            
            #Create output JSON
            speakers_time = []
            for n in self.attendees:
                spkr_time = ''
                for i, s in enumerate(speaker_names):  
                    if s == n:
                        spkr_time = speaker_time[i]
                        
                if spkr_time == '':
                    spkr_time = '00:00:00'
                    
                speakers_time.append(spkr_time)
                
            speakers_summary = []
            for n,t,e,st in zip(self.attendees,self.attendee_late, self.attendees_email, speakers_time):
                spkr = {'name': n, 'email': e, 'speaking_time': st, 'late_meeting_join': t}
                speakers_summary.append(spkr)
                
            out = {
              "name": meeting.get("topic"),
              "meeting_id": meeting_id,
              "rid": rid,
              "meeting_start": meeting_start,
              "meeting_end": meeting_end,
              "conversation" : utter,
              "attendees": speakers_summary
            }
            
            self.outJson.append(out)
              
    def _get_longest_utterance(self, utter):
        longest_utter = []
        for attendee in self.attendee_list:
            speaker_dict =  {key: value['Duration'] for key, value in utter.items() if re.match(attendee, key)}
            if bool(speaker_dict):
                longest = max(speaker_dict.items(), key=operator.itemgetter(1))[0]
                longest_utter.append(longest)

        return longest_utter
            
    def _get_utterances(self, activeSpeakerDF):
        # Get Utterances for each Speaker
        prevSpeaker = ''
        startTime = 0.0
        endTime = 0.0
        utter = {}
        cnt = 1

        for index, row in activeSpeakerDF.iterrows():
            if index == 0:
                prevSpeaker = row["Speaker"]
                startTime = endTime = row["Millisecond"]        

            if prevSpeaker != row["Speaker"]:
                utterTime = {'StartTime': startTime, 'EndTime': endTime, 'Duration': endTime - startTime}
                utter[prevSpeaker + '_' + str(cnt)] = utterTime
                
                prevSpeaker = row["Speaker"]
                startTime = row["Millisecond"]
                cnt = cnt + 1

            endTime = row["Millisecond"]

        utterTime = {'StartTime': startTime, 'EndTime': endTime, 'Duration': endTime - startTime}
        utter[prevSpeaker + '_' + str(cnt)] = utterTime

        return utter
 
    def _process_video_frames(self, save_path):

        # Video Capture Using OpenCV
        cap = cv2.VideoCapture(save_path)
        frame_cnt = int(cap.get(cv2.cv2.CAP_PROP_FRAME_COUNT))

        # To capture frames every 500ms, incremented after every 500ms
        index = 1

        # Pandas Dataframe for EDA
        activeSpeakerDF = pd.DataFrame(columns=['Speaker', 'Millisecond'])

        for x in range(frame_cnt):
            ret,frame = cap.read()
            if not ret:
                break
 

            
            
            #Get frame timestamp
            frame_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

            if frame_timestamp >= (index * 1600.0):

                speaker = self._get_active_speaker(x, frame, False)

                # Add row to active speaker dataframe
                activeSpeakerDF = activeSpeakerDF.append({'Speaker': speaker, 'Millisecond': frame_timestamp},
                     ignore_index=True)

                # Increment index
                index = index + 1
               
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
            
            #if frame_timestamp > 300000.0:
            #    break

        cap.release()
        cv2.destroyAllWindows()
        
        return activeSpeakerDF
    
    # Utility to search response JSON
    def _extract_values(self, obj, key):
        """Pull all values of specified key from nested JSON."""
        arr = []

        def extract(obj, arr, key):
            """Recursively search for values of key in JSON tree."""
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, (dict, list)):
                        extract(v, arr, key)
                    elif k == key:
                        arr.append(v)
            elif isinstance(obj, list):
                for item in obj:
                    extract(item, arr, key)
            return arr

        results = extract(obj, arr, key)
        return results
    '''
    # Apply different sets of transforms according to the image type (video or white text with black background)
    def _process_img(self,img, img_mean_value_threshold=3.6):
        # makes the image black and white
        preprocessed_img = binarization(img)
        
        # if img mean pixel values among all the chanels is bigger that 3.6
        if img.mean() > img_mean_value_threshold:
            # The img contains video
            # make black text thicker
            preprocessed_img = erode(preprocessed_img)
            # crop the image to left buttom corner, 0.21 - x percentage, 0.94 - y percentage
            preprocessed_img = crop_img_left_buttom(preprocessed_img,0.21,0.94)
        else:
            # The img contains black background with name
            # crop the image to the centre, 0.75 - crop ratio
            preprocessed_img = crop_img_centre(preprocessed_img,0.75)
            
        return preprocessed_img
    '''
    # Apply different sets of transforms according to the image type (video or white text with black background)
    def _process_img(self,img, img_mean_value_threshold=20):
        # makes the image black and white
        preprocessed_img = binarization(img)
        
        # if img mean pixel values among all the chanels is bigger that 3.6
        if img.mean() > img_mean_value_threshold:
            # The img contains video
            # make black text thicker
            #preprocessed_img = erode(preprocessed_img)
            # crop the image to left buttom corner, 0.21 - x percentage, 0.94 - y percentage
            preprocessed_img = crop_img_left_buttom(preprocessed_img,0.21,0.94)
        else:
            # The img contains black background with name
            # crop the image to the centre, 0.75 - crop ratio
            preprocessed_img = crop_img_centre(preprocessed_img,0.75)
            
        return preprocessed_img

    # Perform OCR on Frame
    def _get_active_speaker(self, fno, frame, video_on):
        
        #bottom
        text_bottom = pt.image_to_string(crop_img_left_buttom(binarization(frame),0.21,0.94))
        print ("text bottom without pre processing", text_bottom[0:20])
        
        text_centre = pt.image_to_string(crop_img_centre(binarization(frame), 0.75))
        print ("text centre without pre processing", text_centre[0:20])
        
        text_noprocessing = pt.image_to_string(frame,lang = 'eng')
        print ("text without preprocessing",(text_noprocessing[0:20]))
        
      
        # preprocess the frame
        processed_frame = self._process_img(frame)
        # run optical text recognition on the frame
        text = pt.image_to_string(processed_frame)
        #print(text)
        unk = 0
        speaker = ''
        for attendee in self.attendees:
            # if predicted text is 85% similiar to an attendee name
            if self._get_words_similarity(attendee,text)>0.85:
                speaker = attendee
                break
            '''
            if not speaker:  # code for printing the unknown
                filename = str(fno)+"unknown"+ str(unk) + ".jpg" # code for printing the unknown
                unk+=1 # code for printing the unknown
                path = '/Users/sapnasharma/Documents/Speaker-diarization-master/vid/' # code for printing the unknown
                cv2.imwrite(os.path.join(path , filename), frame) # code for printing the unknown
            '''   
                
        print("Speaker",speaker)        
        return speaker if speaker else '<unknown>'


    def _get_words_similarity(self,word1,word2):
        return SequenceMatcher(None,word1,word2).ratio()
