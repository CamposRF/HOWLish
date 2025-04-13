#HOWLish detection pipeline v.1.0.0.

import time
import datetime
import os
import re
import pydub 

import tensorflow as tf
import numpy as np
import pandas as pd

import vggish_input as vg

path_to_howlish_frozen =  #Add here path to HOWLish in frozen graph format

##Load function 
def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    layers = [op.name for op in import_graph.get_operations()]
    if print_graph == True:
        for layer in layers:
            print(layer)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

# Load frozen graph using TensorFlow 1.x functions
with tf.io.gfile.GFile(path_to_howlish_frozen, "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(f.read())

# Wrap frozen graph to ConcreteFunctions
frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                inputs=["x:0"],
                                outputs=["Identity:0"],
                                print_graph=False)

def my_func(img):
    img = 2 * ((img - np.min(img))/(np.max(img)-np.min(img))) - 1
    img = img.astype('float32').reshape(96,-1)
    img = img[np.newaxis, :, :, np.newaxis]
    img_tensor = tf.convert_to_tensor(img)
    prediction = frozen_func(x=img_tensor)
    return prediction[0].numpy()[0,0]

def moving_average(a, n=3): 
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

source_folder =     #Add path to the folder where all the .WAV files of the recorded soundscapes are
save_folder =       #Add path to the folder where you want all the output files to be saved
campaign_list = os.listdir(source_folder)

model_name = "HOWLish_v1.0.0" 
pipeline_name = "rollavg_v1.0.0" 
window_size = 3 # W in original manuscript
threshold = 0.90 # T in original manuscript
lenght_seconds = 110 #110 seconds = 1 minute and 50 seconds

for c in campaign_list:
    campaign_start_time = time.time()
    campaign = c
    
    files = os.listdir(os.path.join(source_folder,campaign))
    pattern = ".*WAV"  #Exclude txt
    files = [x for x in files if re.match(pattern, x)]
    
    save_path = os.path.join(save_folder,campaign)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(len(files)):
        examples = vg.wavfile_to_examples(os.path.join(source_folder,campaign,files[i]))

        predictions = np.apply_along_axis(my_func, axis=1, arr=examples.reshape(-1, 96*64))

        rollavg = moving_average(predictions, n = window_size)
        start = np.array(range(0, len(predictions) - window_size + 1))
        stop = np.array(range(window_size-1, len(predictions)))

        start = start[rollavg > threshold]
        stop = stop[rollavg > threshold]
        rollavg = rollavg[rollavg > threshold]

        if len(rollavg) > 0: 

            lenght_campaign_seconds = (len(examples) + 1) * 0.96

            start_seconds = start * 0.96
            stop_seconds = (stop + 1) * 0.96 
            
            diff_seconds = stop_seconds - start_seconds
            add_seconds = lenght_seconds - diff_seconds
            add_seconds = add_seconds/2

            start_seconds = start_seconds - add_seconds
            stop_seconds = stop_seconds + add_seconds

            for j in range(len(rollavg)):
                if start_seconds[j] < 0:

                    stop_seconds[j] = stop_seconds[j] + abs(start_seconds[j])
                    start_seconds[j] = 0

                elif stop_seconds[j] > lenght_campaign_seconds:

                    start_seconds[j] = start_seconds[j] - (stop_seconds[j] - lenght_campaign_seconds)
                    stop_seconds[j] = lenght_campaign_seconds

            selected_clips = pd.DataFrame({'start': start, 'stop': stop, 'rollavg': rollavg, 'start_seconds': start_seconds, 'stop_seconds': stop_seconds})
            selected_clips["drop"] = 0

            for j in range(len(selected_clips)):
                if selected_clips["drop"][j] == 0:

                    selected_clips["ground_start"] = selected_clips["start_seconds"][j]     
                    selected_clips["ground_stop"] = selected_clips["stop_seconds"][j]       
                    selected_clips["intersection_seconds"] = selected_clips[["stop_seconds", "ground_stop"]].min(axis=1) - selected_clips[["start_seconds", "ground_start"]].max(axis=1) 

                    for k in range(len(selected_clips)):
                        if (selected_clips["intersection_seconds"][k] > 55) and (j != k):
                            if rollavg[j] >= rollavg[k]:
                                selected_clips.loc[k,"drop"] = 1

            selected_clips.drop(selected_clips[selected_clips["drop"] == 1].index, inplace = True)
            selected_clips = selected_clips.reset_index(drop=True)

            source_wav = pydub.AudioSegment.from_wav(os.path.join(source_folder,campaign,files[i]))

            for k in range(len(selected_clips)):
                b = selected_clips["start_seconds"][k] * 1000 #works in miliseconds
                e = selected_clips["stop_seconds"][k]* 1000
                e = np.min([len(source_wav), e])
                temp_clip = source_wav[b:e]

                clip_minute = str(datetime.timedelta(seconds= selected_clips["start_seconds"][k])).split(':')[-2].zfill(2)
                clip_second = str(datetime.timedelta(seconds= selected_clips["start_seconds"][k])).split(':')[-1][:2].zfill(2)
                time_in_code = clip_minute + clip_second

                temp_clip.export(os.path.join(save_path, files[i][:-4] + "_" + time_in_code + "_" + str(int(selected_clips["rollavg"][k]*10000))  + ".wav"), format="wav", tags=None) 
    
    campaign_stop_time = time.time()
    campaign_GPU_time = campaign_stop_time - campaign_start_time

    GPU_time_log = pd.DataFrame({'campaign': c, 'start': campaign_start_time, 'stop': campaign_stop_time, 'GPU_time_seconds': campaign_GPU_time, 
                                'model_name':model_name, 'pipeline_name' : pipeline_name, 'window_size' : window_size, 'threshold':threshold }, index=[0])
    GPU_time_log.to_csv(os.path.join(save_path,campaign + "_log.csv"), index=False)
    
    print(c + " finished in " + str(round(campaign_GPU_time/60, ndigits=0)) + " minutes! \õ/")

