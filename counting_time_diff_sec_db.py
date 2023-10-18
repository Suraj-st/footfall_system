import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import math
from utils.cptracking import CPTracker
from utils.objtrcking import ObjTracker
import csv
import psycopg2
from datetime import datetime, timedelta

confThreshold = 0.6
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416

# Add your PostgreSQL database connection details here
db_host = "localhost"
db_name = "ff_system"
db_user = "postgres"
db_password = "Tharakast@4123"

parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
parser.add_argument('--video', default='ed3.mp4', help='Path to video file.')
args = parser.parse_args()

# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Load weight and cfg files
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

# Load serialized model from disk
print("[INFO] loading model...")
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_OPENCL)

# Initialize the video writer
writer = None

# Initialize the frame dimensions
W = None
H = None

# Instantiate cp tracker
cpt = CPTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalDown = 0
totalUp = 0

# Define variables for tracking counts and positions
count_in = 0
count_out = 0
positions_in = {}
positions_out = {}

# Create a dictionary to store entry times for each object position
entry_times = {}

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]


def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    rects = []

    # Scan through all the bounding boxes output from the network and keep only the ones with high confidence scores.
    # Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Remove the bounding boxes with low confidence using non-maxima suppression
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        # Class "person"
        if classIds[i] == 0:
            rects.append((left, top, left + width, top + height))
            # Use the cp tracker to associate the old object cps with the newly computed object cps
            objects = cpt.update(rects)
            counting(objects)


def counting(objects):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    global totalDown
    global totalUp

    # Loop over the tracked objects
    for (objectID, cp) in objects.items():
        # Check if a trackable object exists for the current object ID
        to = trackableObjects.get(objectID, None)

        # If there is no existing trackable object, create one
        if to is None:
            to = ObjTracker(objectID, cp)

        # If there is a trackable object, utilize it to determine direction
        else:
            # capture the direction of the object is moving (negative for 'up' and positive for 'down')
            y = [c[1] for c in to.cps]
            direction = cp[1] - np.mean(y)
            print(direction)
            to.cps.append(cp)

            # Check if the object has been counted or not
            if not to.counted:
                if direction < 0 and cp[1] in range(frameHeight // 2 - 30, frameHeight // 2 + 30):
                    totalUp += 1
                    to.counted = True

                    # Record the position of the person entering
                    positions_in[objectID] = totalUp
                    entry_times[totalUp] = datetime.now()  # Store the entry time

                    # Write object details to CSV file
                    crowd = totalUp - totalDown
                    write_object_to_postgresql(objectID, "In", crowd, totalUp)

                elif direction > 0 and cp[1] in range(frameHeight // 2 - 30, frameHeight // 2 + 30):
                    totalDown += 1
                    to.counted = True

                    # Record the position of the person exiting
                    positions_out[objectID] = totalDown

                    # Calculate the time difference between IN and OUT for the same position
                    entry_time = entry_times.get(totalDown)
                    if entry_time is not None:
                        exit_time = datetime.now()
                        time_difference = exit_time - entry_time
                        entry_times.pop(totalDown)  # Remove the entry time
                    else:
                        time_difference = timedelta(seconds=0)  # Default to zero if entry time not found

                    # Write object details to CSV file, including time difference
                    crowd = totalUp - totalDown
                    write_object_to_postgresql(objectID, "Out", crowd, totalDown, time_difference)

        # Store the trackable object in to dictionary
        trackableObjects[objectID] = to
        # Draw both the ID of the object and the cp of the object on the output frame
        cv.circle(frame, (cp[0], cp[1]), 4, (0, 255, 0), -1)

    # Update the total crowd inside
    crowd = totalUp-totalDown

    # Construct a tuple of information to display on the frame
    info = [
        ("IN", totalUp),
        ("OUT", totalDown),
        ("CROWD", crowd)
    ]

    # Loop over the info tuples and draw them on the frame
    for (i, (k, v)) in enumerate(info):
        text = "{}".format(v)
        if k == 'IN':
            cv.putText(frame, f'IN : {text}', (10, 55), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif k == 'OUT':
            cv.putText(frame, f'OUT : {text}', (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif k == 'CROWD':
            cv.putText(frame, f'CROWD : {text}', (10, 105), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Draw the Central line in middle of the frame
    cv.line(frame, (0, frameHeight // 2), (frameWidth, frameHeight // 2), (0, 255, 255), 2)

# Write object details to a CSV file
# def write_object_to_csv(objectID, direction, crowd):
#     # Check if the CSV file exists
#     file_exists = os.path.isfile('object_details.csv')
#
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Format timestamp as 'YYYY-MM-DD HH:MM:SS'
#
#     with open('object_details.csv', mode='a', newline='') as file:
#         writer = csv.writer(file)
#
#         # Write column headers if the file is newly created
#         if not file_exists:
#             writer.writerow(['Timestamp', 'ObjectID', 'Direction', 'Crowd'])
#
#         writer.writerow([timestamp, objectID, direction, crowd])

# Write object details to a CSV file
# Modify the write_object_to_csv function to include positions
# Modify the write_object_to_csv function to include positions and time difference
def write_object_to_postgresql(objectID, direction, crowd, position, time_difference=None):
    try:
        connection = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password
        )

        cursor = connection.cursor()

        current_datetime = datetime.now()
        date = current_datetime.strftime('%Y-%m-%d')
        time = current_datetime.strftime('%H:%M:%S')

        entry_exit = positions_in.get(objectID, positions_out.get(objectID, 0))
        store_area = "A_01"

        if time_difference is None:
            time_difference_str = ""  # Leave the time difference empty if not provided
            time_difference_sec = ""  # Leave the time difference in seconds empty if not provided
            frames_num = ""  # Leave the frame number empty if not provided
            actual_time = ""  # Leave the actual time empty if not provided
        else:
            # Format the time difference as a string (e.g., "0 days 00:02:34")
            time_difference_str = str(time_difference)

            # Calculate the time difference in seconds
            time_difference_sec = time_difference.total_seconds()

            # Calculate the number of frames
            frames_num = time_difference_sec * 1.35

            # Calculate the actual dwel time
            actual_time = frames_num / 30

        insert_query = """
              INSERT INTO main_entrance (date, time, object_id, direction, crowd, position, entry_exit, time_difference_str, time_difference_sec, frames_num, actual_time, store_area)
              VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
              """
        record_to_insert = (date, time, objectID, direction, crowd, position, entry_exit, time_difference_str, time_difference_sec, frames_num, actual_time, store_area)

        cursor.execute(insert_query, record_to_insert)

        connection.commit()
        cursor.close()
        connection.close()
        print("Record inserted successfully into the PostgreSQL database.")

    except (Exception, psycopg2.Error) as error:
        print("Error while inserting record into the PostgreSQL database:", error)


    #     # Insert the record into the database
    #     cursor.execute(
    #         "INSERT INTO main_entrance (date, time, object_id, direction, crowd, position, entry_exit, time_difference_min, time_difference_sec, frame_num, actual_time) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
    #         (date, time, objectID, direction, crowd, position, entry_exit, time_difference_str, time_difference_sec, frames_num, actual_time)
    #     )
    #
    #     connection.commit()
    #     cursor.close()
    #
    # except (Exception, psycopg2.Error) as error:
    #     print("Error while inserting data into PostgreSQL:", error)
    #
    # finally:
    #     if connection:
    #         connection.close()

# Open the video file
if not os.path.isfile(args.video):
    print("Input video file ", args.video, " doesn't exist")
    sys.exit(1)
cap = cv.VideoCapture(args.video)

# Process inputs
winName = 'Customer Counting System'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

# Create a video writer for the output video
video_name = args.video.split('/')[-1].split('.')[0]
output_video_path = f'{video_name}_output.avi'
video_writer = cv.VideoWriter(output_video_path, cv.VideoWriter_fourcc(*"MJPG"), 30.0,
                              (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:
    # Get frame from the video
    hasFrame, frame = cap.read()
    # frameHeight = frame.shape[0]
    # frameWidth = frame.shape[1]
    # cv.line(frame, (0, frameHeight // 2), (frameWidth, frameHeight // 2), (0, 255, 255), 2)

    # Stop the program if reached end of video
    if not hasFrame:
        print("Processing video file is complete.")
        print("Output video file is saved as", output_video_path)
        break

    # Create the blob from a frame
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Set the input blob for the network
    net.setInput(blob)

    # Run the forward pass to get output from the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    # Write the frame with the detection boxes to video file
    video_writer.write(frame)

    # Display the resulting frame
    cv.imshow(winName, frame)

# Release resources
cap.release()
video_writer.release()
cv.destroyAllWindows()
