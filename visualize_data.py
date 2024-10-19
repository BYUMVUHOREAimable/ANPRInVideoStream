import ast
import cv2 as cv
import numpy as np
import pandas as pd

# read interpolated data
res = pd.read_csv('./test_interpolated.csv')

cap = cv.VideoCapture('demo_1.mp4')

# check if the video file is loaded correctly
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# load video to write
fourcc = cv.VideoWriter.fourcc(*'mp4v')
fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
output = cv.VideoWriter('./output.mp4', fourcc, fps, (width, height))

license_plate = {}

# identify the number plate with most confidence score for each car
for car_id in np.unique(res['car_id']):
    idx = res[(res['car_id'] == car_id)]['license_nmb_score'].idxmax()
    license_number = res.loc[idx, 'license_nmb']
    frame_number = res.loc[idx, 'frame_nmb']
    plate_bbox = res.loc[idx, 'plate_bbox']

    # remove spaces in license number
    license_number = license_number.replace(' ', '')

    # set video to the correct frame
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if not ret:
        print(f"Error: Could not read frame {frame_number}.")
        continue

    x1, y1, x2, y2 = ast.literal_eval(plate_bbox.replace(' ', ','))

    # crop and resize license plate area
    plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
    plate_crop = cv.resize(plate_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

    license_plate[car_id] = {
        'license_crop': plate_crop,
        'license_plate_nmb': license_number
    }

frame_nmb = 0
cap.set(cv.CAP_PROP_POS_FRAMES, 0)

# read frames
ret = True
while ret:
    ret, frame = cap.read()

    if ret:
        df = res[res['frame_nmb'] == frame_nmb]
        for row_idx in range(len(df)):
            # draw car bounding box
            x1car, y1car, x2car, y2car = map(int, ast.literal_eval(df.iloc[row_idx]['car_bbox'].replace(' ', ',')))
            cv.rectangle(frame, (x1car, y1car), (x2car, y2car), (0, 255, 0), 15)

            try:
                # draw license plate information
                license_crop = license_plate[df.iloc[row_idx]['car_id']]['license_crop']
                text = license_plate[df.iloc[row_idx]['car_id']]['license_plate_nmb']

                if text in ['0', '-1']:
                    continue

                # calculate text size
                (text_width, text_height), baseline = cv.getTextSize(
                    text,
                    cv.FONT_HERSHEY_SIMPLEX,
                    3,
                    10
                )

                # draw white background for license number
                frame[y1car - 50 - text_height:y1car - 10, x1car: x1car + text_width, :] = (255, 255, 255)

                # draw license plate text
                cv.putText(
                    frame,
                    text,
                    (x1car, y1car - 30),
                    cv.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 0, 0),
                    10
                )
            except Exception as e:
                print(f"Error drawing license plate: {e}")
                continue

        # write the frame to output
        output.write(frame)

        # resize for display (optional)
        frame_resized = cv.resize(frame, (1280, 729))
        cv.imshow('Frame', frame_resized)

        # exit if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    frame_nmb += 1

# release resources
output.release()
cap.release()
cv.destroyAllWindows()
