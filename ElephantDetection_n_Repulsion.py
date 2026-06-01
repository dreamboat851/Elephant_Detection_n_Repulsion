import argparse
import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import utils


def run(model, camera_id, width, height, num_threads):
    counter, fps = 0, 0
    start_time = time.time()
    fps_avg_frame_count = 10

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        sys.exit("ERROR: Could not open camera.")

    base_options = python.BaseOptions(model_asset_path=model)

    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        max_results=3,
        score_threshold=0.3,
    )

    detector = vision.ObjectDetector.create_from_options(options)

    while cap.isOpened():
        success, image = cap.read()

        if not success:
            sys.exit("ERROR: Unable to read from webcam.")

        counter += 1

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        detection_result = detector.detect(mp_image)

        image = utils.visualize(image, detection_result)

        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)
            start_time = time.time()

        cv2.putText(
            image,
            f"FPS = {fps:.1f}",
            (24, 20),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 255),
            1,
        )

        cv2.imshow("Elephant Detector", image)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EfficientDet_lite1.tflite")
    parser.add_argument("--cameraId", type=int, default=0)
    parser.add_argument("--frameWidth", type=int, default=640)
    parser.add_argument("--frameHeight", type=int, default=480)
    parser.add_argument("--numThreads", type=int, default=4)

    args = parser.parse_args()

    run(
        args.model,
        args.cameraId,
        args.frameWidth,
        args.frameHeight,
        args.numThreads,
    )


if __name__ == "__main__":
    main()
