import cv2
import os

_TEXT_COLOR = (0, 0, 255)
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_MARGIN = 10
_ROW_SIZE = 20

sound_enabled = False
detected_sound = None

try:
    from pygame import mixer

    if os.path.exists("lion_sound.ogg"):
        mixer.init()
        detected_sound = mixer.Sound("lion_sound.ogg")
        sound_enabled = True
    else:
        print("Warning: lion_sound.ogg not found. Running without sound.")

except Exception as e:
    print("Warning: sound disabled:", e)


def visualize(image, detection_result):
    for detection in detection_result.detections:
        bbox = detection.bounding_box

        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)

        category = detection.categories[0]
        class_name = category.category_name
        probability = round(category.score, 2)

        result_text = f"{class_name} ({probability})"

        text_location = (
            bbox.origin_x + _MARGIN,
            bbox.origin_y + _MARGIN + _ROW_SIZE,
        )

        if probability >= 0.8:
            cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)
            cv2.putText(
                image,
                result_text,
                text_location,
                cv2.FONT_HERSHEY_PLAIN,
                _FONT_SIZE,
                _TEXT_COLOR,
                _FONT_THICKNESS,
            )

            if sound_enabled and detected_sound is not None:
                detected_sound.play(maxtime=2000)

    return image
