import os
import json

import cv2


def main() -> None:
    os.makedirs("first_frame", exist_ok=True)

    for video_id in json.load(open("sharegpt4video_100.json"))["video_id"]:
        cap = cv2.VideoCapture(f"panda/{video_id}.mp4")
        ret, frame = cap.read()
        assert ret, f"failed to read first frame of video {video_id}"
        cv2.imwrite(f"first_frame/{video_id}.jpg", frame)
        cap.release()


if __name__ == "__main__":
    main()
