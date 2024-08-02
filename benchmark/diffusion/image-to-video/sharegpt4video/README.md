# ShareGPT4Video dataset

For the image-to-video task, we sample 100 video-caption pairs from the ShareGPT4Video datset to feed to the diffusion model to generate videos.

## Filtering the dataset

Download the dataset with captions and video paths.

```sh
wget https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video/resolve/main/sharegpt4video_40k.jsonl
```

Sample video-caption pairs.

```sh
python sample.py
```

Download and unzip the chunk of videos.The sampled dataset will be saved under `sharegpt4video_100.json`.

```sh
wget https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video/resolve/main/zip_folder/panda/panda_videos_1.zip
unzip panda_videos_1.zip -d panda
```

Extract the first frame of the video and save under `first_frame/`.

```sh
pip install opencv-python
python extract_first_frame.py
```
