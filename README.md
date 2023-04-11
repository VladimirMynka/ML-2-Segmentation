# ML-2-Segmentation
## Prepare dataset

To prepare dataset you should do next steps:

1. Mark up your video in Label Studio. Use polygon segmentation for doing it
2. Load your mark-up json from Label Studio into `data/dataset/markup/` folder.
3. Load your video which was marked up into `data/dataset/markup/` folder. \
We think that your frames that were used for marking up, can be preprocessed. 
So, on our mind, it's better to extract source frames directly from video
4. Set split_k for data splitting (train and validation parts) in `src/config_and_utils/config.py`
5. From root of this project, run
    ```
    python -m src.dataset_preparation.main run --movie_name=<your movie name>
    ```
   (or you can set movie_name in the config)
6. Your dataset will be named as current datetime `data/dataset/<datetime>`. 
It will consist of `train` and `val` parts with `x` and `y` folders