from logging import Logger

import cv2
import numpy as np
from tqdm import trange

from src.config_and_utils.config import DemoPipelineConfig
from src.core.pipelines.predictor import Predictor


class DemoPipeline:
    MAX_ITERATION_TO_BREAK = 10000

    def __init__(self, logger: Logger, config: DemoPipelineConfig):
        self.logger = logger
        self.config = config

    def run(self):
        self.logger.info("Load model...")
        predictor = Predictor.with_preloaded_model(
            self.config.clod_size_threshold,
            self.config.model_config,
            self.config.transforms_config
        )
        self.logger.info("Start realtime processing...")
        self.realtime_processing(predictor)
        self.logger.info("Processing ended!")

    def realtime_processing(self, predictor: Predictor):
        video = cv2.VideoCapture(str(self.config.movie_path))
        output = None
        if self.config.save_video:
            size = self.config.transforms_config.image_size \
                if self.config.size_type == "model" \
                else self.config.source_size
            output = cv2.VideoWriter(
                str(self.config.save_video), cv2.VideoWriter_fourcc(*'XVID'),
                30 // (self.config.skip_frames + 1), size
            )

        for i in trange(self.MAX_ITERATION_TO_BREAK):
            if not video.isOpened():
                self.logger.warning("Video is not opened!")
                break
            one, frame = video.read()

            if not one:
                self.logger.info("All frames read!")
                break

            if i % (self.config.skip_frames + 1) != 0:
                continue

            mask, source_image = predictor.get_mask(frame)
            mask = mask[0, 1]
            if self.config.size_type == "source":
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                source_image = frame

            mask = cv2.cvtColor(255 - mask, cv2.COLOR_GRAY2RGB)
            mask[mask > 100] = 255
            mask[mask <= 100] = 0
            image = np.uint8(self.draw(predictor, mask, source_image))
            cv2.imshow("Frame", image)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break

            if self.config.save_video:
                output.write(image)

        if self.config.save_video:
            output.release()
        cv2.destroyAllWindows()

    def draw(self, predictor: Predictor, mask: np.uint8, img: np.uint8):
        img = img.copy()
        mask = mask.copy()
        bbox_list, labeled_array = predictor.get_bboxes_and_areas(mask)
        sizes, the_biggest = predictor.get_sizes(labeled_array, len(bbox_list))

        for i, bbox in enumerate(bbox_list):
            # extract the coordinates of the bounding box
            x1, y1, x2, y2 = bbox
            size = sizes[i]

            if size < self.config.clod_size_threshold:
                continue

            colors = self.config.biggest_color_config if i == the_biggest else self.config.usual_color_config
            drawing_size = self.config.biggest_text_bold if i == the_biggest else self.config.usual_text_bold

            # draw the bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), colors.bbox_color, drawing_size)

            # write the size near the bounding box
            cv2.putText(
                img, f"{size:10.3f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors.text_color, drawing_size
            )

        count = len([i for i in range(len(bbox_list)) if sizes[i] > self.config.clod_size_threshold])

        cv2.putText(
            img, f"Count: {count}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, self.config.usual_color_config.text_color, self.config.usual_text_bold
        )

        if self.config.draw_mask:
            mask[:, :, :1] = 0
            return img // 1.5 + img * mask // 5
        else:
            return img
