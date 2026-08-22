from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from egg_companion.config import VisionConfig
from egg_companion.core.behavior import classify_pose
from egg_companion.core.gaze import classify_gaze
from egg_companion.core.orientation import select_rotation, upright_pose_score
from egg_companion.models import BoundingBox, Detection


class FaceCrop:
    def __init__(
        self,
        image: np.ndarray,
        confidence: float,
        kind: str = "face",
        face_embedding: np.ndarray | None = None,
    ) -> None:
        self.image = image
        self.confidence = confidence
        self.kind = kind
        self.face_embedding = face_embedding


class SegmentedObject:
    def __init__(self, image: np.ndarray, mask: np.ndarray, confidence: float) -> None:
        self.image = image
        self.mask = mask
        self.confidence = confidence


class VisionEngine:
    def __init__(self, config: VisionConfig) -> None:
        try:
            import open_clip
            import torch
            from ultralytics import SAM, YOLO, YOLOE
        except ImportError as error:
            raise RuntimeError("vision dependencies are missing; install the project on the Jetson") from error
        self.config = config
        self._torch = torch
        detector_path = Path(config.detector_model)
        if not detector_path.is_file():
            raise RuntimeError(f"instance-segmentation model is unavailable: {detector_path}")
        self._detector = YOLOE(str(detector_path))
        if len(self._detector.names) < config.minimum_detector_classes:
            raise RuntimeError(
                f"segmentation vocabulary has {len(self._detector.names)} classes; "
                f"expected at least {config.minimum_detector_classes}"
            )
        pose_path = Path(config.pose_model)
        if not pose_path.is_file():
            raise RuntimeError(f"pose model is unavailable: {pose_path}")
        self._pose_model = YOLO(str(pose_path))
        self._sam = SAM(config.sam_model)
        self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
            config.clip_model, pretrained=config.clip_pretrained, device=config.device
        )
        self._clip_tokenizer = open_clip.get_tokenizer(config.clip_model)
        import cv2

        cascade_dir = getattr(getattr(cv2, "data", None), "haarcascades", "/usr/share/opencv4/haarcascades")
        cascade_path = Path(cascade_dir) / "haarcascade_frontalface_default.xml"
        self._face_detector = cv2.CascadeClassifier(str(cascade_path))
        self._profile_face_detector = cv2.CascadeClassifier(str(Path(cascade_dir) / "haarcascade_profileface.xml"))
        if self._face_detector.empty() or self._profile_face_detector.empty():
            raise RuntimeError("OpenCV frontal-face cascade is unavailable")
        yunet_path = Path(__file__).resolve().parents[2] / "models" / "face_detection_yunet_2023mar.onnx"
        if not yunet_path.is_file():
            raise RuntimeError(f"YuNet face model is unavailable: {yunet_path}")
        self._yunet_face_detector = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), 0.75, 0.3, 5000)
        sface_path = Path(config.sface_model)
        if not sface_path.is_file():
            raise RuntimeError(f"SFace recognition model is unavailable: {sface_path}")
        self._face_recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")
        self._inference_lock = threading.RLock()

    def analyze(
        self, frame: np.ndarray, include_pose: bool = True, include_semantics: bool = True
    ) -> tuple[tuple[Detection, ...], tuple[str, ...]]:
        frame_height, frame_width = frame.shape[:2]
        with self._inference_lock:
            results = self._detector(
                frame,
                device=self.config.device,
                conf=self.config.confidence_threshold,
                max_det=self.config.max_instances,
                verbose=False,
            )
            pose_results = (
                self._pose_model(frame, device=self.config.device, conf=self.config.confidence_threshold, verbose=False)
                if include_pose
                else ()
            )
            all_pose = self._pose_keypoints(pose_results) if include_pose else []
            all_pose_conf = self._pose_confidence(pose_results) if include_pose else []
            # Merge (x, y) with confidence once per person -- classify_pose and
            # classify_gaze both need the 3rd (confidence) element to do their
            # visibility checks; .xyn alone only carries (x, y).
            merged_keypoints: list[list[list[float]]] = []
            for person_index, pose_kps in enumerate(all_pose):
                pose_cf = all_pose_conf[person_index] if person_index < len(all_pose_conf) else []
                merged_keypoints.append([
                    [round(kp[0], 4), round(kp[1], 4), round(pose_cf[kp_i] if kp_i < len(pose_cf) else 0.0, 3)]
                    for kp_i, kp in enumerate(pose_kps)
                ])
            behaviors = [classify_pose(keypoints) for keypoints in merged_keypoints] if include_pose else []
            gazes = [classify_gaze(keypoints) for keypoints in merged_keypoints] if include_pose else []
            detections: list[Detection] = []
            for result in results:
                polygons = result.masks.xy if result.masks is not None else []
                for index, box in enumerate(result.boxes):
                    class_id = int(box.cls[0].item())
                    label = result.names[class_id]
                    coordinates = box.xyxy[0].tolist()
                    attributes = {"frame_shape": [frame_height, frame_width]}
                    if label == "person" and index < len(behaviors) and behaviors[index]:
                        attributes["behavior"] = behaviors[index]
                    if label == "person" and index < len(gazes) and gazes[index]:
                        attributes["gaze"] = gazes[index]
                    if label == "person" and index < len(merged_keypoints):
                        attributes["pose_keypoints"] = merged_keypoints[index]
                    if index < len(polygons) and polygons[index] is not None and len(polygons[index]) >= 3:
                        attributes["mask_polygon"] = [
                            [round(float(x), 1), round(float(y), 1)] for x, y in polygons[index].tolist()
                        ]
                    detections.append(
                        Detection(
                            label=label,
                            confidence=float(box.conf[0].item()),
                            bbox=BoundingBox(*map(float, coordinates)),
                            attributes=attributes,
                        )
                    )
            return tuple(detections), self._semantic_labels(frame) if include_semantics else ()

    def detect_rotation(self, frame: np.ndarray) -> int | None:
        import cv2

        candidates = {
            0: frame,
            90: cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE),
            180: cv2.rotate(frame, cv2.ROTATE_180),
            270: cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE),
        }
        if frame.shape[0] > frame.shape[1]:
            candidates = {angle: candidate for angle, candidate in candidates.items() if angle in {90, 270}}
        scores: dict[int, float] = {}
        with self._inference_lock:
            for angle, candidate in candidates.items():
                results = self._pose_model(candidate, device=self.config.device, conf=self.config.confidence_threshold, verbose=False)
                poses = self._pose_keypoints(results)
                scores[angle] = max((upright_pose_score(pose) for pose in poses), default=0.0)
        pose_rotation = select_rotation(scores)
        if pose_rotation is not None:
            return pose_rotation
        return self._clip_orientation(candidates) if frame.shape[0] > frame.shape[1] else None

    @staticmethod
    def rotate(frame: np.ndarray, angle: int) -> np.ndarray:
        import cv2
        return {0: frame, 90: cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), 180: cv2.rotate(frame, cv2.ROTATE_180), 270: cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)}[angle]

    def face_crops(self, frame: np.ndarray, detection: Detection) -> tuple[FaceCrop, ...]:
        import cv2

        height, width = frame.shape[:2]
        left = max(0, int(detection.bbox.x1))
        top = max(0, int(detection.bbox.y1))
        right = min(width, int(detection.bbox.x2))
        bottom = min(height, int(detection.bbox.y2))
        person = frame[top:bottom, left:right]
        if person.size == 0 or min(person.shape[:2]) < 72:
            return ()
        yunet_faces = self._yunet_faces(person)
        try:
            grayscale = cv2.cvtColor(person, cv2.COLOR_BGR2GRAY)
            frontal = self._face_detector.detectMultiScale(
                grayscale, scaleFactor=1.1, minNeighbors=6, minSize=(64, 64)
            )
            faces = frontal.tolist() if hasattr(frontal, "tolist") else []
            profile_detections = self._profile_face_detector.detectMultiScale(
                grayscale, scaleFactor=1.08, minNeighbors=4, minSize=(56, 56)
            )
            profiles = profile_detections.tolist() if hasattr(profile_detections, "tolist") else []
            mirrored_detections = self._profile_face_detector.detectMultiScale(
                cv2.flip(grayscale, 1), scaleFactor=1.08, minNeighbors=4, minSize=(56, 56)
            )
            mirrored = mirrored_detections.tolist() if hasattr(mirrored_detections, "tolist") else []
            profiles.extend([[person.shape[1] - x - face_width, y, face_width, face_height] for x, y, face_width, face_height in mirrored])
            faces.extend(profiles)
        except cv2.error:
            return ()
        accepted: list[FaceCrop] = []
        for yunet_face in yunet_faces:
            x, y, face_width, face_height = (int(round(value)) for value in yunet_face[:4])
            confidence = float(yunet_face[-1])
            aspect_ratio = face_width / face_height
            if not 0.65 <= aspect_ratio <= 1.5:
                continue
            face = person[y : y + face_height, x : x + face_width]
            if min(face.shape[:2]) >= 64:
                aligned = self._face_recognizer.alignCrop(person, yunet_face)
                if not self._face_crop_is_valid(aligned):
                    continue
                face_embedding = self._face_embedding(aligned)
                accepted.append(FaceCrop(aligned, confidence, "face", face_embedding))
        if accepted:
            return tuple(accepted)
        for x, y, face_width, face_height in faces:
            aspect_ratio = face_width / face_height
            if not 0.65 <= aspect_ratio <= 1.5:
                continue
            face = person[y : y + face_height, x : x + face_width]
            if min(face.shape[:2]) < 64:
                continue
            confidence = self._face_crop_confidence(face)
            if confidence >= 0.45:
                accepted.append(FaceCrop(face, confidence, "face-clip"))
        if accepted:
            return tuple(accepted)
        if detection.confidence >= 0.75:
            return (FaceCrop(person, detection.confidence, "appearance"),)
        return ()

    def _yunet_faces(self, person: np.ndarray) -> tuple[np.ndarray, ...]:
        height, width = person.shape[:2]
        with self._inference_lock:
            self._yunet_face_detector.setInputSize((width, height))
            _, detections = self._yunet_face_detector.detect(person)
        if detections is None:
            return ()
        faces: list[np.ndarray] = []
        for detection in detections:
            x, y, face_width, face_height = (int(round(value)) for value in detection[:4])
            confidence = float(detection[-1])
            if confidence < 0.75:
                continue
            left = max(0, x)
            top = max(0, y)
            right = min(width, x + face_width)
            bottom = min(height, y + face_height)
            if right > left and bottom > top:
                face = detection.copy()
                face[:4] = (left, top, right - left, bottom - top)
                faces.append(face)
        return tuple(faces)

    def _face_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        with self._inference_lock:
            embedding = self._face_recognizer.feature(aligned_face).reshape(-1).astype(np.float32)
        return embedding / np.linalg.norm(embedding)

    @staticmethod
    def _head_region_candidates(person: np.ndarray) -> tuple[np.ndarray, ...]:
        """Propose large head regions for profile views missed by Haar cascades."""
        height, width = person.shape[:2]
        side = min(int(height * 0.42), int(width * 0.68))
        if side < 128:
            return ()
        top = max(0, int(height * 0.02))
        centers = (int(width * 0.32), int(width * 0.50), int(width * 0.68))
        regions = []
        for center in centers:
            left = max(0, min(width - side, center - side // 2))
            face = person[top : min(height, top + side), left : left + side]
            if face.shape[0] == side and face.shape[1] == side:
                regions.append(face)
        return tuple(regions)

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        import cv2
        from PIL import Image

        rgb_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        with self._inference_lock, self._torch.no_grad():
            features = self._clip_model.encode_image(self._clip_preprocess(rgb_image).unsqueeze(0).to(self.config.device))
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].detach().cpu().numpy().astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        tokens = self._clip_tokenizer([text]).to(self.config.device)
        with self._inference_lock, self._torch.no_grad():
            features = self._clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].detach().cpu().numpy().astype(np.float32)

    def segment_held_object(self, frame: np.ndarray) -> SegmentedObject | None:
        import cv2

        height, width = frame.shape[:2]
        with self._inference_lock:
            pose_results = self._pose_model(frame, device=self.config.device, conf=self.config.confidence_threshold, verbose=False)
            result = self._sam.predict(frame, device=self.config.device, imgsz=self.config.sam_image_size, verbose=False)[0]
        if result.masks is None or result.boxes is None:
            return None
        wrists = self._wrist_points(pose_results, width, height)
        candidates: list[tuple[float, SegmentedObject]] = []
        for polygon, box, confidence in zip(result.masks.xy, result.boxes.xyxy, result.boxes.conf):
            if polygon is None or len(polygon) < 3:
                continue
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
            left, top, right, bottom = (int(round(value)) for value in box.tolist())
            left, top = max(left, 0), max(top, 0)
            right, bottom = min(right, width), min(bottom, height)
            if right <= left or bottom <= top:
                continue
            area_ratio = float(np.count_nonzero(mask)) / float(height * width)
            if not 0.001 <= area_ratio <= 0.20:
                continue
            crop = frame[top:bottom, left:right]
            crop_mask = mask[top:bottom, left:right]
            if crop.size == 0 or min(crop.shape[:2]) < 32:
                continue
            object_score = self._held_object_score(crop, crop_mask)
            if object_score < 0.45:
                continue
            center = np.array(((left + right) / 2, (top + bottom) / 2), dtype=np.float32)
            proximity = max(
                (1 - float(np.linalg.norm(center - wrist)) / (0.35 * min(width, height)) for wrist in wrists),
                default=0.35,
            )
            if wrists and proximity < 0.35:
                continue
            candidates.append(
                (object_score * float(confidence.item()) * proximity, SegmentedObject(crop, crop_mask, object_score))
            )
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def segment_detection(self, frame: np.ndarray, detection: Detection) -> SegmentedObject | None:
        """Extract an object crop from an actual instance-segmentation polygon."""
        import cv2

        polygon = detection.attributes.get("mask_polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            return None
        height, width = frame.shape[:2]
        points = np.asarray(polygon, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            return None
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
        left = max(0, int(np.floor(points[:, 0].min())))
        top = max(0, int(np.floor(points[:, 1].min())))
        right = min(width, int(np.ceil(points[:, 0].max())))
        bottom = min(height, int(np.ceil(points[:, 1].max())))
        if right - left < 24 or bottom - top < 24:
            return None
        image, crop_mask = frame[top:bottom, left:right], mask[top:bottom, left:right]
        if image.size == 0 or np.count_nonzero(crop_mask) < 128:
            return None
        return SegmentedObject(image, crop_mask, detection.confidence)

    @staticmethod
    def encode_segmented_object(segmented: SegmentedObject, max_size: int = 512) -> bytes:
        import cv2

        image, mask = segmented.image, segmented.mask
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        success, encoded = cv2.imencode(".png", np.dstack((image, mask)))
        if not success:
            raise RuntimeError("failed to encode segmented object for VLM classification")
        return encoded.tobytes()

    def _held_object_score(self, crop: np.ndarray, mask: np.ndarray) -> float:
        import cv2
        from PIL import Image

        masked = cv2.bitwise_and(crop, crop, mask=mask)
        image = Image.fromarray(cv2.cvtColor(masked, cv2.COLOR_BGR2RGB))
        labels = [
            "a handheld everyday object",
            "a human hand",
            "a human arm or body part",
            "a human face",
            "background scenery",
        ]
        tokens = self._clip_tokenizer(labels).to(self.config.device)
        with self._inference_lock, self._torch.no_grad():
            image_features = self._clip_model.encode_image(self._clip_preprocess(image).unsqueeze(0).to(self.config.device))
            text_features = self._clip_model.encode_text(tokens)
            similarity = (image_features / image_features.norm(dim=-1, keepdim=True)) @ (
                text_features / text_features.norm(dim=-1, keepdim=True)
            ).T
        scores = similarity[0].softmax(dim=0)
        return float(scores[0].item()) if int(scores.argmax().item()) == 0 else 0.0

    def _wrist_points(self, results: Sequence[object], width: int, height: int) -> list[np.ndarray]:
        wrists: list[np.ndarray] = []
        for keypoints in self._pose_keypoints(results):
            for index in (9, 10):
                if index >= len(keypoints):
                    continue
                x, y = keypoints[index][:2]
                if 0 < x < 1 and 0 < y < 1:
                    wrists.append(np.array((x * width, y * height), dtype=np.float32))
        return wrists

    def _clip_orientation(self, candidates: dict[int, np.ndarray]) -> int | None:
        import cv2
        from PIL import Image

        if not candidates:
            return None
        images = [Image.fromarray(cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)) for candidate in candidates.values()]
        prompt_tokens = self._clip_tokenizer(["an upright photograph", "a sideways or upside down photograph"]).to(
            self.config.device
        )
        with self._inference_lock, self._torch.no_grad():
            image_features = self._clip_model.encode_image(
                self._torch.stack([self._clip_preprocess(image) for image in images]).to(self.config.device)
            )
            text_features = self._clip_model.encode_text(prompt_tokens)
            similarity = (image_features / image_features.norm(dim=-1, keepdim=True)) @ (
                text_features / text_features.norm(dim=-1, keepdim=True)
            ).T
        angles = list(candidates)
        return angles[int(similarity[:, 0].argmax().item())]

    def _face_crop_confidence(self, face: np.ndarray) -> float:
        valid, confidence = self._face_crop_validity_batch([face])[0]
        return confidence if valid else 0.0

    def _face_crop_is_valid(self, face: np.ndarray) -> bool:
        return self._face_crop_validity_batch([face])[0][0]

    def validate_face_evidence(self, jpeg_images: list[bytes]) -> list[bool]:
        """Reject retained ears, hands, clothing, and objects without deleting evidence."""

        import cv2

        decoded: list[np.ndarray] = []
        for payload in jpeg_images:
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                decoded.append(np.empty((0, 0, 3), dtype=np.uint8))
            else:
                decoded.append(image)
        output: list[bool] = []
        for start in range(0, len(decoded), 64):
            batch = decoded[start : start + 64]
            valid_indices = [index for index, image in enumerate(batch) if image.size]
            decisions = [False] * len(batch)
            if valid_indices:
                validated = self._face_crop_validity_batch(
                    [batch[index] for index in valid_indices]
                )
                for index, (valid, _) in zip(valid_indices, validated):
                    decisions[index] = valid
            output.extend(decisions)
        return output

    def _face_crop_validity_batch(
        self, faces: list[np.ndarray]
    ) -> list[tuple[bool, float]]:
        import cv2
        from PIL import Image

        labels = [
            "a clear front-facing human face",
            "a clear human face in side profile",
            "a clear human face wearing eyeglasses",
            "an isolated ear, eye, or mouth without a complete face",
            "a hand or other body part",
            "clothing or fabric without a complete face",
            "background scenery or a physical object",
        ]
        if not faces:
            return []
        images = [
            Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)) for face in faces
        ]
        prompt_tokens = self._clip_tokenizer(labels).to(self.config.device)
        with self._inference_lock, self._torch.no_grad():
            image_features = self._clip_model.encode_image(
                self._torch.stack([self._clip_preprocess(image) for image in images]).to(
                    self.config.device
                )
            )
            text_features = self._clip_model.encode_text(prompt_tokens)
            similarity = (image_features / image_features.norm(dim=-1, keepdim=True)) @ (
                text_features / text_features.norm(dim=-1, keepdim=True)
            ).T
        scores = similarity.softmax(dim=1).cpu()
        return [
            (int(row.argmax().item()) in {0, 1, 2}, float(row[:3].max().item()))
            for row in scores
        ]

    @staticmethod
    def _pose_keypoints(results: Sequence[object]) -> list[list[list[float]]]:
        people: list[list[list[float]]] = []
        for result in results:
            keypoints = getattr(result, "keypoints", None)
            data = getattr(keypoints, "xyn", None)
            if data is None:
                continue
            for person_keypoints in data.cpu().tolist():
                people.append(person_keypoints)
        return people

    @staticmethod
    def _pose_confidence(results: Sequence[object]) -> list[list[float]]:
        people_conf: list[list[float]] = []
        for result in results:
            keypoints = getattr(result, "keypoints", None)
            conf = getattr(keypoints, "conf", None)
            if conf is None:
                continue
            for person_conf in conf.cpu().tolist():
                people_conf.append(person_conf)
        return people_conf

    def _semantic_labels(self, frame: np.ndarray) -> tuple[str, ...]:
        import cv2
        from PIL import Image

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prompt_tokens = self._clip_tokenizer(self.config.semantic_prompts).to(self.config.device)
        with self._torch.no_grad():
            image_features = self._clip_model.encode_image(self._clip_preprocess(image).unsqueeze(0).to(self.config.device))
            text_features = self._clip_model.encode_text(prompt_tokens)
            similarity = (image_features / image_features.norm(dim=-1, keepdim=True)) @ (
                text_features / text_features.norm(dim=-1, keepdim=True)
            ).T
        scores = similarity[0].softmax(dim=0).cpu().tolist()
        return tuple(prompt for prompt, score in zip(self.config.semantic_prompts, scores) if score >= 0.20)
