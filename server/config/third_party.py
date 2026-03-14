from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[np.uint8]


class OpenCVVideoCaptureProtocol(Protocol):
    def read(self) -> tuple[bool, ImageArray | None]: ...

    def release(self) -> None: ...

    def isOpened(self) -> bool: ...

    def get(self, prop_id: int) -> float: ...


class _OpenCVTextSizeProtocol(Protocol):
    def __iter__(self) -> object: ...


@runtime_checkable
class OpenCVModuleProtocol(Protocol):
    CAP_PROP_FRAME_WIDTH: int
    CAP_PROP_FRAME_HEIGHT: int
    IMREAD_COLOR: int
    IMWRITE_JPEG_QUALITY: int
    COLOR_BGR2RGB: int
    FONT_HERSHEY_SIMPLEX: int
    LINE_AA: int
    WINDOW_NORMAL: int

    def VideoCapture(self, index: int) -> OpenCVVideoCaptureProtocol: ...

    def imdecode(
        self,
        buffer: NDArray[np.uint8],
        flags: int,
    ) -> ImageArray | None: ...

    def cvtColor(self, image: ImageArray, code: int) -> ImageArray: ...

    def imencode(
        self,
        extension: str,
        image: ImageArray,
        params: Sequence[int],
    ) -> tuple[bool, NDArray[np.uint8]]: ...

    def putText(
        self,
        image: ImageArray,
        text: str,
        org: tuple[int, int],
        font_face: int,
        font_scale: float,
        color: tuple[int, int, int],
        thickness: int,
        line_type: int,
    ) -> ImageArray: ...

    def getTextSize(
        self,
        text: str,
        font_face: int,
        font_scale: float,
        thickness: int,
    ) -> tuple[tuple[int, int], int]: ...

    def rectangle(
        self,
        image: ImageArray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
    ) -> ImageArray: ...

    def namedWindow(self, window_name: str, flags: int) -> None: ...

    def flip(self, image: ImageArray, flip_code: int) -> ImageArray: ...

    def imshow(self, window_name: str, image: ImageArray) -> None: ...

    def waitKey(self, delay: int) -> int: ...

    def destroyAllWindows(self) -> None: ...

    def addWeighted(
        self,
        src1: ImageArray,
        alpha: float,
        src2: ImageArray,
        beta: float,
        gamma: float,
        dst: ImageArray,
    ) -> ImageArray: ...

    def line(
        self,
        image: ImageArray,
        pt1: tuple[int, int],
        pt2: tuple[int, int],
        color: tuple[int, int, int],
        thickness: int,
    ) -> ImageArray: ...


class LandmarkProtocol(Protocol):
    x: float
    y: float
    z: float


class LandmarkListProtocol(Protocol):
    landmark: Sequence[LandmarkProtocol]


class FaceMeshResultProtocol(Protocol):
    multi_face_landmarks: Sequence[LandmarkListProtocol] | None


class FaceMeshProtocol(Protocol):
    def process(self, frame_rgb: ImageArray) -> FaceMeshResultProtocol: ...

    def close(self) -> None: ...


class FaceMeshFactoryProtocol(Protocol):
    def __call__(
        self,
        *,
        static_image_mode: bool,
        max_num_faces: int,
        refine_landmarks: bool,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> FaceMeshProtocol: ...


class _FaceMeshNamespaceProtocol(Protocol):
    FaceMesh: FaceMeshFactoryProtocol


class PoseResultProtocol(Protocol):
    pose_landmarks: LandmarkListProtocol | None


class PoseProtocol(Protocol):
    def process(self, frame_rgb: ImageArray) -> PoseResultProtocol: ...

    def close(self) -> None: ...


class PoseFactoryProtocol(Protocol):
    def __call__(
        self,
        *,
        static_image_mode: bool,
        model_complexity: int,
        smooth_landmarks: bool,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> PoseProtocol: ...


class _PoseNamespaceProtocol(Protocol):
    Pose: PoseFactoryProtocol


class _SolutionsProtocol(Protocol):
    face_mesh: _FaceMeshNamespaceProtocol
    pose: _PoseNamespaceProtocol


class ImageFactoryProtocol(Protocol):
    def __call__(self, *, image_format: int, data: ImageArray) -> object: ...


class ImageFormatProtocol(Protocol):
    SRGB: int


@runtime_checkable
class MediaPipeModuleProtocol(Protocol):
    solutions: _SolutionsProtocol
    Image: ImageFactoryProtocol
    ImageFormat: ImageFormatProtocol


class BaseOptionsFactoryProtocol(Protocol):
    def __call__(self, *, model_asset_path: str) -> object: ...


@runtime_checkable
class MediaPipeTasksPythonProtocol(Protocol):
    BaseOptions: BaseOptionsFactoryProtocol


class BlendshapeCategoryProtocol(Protocol):
    category_name: str
    score: float


class FaceLandmarkerResultProtocol(Protocol):
    face_landmarks: Sequence[Sequence[LandmarkProtocol]] | None
    face_blendshapes: Sequence[Sequence[BlendshapeCategoryProtocol]] | None


class FaceLandmarkerProtocol(Protocol):
    def detect(self, image: object) -> FaceLandmarkerResultProtocol: ...

    def close(self) -> None: ...


class FaceLandmarkerOptionsFactoryProtocol(Protocol):
    def __call__(
        self,
        *,
        base_options: object,
        output_face_blendshapes: bool,
        num_faces: int,
        min_face_detection_confidence: float,
        min_face_presence_confidence: float,
        min_tracking_confidence: float,
    ) -> object: ...


class FaceLandmarkerClassProtocol(Protocol):
    def create_from_options(self, options: object) -> FaceLandmarkerProtocol: ...


@runtime_checkable
class MediaPipeVisionProtocol(Protocol):
    FaceLandmarkerOptions: FaceLandmarkerOptionsFactoryProtocol
    FaceLandmarker: FaceLandmarkerClassProtocol


class AudioClassificationPipelineProtocol(Protocol):
    def __call__(
        self,
        inputs: dict[str, object],
        *,
        top_k: int,
    ) -> Sequence[dict[str, object]]: ...


class PipelineFactoryProtocol(Protocol):
    def __call__(
        self,
        task: str,
        *,
        model: str,
    ) -> AudioClassificationPipelineProtocol: ...


@runtime_checkable
class TransformersModuleProtocol(Protocol):
    pipeline: PipelineFactoryProtocol


def load_cv2() -> OpenCVModuleProtocol:
    module = importlib.import_module("cv2")
    if not isinstance(module, OpenCVModuleProtocol):
        raise RuntimeError("cv2 imported successfully, but required symbols are unavailable.")
    return module


def load_mediapipe() -> MediaPipeModuleProtocol:
    module = importlib.import_module("mediapipe")
    if not isinstance(module, MediaPipeModuleProtocol):
        raise RuntimeError("mediapipe imported successfully, but required symbols are unavailable.")
    return module


def load_mediapipe_tasks_python() -> MediaPipeTasksPythonProtocol:
    module = importlib.import_module("mediapipe.tasks.python")
    if not isinstance(module, MediaPipeTasksPythonProtocol):
        raise RuntimeError("mediapipe.tasks.python imported successfully, but BaseOptions is unavailable.")
    return module


def load_mediapipe_tasks_vision() -> MediaPipeVisionProtocol:
    module = importlib.import_module("mediapipe.tasks.python.vision")
    if not isinstance(module, MediaPipeVisionProtocol):
        raise RuntimeError(
            "mediapipe.tasks.python.vision imported successfully, but required symbols are unavailable."
        )
    return module


def load_transformers_pipeline() -> PipelineFactoryProtocol:
    module = importlib.import_module("transformers")
    if not isinstance(module, TransformersModuleProtocol):
        raise RuntimeError("transformers imported successfully, but pipeline is unavailable.")
    return module.pipeline
