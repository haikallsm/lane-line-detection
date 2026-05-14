import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# =========================================================
# CONFIGURATION (Journal-aligned defaults)
# =========================================================

# File
IMAGE_PATH = "lane1.jpg"
VIDEO_PATH = "Lane detect test data.mp4"

# Gaussian Blur (optimal from paper)
BLUR_KERNEL = (9, 9)
BLUR_SIGMA = 0

# Canny (paper)
CANNY_THRESHOLD1 = 100
CANNY_THRESHOLD2 = 200

# Hough Transform (paper)
HOUGH_RHO = 6
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 160
HOUGH_MIN_LINE_LENGTH = 20
HOUGH_MAX_LINE_GAP = 25

# Drawing
LINE_COLOR = (0, 255, 0)
LINE_THICKNESS = 10
LINE_BLEND_ALPHA = 0.8

# ROI (scaled from paper reference @ 1024x720)
# left-bottom  = (0,700)
# right-bottom = (1024,700)
# apex         = (560,330)
ROI_BOTTOM_Y_FRAC = 700 / 720
ROI_APEX_X_FRAC = 560 / 1024
ROI_APEX_Y_FRAC = 330 / 720

# Non-Local Maximum Suppression (NMS-style)
ENABLE_NLMS = True

# Morphological refinement (optional; improves broken edges but can add false positives)
ENABLE_MORPHOLOGY = True
MORPH_KERNEL_SIZE = 3
# "close" or "close_open"
MORPH_MODE = "close"

# Lane fitting
SLOPE_THRESHOLD = 0.3  # same idea as your filter; adjust if needed
Y2_FRAC = 0.60         # top end of rendered lane line (relative to height)

# Video stability
ENABLE_TEMPORAL_SMOOTHING = True
EMA_ALPHA = 0.20  # 0..1 (smaller = smoother, slower response)

# =========================================================


class LaneDetector:
    """
    Lane Detection System (Classic CV)
    Based on:
    "Lane Line Detection and Object Scene Segmentation Using Otsu Thresholding
     and the Fast Hough Transform for Intelligent Vehicles in Complex Road Conditions"
    """

    def __init__(self):
        self.image = None
        self.roi_vertices = None

        # For temporal smoothing (video)
        self._prev_left_params = None   # (slope, intercept)
        self._prev_right_params = None  # (slope, intercept)

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    def load_image(self, image_path: str) -> bool:
        print(f"[*] Loading image: {image_path}")
        self.image = cv2.imread(image_path)

        if self.image is None:
            print(f"[!] ERROR: Cannot load image {image_path}")
            return False

        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        print(f"[+] Image loaded: {self.image.shape}")
        return True

    # =====================================================
    # ROI
    # =====================================================

    def set_roi_vertices(self, width: int | None = None, height: int | None = None):
        """
        Triangular ROI scaled from paper reference points.
        """
        if width is None or height is None:
            if self.image is None:
                raise ValueError("No image set; provide width/height.")
            height = self.image.shape[0]
            width = self.image.shape[1]

        y_bottom = int(height * ROI_BOTTOM_Y_FRAC)
        y_bottom = min(max(y_bottom, 0), height - 1)

        apex_x = int(width * ROI_APEX_X_FRAC)
        apex_y = int(height * ROI_APEX_Y_FRAC)
        apex_x = min(max(apex_x, 0), width - 1)
        apex_y = min(max(apex_y, 0), height - 1)

        self.roi_vertices = np.array(
            [[(0, y_bottom), (apex_x, apex_y), (width - 1, y_bottom)]],
            dtype=np.int32,
        )

        # print once for image mode; video mode calls this once anyway
        print("[+] ROI set (Triangular, paper-scaled)")

    def region_of_interest(self, img: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(img)

        if len(img.shape) > 2:
            ignore_mask_color = (255,) * img.shape[2]
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, self.roi_vertices, ignore_mask_color)
        return cv2.bitwise_and(img, mask)

    # =====================================================
    # PREPROCESSING
    # =====================================================

    @staticmethod
    def convert_to_grayscale(image_rgb: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def apply_gaussian_blur(image_gray: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image_gray, BLUR_KERNEL, BLUR_SIGMA)

    # =====================================================
    # SOBEL (Paper-aligned: |Gx| + |Gy|)
    # =====================================================

    @staticmethod
    def apply_sobel(image_gray_blur: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          gx (float32), gy (float32), magnitude (float32) where magnitude = |gx| + |gy|
        """
        gx = cv2.Sobel(image_gray_blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image_gray_blur, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.abs(gx) + np.abs(gy)
        return gx, gy, magnitude

    # =====================================================
    # NLMS / NMS (directional non-maximum suppression)
    # =====================================================

    @staticmethod
    def non_local_maximum_suppression(magnitude: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
        """
        NMS-style thinning using gradient direction (Canny-like NMS).
        Output: suppressed magnitude (float32), same shape as input.
        """
        if magnitude.size == 0:
            return magnitude

        # Direction in degrees [0, 180)
        angle = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0

        m = magnitude.astype(np.float32)
        p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=0)

        center = p[1:-1, 1:-1]
        left = p[1:-1, 0:-2]
        right = p[1:-1, 2:]
        up = p[0:-2, 1:-1]
        down = p[2:, 1:-1]
        up_right = p[0:-2, 2:]
        down_left = p[2:, 0:-2]
        up_left = p[0:-2, 0:-2]
        down_right = p[2:, 2:]

        # Quantize angle into 4 directions: 0, 45, 90, 135
        dir0 = (angle < 22.5) | (angle >= 157.5)
        dir45 = (angle >= 22.5) & (angle < 67.5)
        dir90 = (angle >= 67.5) & (angle < 112.5)
        dir135 = (angle >= 112.5) & (angle < 157.5)

        keep = np.zeros_like(center, dtype=bool)
        keep |= dir0 & (center >= left) & (center >= right)
        keep |= dir90 & (center >= up) & (center >= down)
        keep |= dir45 & (center >= up_right) & (center >= down_left)
        keep |= dir135 & (center >= up_left) & (center >= down_right)

        suppressed = np.zeros_like(center, dtype=np.float32)
        suppressed[keep] = center[keep]
        return suppressed

    # =====================================================
    # HELPERS
    # =====================================================

    @staticmethod
    def normalize_to_uint8(image_f32: np.ndarray) -> np.ndarray:
        """
        Normalize float image to [0..255] uint8 for Otsu/Canny.
        """
        img = image_f32.astype(np.float32)
        maxv = float(img.max()) if img.size else 0.0
        if maxv <= 1e-6:
            return np.zeros(img.shape, dtype=np.uint8)
        out = (255.0 * img / maxv).clip(0, 255).astype(np.uint8)
        return out

    # =====================================================
    # OTSU
    # =====================================================

    @staticmethod
    def apply_otsu_threshold(image_u8: np.ndarray) -> np.ndarray:
        _, otsu = cv2.threshold(image_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu

    # =====================================================
    # MORPHOLOGICAL REFINEMENT (optional)
    # =====================================================

    @staticmethod
    def morphological_refinement(binary_u8: np.ndarray) -> np.ndarray:
        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)

        if MORPH_MODE == "close":
            return cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, kernel)

        # default: close_open
        refined = cv2.morphologyEx(binary_u8, cv2.MORPH_CLOSE, kernel)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
        return refined

    # =====================================================
    # CANNY
    # =====================================================

    @staticmethod
    def apply_canny(image_u8: np.ndarray) -> np.ndarray:
        return cv2.Canny(image_u8, CANNY_THRESHOLD1, CANNY_THRESHOLD2)

    # =====================================================
    # HOUGH TRANSFORM
    # =====================================================

    @staticmethod
    def detect_lines(edge_u8: np.ndarray):
        return cv2.HoughLinesP(
            edge_u8,
            rho=HOUGH_RHO,
            theta=HOUGH_THETA,
            threshold=HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LINE_LENGTH,
            maxLineGap=HOUGH_MAX_LINE_GAP,
        )

    # =====================================================
    # LEAST SQUARES FITTING (weighted + optional temporal smoothing)
    # =====================================================

    def _ema(self, prev: tuple[float, float] | None, new: tuple[float, float] | None) -> tuple[float, float] | None:
        if not ENABLE_TEMPORAL_SMOOTHING:
            return new

        if new is None:
            return prev
        if prev is None:
            return new

        a = EMA_ALPHA
        return (a * new[0] + (1 - a) * prev[0], a * new[1] + (1 - a) * prev[1])

    def calculate_lane_lines(self, image_rgb: np.ndarray, lines, *, allow_temporal_smoothing: bool) -> list | None:
        if lines is None:
            if allow_temporal_smoothing and ENABLE_TEMPORAL_SMOOTHING:
                # keep previous if available
                left_params = self._prev_left_params
                right_params = self._prev_right_params
                return self._params_to_lines(image_rgb, left_params, right_params)
            return None

        left_params = []
        left_weights = []
        right_params = []
        right_weights = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue

            # Fit line y = slope*x + intercept (polyfit on (x,y))
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)

            # Filter slope noise
            if -SLOPE_THRESHOLD < slope < SLOPE_THRESHOLD:
                continue

            length = float(np.hypot(dx, dy))
            if slope < 0:
                left_params.append((float(slope), float(intercept)))
                left_weights.append(length)
            else:
                right_params.append((float(slope), float(intercept)))
                right_weights.append(length)

        left_avg = None
        if left_params:
            left_avg = tuple(np.average(np.array(left_params), axis=0, weights=np.array(left_weights)))

        right_avg = None
        if right_params:
            right_avg = tuple(np.average(np.array(right_params), axis=0, weights=np.array(right_weights)))

        if allow_temporal_smoothing and ENABLE_TEMPORAL_SMOOTHING:
            left_avg = self._ema(self._prev_left_params, left_avg)
            right_avg = self._ema(self._prev_right_params, right_avg)
            self._prev_left_params = left_avg
            self._prev_right_params = right_avg

        return self._params_to_lines(image_rgb, left_avg, right_avg)

    def _params_to_lines(
        self,
        image_rgb: np.ndarray,
        left_params: tuple[float, float] | None,
        right_params: tuple[float, float] | None,
    ) -> list | None:
        h, w = image_rgb.shape[:2]

        def make_coordinates(params: tuple[float, float] | None):
            if params is None:
                return None
            slope, intercept = params
            if abs(slope) < 1e-6:
                return None

            y1 = h - 1
            y2 = int(h * Y2_FRAC)

            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)

            # Clamp for safety
            x1 = int(np.clip(x1, 0, w - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            y2 = int(np.clip(y2, 0, h - 1))

            return np.array([x1, y1, x2, y2])

        left_line = make_coordinates(left_params)
        right_line = make_coordinates(right_params)

        final_lines = []
        if left_line is not None:
            final_lines.append([left_line])
        if right_line is not None:
            final_lines.append([right_line])

        return final_lines if final_lines else None

    # =====================================================
    # DRAW LINES
    # =====================================================

    @staticmethod
    def draw_lines(image_rgb: np.ndarray, lines) -> np.ndarray:
        line_image = np.zeros_like(image_rgb)

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_image, (x1, y1), (x2, y2), LINE_COLOR, LINE_THICKNESS)

        return cv2.addWeighted(image_rgb, LINE_BLEND_ALPHA, line_image, 1, 0)

    # =====================================================
    # PROCESS PIPELINE (Journal order)
    # =====================================================

    def process_pipeline(self, frame_rgb: np.ndarray, *, for_video: bool):
        """
        Journal order:
        Grayscale -> Blur -> Sobel -> NLMS -> Otsu -> ROI -> Canny -> Hough -> LS fit -> Render

        Returns intermediates for debugging/visualization.
        """
        gray = self.convert_to_grayscale(frame_rgb)
        blur = self.apply_gaussian_blur(gray)

        gx, gy, mag_f32 = self.apply_sobel(blur)

        if ENABLE_NLMS:
            nms_f32 = self.non_local_maximum_suppression(mag_f32, gx, gy)
        else:
            nms_f32 = mag_f32

        nms_u8 = self.normalize_to_uint8(nms_f32)
        otsu = self.apply_otsu_threshold(nms_u8)

        roi = self.region_of_interest(otsu)

        # Optional morphology after ROI (safer than global morphology)
        if ENABLE_MORPHOLOGY:
            roi_refined = self.morphological_refinement(roi)
        else:
            roi_refined = roi

        canny = self.apply_canny(roi_refined)
        raw_lines = self.detect_lines(canny)

        fitted_lines = self.calculate_lane_lines(frame_rgb, raw_lines, allow_temporal_smoothing=for_video)
        result = self.draw_lines(frame_rgb, fitted_lines)

        # Provide Sobel magnitude as uint8 for display
        sobel_u8 = self.normalize_to_uint8(mag_f32)
        return gray, blur, sobel_u8, nms_u8, otsu, roi_refined, canny, result

    # =====================================================
    # PROCESS IMAGE
    # =====================================================

    def process_image(self, image_path: str):
        print("\n" + "=" * 60)
        print("PROCESSING IMAGE")
        print("=" * 60)

        if not self.load_image(image_path):
            return

        h, w = self.image.shape[:2]
        self.set_roi_vertices(w, h)

        gray, blur, sobel, nms, otsu, roi, canny, result = self.process_pipeline(self.image, for_video=False)

        print("[*] Displaying results...")
        self.display_results(
            Original=self.image,
            Grayscale=gray,
            Blur=blur,
            Sobel_Magnitude=sobel,
            NLMS_NMS=nms,
            Otsu=otsu,
            ROI=roi,
            Canny=canny,
            Result=result,
        )

    # =====================================================
    # DISPLAY
    # =====================================================

    @staticmethod
    def display_results(**images):
        count = len(images)
        cols = 3
        rows = (count + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        axes = axes.flatten()

        for idx, (title, img) in enumerate(images.items()):
            if len(img.shape) == 2:
                axes[idx].imshow(img, cmap="gray")
            else:
                axes[idx].imshow(img)

            axes[idx].set_title(title, fontsize=10, fontweight="bold")
            axes[idx].axis("off")

        for idx in range(count, len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.show()

    # =====================================================
    # VIDEO PROCESSING
    # =====================================================

    def process_video(self, video_path: str):
        print("\n" + "=" * 60)
        print("PROCESSING VIDEO")
        print("=" * 60)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[!] Cannot open video")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.set_roi_vertices(width, height)

        frame_count = 0
        print("[+] Video started. Press 'ESC' to exit.")

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            _, _, _, _, _, _, canny, result_rgb = self.process_pipeline(frame_rgb, for_video=True)

            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                result_bgr,
                f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Lane Detection Result", result_bgr)
            cv2.imshow("Canny Edge", canny)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                print("[*] Exit by user")
                break

        cap.release()
        cv2.destroyAllWindows()


# =========================================================
# MENU
# =========================================================

def print_menu():
    print("\n" + "=" * 60)
    print("LANE LINE DETECTION SYSTEM")
    print("OTSU + (N)LMS + HOUGH (Paper-aligned)")
    print("=" * 60)
    print("1. Process Image")
    print("2. Process Video")
    print("3. Exit")
    print("=" * 60)


def main():
    detector = LaneDetector()

    while True:
        print_menu()
        choice = input("Choose menu (1-3): ").strip()

        if choice == "1":
            image_file = input(f"Input image file (default: {IMAGE_PATH}): ").strip()
            if not image_file:
                image_file = IMAGE_PATH

            if not Path(image_file).exists():
                print(f"[!] File not found: {image_file}")
                continue

            detector.process_image(image_file)

        elif choice == "2":
            video_file = input(f"Input video file (default: {VIDEO_PATH}): ").strip()
            if not video_file:
                video_file = VIDEO_PATH

            if not Path(video_file).exists():
                print(f"[!] File not found: {video_file}")
                continue

            detector.process_video(video_file)

        elif choice == "3":
            print("\n[+] Thank you!")
            print("[+] Exit program")
            break

        else:
            print("[!] Invalid menu")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[!] ERROR: {str(e)}")
        sys.exit(1)