from inference import get_model
import os
import supervision as sv
import tqdm
import opencv as cv2
import argparse
import torch
import numpy as np
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
import umap
from sklearn.cluster import KMeans
from more_itertools import chunked
from transformers import AutoProcessor, SiglipVisionModel
from utils import resolve_goalkeepers_team_id, radar_view, voronoi,voronoi_blend,ball_tracking
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"

# Constants
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3
STRIDE = 30
BATCH_SIZE = 32

def load_models():
    """Load and initialize all required models"""
    ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY')
    if not ROBOFLOW_API_KEY:
        raise ValueError("ROBOFLOW_API_KEY environment variable is required. Please set it in your .env file or environment.")
    
    PLAYER_DETECTION_MODEL_ID = "football-players-detection-3zvbc/11"
    PLAYER_DETECTION_MODEL = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
    
    FIELD_DETECTION_MODEL_ID = "football-field-detection-f07vi/14"
    FIELD_DETECTION_MODEL = get_model(model_id=FIELD_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
    
    SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EMBEDDINGS_MODEL = SiglipVisionModel.from_pretrained(SIGLIP_MODEL_PATH).to(DEVICE)
    EMBEDDINGS_PROCESSOR = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
    
    return PLAYER_DETECTION_MODEL, FIELD_DETECTION_MODEL, EMBEDDINGS_MODEL, EMBEDDINGS_PROCESSOR, DEVICE

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser('Football AI inputs')
    
    parser.add_argument('--normal', action='store_true', help='normal detection of players')
    parser.add_argument('--voronoi', action='store_true', help='simple voronoi of players')
    parser.add_argument('--voronoi_blend', action='store_true', help='voronoi blend of players')
    parser.add_argument('--radar_view', action='store_true', help='radar view of players and pitch')
    parser.add_argument('--btrack', action='store_true', help='ball tracking')
    parser.add_argument('--video_path', type=str, required=True, help='input video source path')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    SOURCE_VIDEO_PATH = args.video_path
    
    # Load models
    PLAYER_DETECTION_MODEL, FIELD_DETECTION_MODEL, EMBEDDINGS_MODEL, EMBEDDINGS_PROCESSOR, DEVICE = load_models()
    
    # Initialize configuration
    CONFIG = SoccerPitchConfiguration()
    
    # Collect player crops for team classification
    print("Collecting player crops for team classification...")
    frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH, stride=STRIDE)
    
    crops = []
    for frame in tqdm.tqdm(frame_generator, desc='collecting crops'):
        result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)
        players_detections = detections[detections.class_id == PLAYER_ID]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        crops += players_crops
    
    # Train team classifier
    print("Training team classifier...")
    team_classifier = TeamClassifier(device=DEVICE)
    team_classifier.fit(crops)
    
    # Initialize annotators
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=20, height=17
    )

    tracker = sv.ByteTrack()
    tracker.reset()
    
    # Process first frame for detection and visualization
    print("Processing first frame...")
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
    frame = next(frame_generator)

    # ball, goalkeeper, player, referee detection

    result = PLAYER_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
    detections = sv.Detections.from_inference(result)

    ball_detections = detections[detections.class_id == BALL_ID]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    all_detections = detections[detections.class_id != BALL_ID]
    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
    all_detections = tracker.update_with_detections(detections=all_detections)

    goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
    players_detections = all_detections[all_detections.class_id == PLAYER_ID]
    referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

    # team assignment

    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
        players_detections, goalkeepers_detections)

    referees_detections.class_id -= 1

    all_detections = sv.Detections.merge([
        players_detections, goalkeepers_detections, referees_detections])

    # frame visualization

    labels = [
        f"#{tracker_id}"
        for tracker_id
        in all_detections.tracker_id
    ]

    all_detections.class_id = all_detections.class_id.astype(int)

    annotated_frame = frame.copy()
    annotated_frame = ellipse_annotator.annotate(
        scene=annotated_frame,
        detections=all_detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=all_detections,
        labels=labels)
    annotated_frame = triangle_annotator.annotate(
        scene=annotated_frame,
        detections=ball_detections)

    sv.plot_image(annotated_frame)

    players_detections = sv.Detections.merge([
        players_detections, goalkeepers_detections
    ])

    # detect pitch key points

    result = FIELD_DETECTION_MODEL.infer(frame, confidence=0.3)[0]
    key_points = sv.KeyPoints.from_inference(result)

    # project ball, players and referies on pitch

    filter = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    transformer = ViewTransformer(
        source=frame_reference_points,
        target=pitch_reference_points
    )

    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)

    players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = transformer.transform_points(points=players_xy)

    referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_referees_xy = transformer.transform_points(points=referees_xy)

    if args.voronoi:
        print("Running Voronoi analysis...")
        voronoi(CONFIG, pitch_ball_xy, pitch_players_xy, players_detections)
    elif args.voronoi_blend:
        print("Running Voronoi blend analysis...")
        voronoi_blend(CONFIG, pitch_ball_xy, pitch_players_xy, players_detections)
    elif args.radar_view:
        print("Running radar view analysis...")
        radar_view(CONFIG, pitch_ball_xy, pitch_players_xy, pitch_referees_xy, players_detections)
    elif args.btrack:
        print("Running ball tracking analysis...")
        ball_tracking(CONFIG, SOURCE_VIDEO_PATH, PLAYER_DETECTION_MODEL, FIELD_DETECTION_MODEL, ViewTransformer)
    elif args.normal:
        print("Running normal detection...")
        # Normal detection is already done above
    else:
        print("No analysis mode selected. Use --normal, --voronoi, --voronoi_blend, --radar_view, or --btrack")

if __name__ == "__main__":
    main()
