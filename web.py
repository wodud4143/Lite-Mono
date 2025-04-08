import os
import cv2
import time
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
from ptflops import get_model_complexity_info

import networks
from layers import disp_to_depth

# --- 카메라 파라미터 ---
f_kitti = 721.5377
B_kitti = 0.5327
f_realsense = 1.33
B_realsense = 0.05

# --- 모델 로딩 ---
def print_model_flops(model, height, width):
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model, (3, height, width), as_strings=True,
            print_per_layer_stat=False, verbose=False)
        print(f"[FLOPs] Encoder MACs: {macs} | Parameters: {params}")

def load_model(weights_folder, model_name="lite-mono-8m"):
    encoder_path = os.path.join(weights_folder, "encoder.pth")
    decoder_path = os.path.join(weights_folder, "depth.pth")
    encoder_dict = torch.load(encoder_path, map_location=torch.device('cpu'))
    decoder_dict = torch.load(decoder_path, map_location=torch.device('cpu'))

    height = encoder_dict['height']
    width = encoder_dict['width']

    encoder = networks.LiteMono(model=model_name, height=height, width=width)
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in encoder.state_dict()})
    encoder.eval()

    decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    decoder.load_state_dict({k: v for k, v in decoder_dict.items() if k in decoder.state_dict()})
    decoder.eval()

    print_model_flops(encoder, height, width)

    return encoder, decoder, height, width

# --- 추론 ---
def infer_frame(encoder, decoder, frame, feed_width, feed_height, device):
    input_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    original_width, original_height = input_image.size
    input_image = input_image.resize((feed_width, feed_height), Image.LANCZOS)
    input_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(input_tensor)
        outputs = decoder(features)
        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)

    return disp_resized.squeeze().cpu().numpy()

# --- 시각화 ---
def create_colormap(disp_np):
    vmax = np.percentile(disp_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped = (mapper.to_rgba(disp_np)[:, :, :3] * 255).astype(np.uint8)
    return colormapped

# --- 실시간 추론 및 클릭 분석 ---
def run_webcam_inference(weights_folder, model_name="lite-mono-8m"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, decoder, feed_height, feed_width = load_model(weights_folder, model_name)
    encoder.to(device)
    decoder.to(device)

    cap = cv2.VideoCapture("/dev/video4", cv2.CAP_V4L2)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    global disparity_map, latest_frame
    disparity_map = None
    latest_frame = None

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and disparity_map is not None:
            h, w = disparity_map.shape
            x_img = x if x < w else x - w
            if x_img >= w or y >= h:
                print(f"(x={x}, y={y}) → 클릭 범위 초과")
                return

            disparity_value = disparity_map[y, x_img]
            pixel_rgb = depth_map[y, x_img]  # ← 컬러맵된 뎁스 이미지에서 픽셀 추출
            brightness = int(0.299 * pixel_rgb[0] + 0.587 * pixel_rgb[1] + 0.114 * pixel_rgb[2])


            if disparity_value > 0:
                depth_kitti = (f_kitti * B_kitti) / disparity_value
                depth_realsense = (f_realsense * B_realsense) / disparity_value
                print(f"(x={x_img}, y={y}) → Disparity: {disparity_value:.3f}, Brightness: {brightness}, KITTI: {depth_kitti:.3f} m, RealSense: {depth_realsense:.3f} m")
            else:
                print(f"(x={x_img}, y={y}) → Disparity: 0.000, Brightness: {brightness}, Depth: undefined")

    cv2.namedWindow("Input | Depth Map")
    cv2.setMouseCallback("Input | Depth Map", click_event)

    print("웹캠 추론 시작. 클릭 시 픽셀 정보 출력. 종료하려면 'q' 키를 누르세요.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        disparity_map = infer_frame(encoder, decoder, frame, feed_width, feed_height, device)
        depth_map = create_colormap(disparity_map)
        depth_map_bgr = cv2.cvtColor(depth_map, cv2.COLOR_RGB2BGR)

        latest_frame = cv2.resize(frame, (depth_map_bgr.shape[1], depth_map_bgr.shape[0]))
        combined = np.hstack((latest_frame, depth_map_bgr))

        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(combined, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Input | Depth Map", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    weights_folder = "lite-mono-8m_640x192"
    run_webcam_inference(weights_folder)