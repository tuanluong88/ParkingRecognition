import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# 1. 경로 설정
CSV_LOG_PATH = "/home/kimminsu/바탕화면/second_result_file/second_check/results_check/results_check.csv"
GROUND_TRUTH_PATH = "/home/kimminsu/바탕화면/second_result_file/ground_truth.json"

# 2. 각 카메라별 검증하고 싶은 다중 구역 설정
VALID_RANGES = {
    "camera_1": [(11,23), (38, 71),
    (77, 110),
    (116,149),
    (155,188)],
    "camera_2": [(1, 6),(9,10),(13,54),
    (60, 113),
    (119,172),
    (178,229),
    (341,360)]
}

# 데이터 로드
df_log = pd.read_csv(CSV_LOG_PATH)
df_log['frame_name'] = df_log['frame_name'].str.strip()
df_log['camera_id'] = df_log['camera_id'].str.strip()

with open(GROUND_TRUTH_PATH, "r") as f:
    ground_truth = json.load(f)

print("=========================================")
print("    종합 성능 평가    ")
print("=========================================\n")

for cam_id in ["camera_1", "camera_2"]:
    if cam_id not in ground_truth:
        continue
        
    camera_gt = ground_truth[cam_id]
    sorted_frame_names = sorted(list(camera_gt.keys()))
    ranges = VALID_RANGES[cam_id]
    
    # [방식 1] 전체 데이터를 하나로 묶을 리스트
    y_true_all = []
    y_pred_all = []
    
    # [방식 2] 프레임별 점수를 저장할 리스트
    frame_accuracies = []
    frame_f1_scores = []
    
    matched_frame_count = 0
    df_cam = df_log[df_log['camera_id'] == cam_id].drop_duplicates(subset=['frame_name'], keep='last')
    
    for f_name in sorted_frame_names:
        row_match = df_cam[df_cam['frame_name'] == f_name]
        
        if not row_match.empty:
            row = row_match.iloc[0]
            status_column = f"{cam_id}_status"
            
            full_pred_list = json.loads(row[status_column])
            true_list = camera_gt[f_name]
            
            # 다중 구역 슬라이싱 병합
            filtered_pred_list = []
            for start_num, end_num in ranges:
                chunk = full_pred_list[start_num - 1 : end_num]
                filtered_pred_list.extend(chunk)
            
            if len(filtered_pred_list) == len(true_list):
                # 1. 전체 합산용 데이터 누적 (방식 1)
                y_true_all.extend(true_list)
                y_pred_all.extend(filtered_pred_list)
                
                # 2. 개별 프레임별 점수 계산 (방식 2)
                f_acc = accuracy_score(true_list, filtered_pred_list)
                # 정답과 예측이 모두 0일 때 F1 점수가 0으로 왜곡되는 걸 방지하기 위해 zero_division 설정
                f_f1 = f1_score(true_list, filtered_pred_list, average='binary', zero_division=1)
                
                frame_accuracies.append(f_acc)
                frame_f1_scores.append(f_f1)
                
                matched_frame_count += 1

    # 최종 결과 출력
    print(f"[{cam_id} 분석 리포트]")
    print(f" - 매칭된 총 프레임 수: {matched_frame_count} 개")
    
    if len(y_pred_all) > 0:
        # 방식 1: 전체 합산 지표
        total_accuracy = accuracy_score(y_true_all, y_pred_all)
        total_f1 = f1_score(y_true_all, y_pred_all, average='binary', zero_division=1)
        
        # 방식 2: 프레임별 점수들의 평균
        avg_frame_accuracy = np.mean(frame_accuracies)
        avg_frame_f1 = np.mean(frame_f1_scores)
        
        print(f" [방법 1] 전체 데이터 합산 후 단일 계산")
        print(f"   • 종합 Accuracy : {total_accuracy:.4f}")
        print(f"   • 종합 F1-Score : {total_f1:.4f}")
        print(f" [방법 2] 프레임별로 지표 계산 후 평균 산출")
        print(f"   • 평균 Accuracy : {avg_frame_accuracy:.4f}")
        print(f"   • 평균 F1-Score : {avg_frame_f1:.4f}")
    else:
        print("  비교 가능한 매칭 데이터가 없습니다.")
        
    print("-" * 50)