import os
import pandas as pd

# ==========================================
# 사용자 설정: 각 카메라별 3개 CSV 파일의 절대 경로
# ==========================================
CSV_PATHS = {
    "camera_1": {
        "first":  "/home/kimminsu/바탕화면/second_result_file/second_check/jetson/final_check_0605/cam1_latency.csv",
        "second": "/home/kimminsu/바탕화면/second_result_file/second_check/latency/latency_cam1.csv",
        "third":  "/home/kimminsu/바탕화면/second_result_file/second_check/latency/latency_lambda1.csv"
    },
    "camera_2": {
        "first":  "/home/kimminsu/바탕화면/second_result_file/second_check/jetson/final_check_0605/cam2_latency.csv",
        "second": "/home/kimminsu/바탕화면/second_result_file/second_check/latency/latency_cam2.csv",
        "third":  "/home/kimminsu/바탕화면/second_result_file/second_check/latency/latency_lambda2.csv"
    }
}
# ==========================================

def analyze_camera_delay(cam_id, files):
    print(f"=========================================")
    print(f" {cam_id.upper()} 구간별 딜레이 및 총 딜레이 분석")
    print(f"=========================================")

    # 각 구간의 평균 딜레이를 더하기 위한 변수 초기화
    total_delay = 0.0
    has_error = False

    # 1. 첫 번째 CSV 파일 처리 (Jetson 요청, Jetson-Server 통신)
    if os.path.exists(files["first"]):
        df1 = pd.read_csv(files["first"])
        
        if 'processing_duration_sec' in df1.columns:
            jetson_req_avg = df1['processing_duration_sec'].mean()
            total_delay += jetson_req_avg
            print(f"  Jetson 요청 딜레이       : {jetson_req_avg:.4f} 초")
        else:
            print(" 첫 번째 CSV에 'processing_duration_sec' 컬럼이 없습니다.")
            has_error = True
            
        if 'JETSON-server_delay' in df1.columns:
            jetson_server_comm_avg = df1['JETSON-server_delay'].mean()
            total_delay += jetson_server_comm_avg
            print(f" Jetson-server 통신 딜레이 : {jetson_server_comm_avg:.4f} 초")
        else:
            print(" 첫 번째 CSV에 'JETSON-server_delay' 컬럼이 없습니다.")
            has_error = True
    else:
        print(f" 첫 번째 CSV 파일을 찾을 수 없습니다: {files['first']}")
        has_error = True

    print("-" * 45)

    # 2. 두 번째 CSV 파일 처리 (서버 계산)
    if os.path.exists(files["second"]):
        df2 = pd.read_csv(files["second"])
        
        if 'process_latency' in df2.columns:
            server_calc_avg = df2['process_latency'].mean()
            total_delay += server_calc_avg
            print(f"  서버 계산 딜레이         : {server_calc_avg:.4f} 초")
        else:
            print(" 두 번째 CSV에 'process_latency' 컬럼이 없습니다.")
            has_error = True
    else:
        print(f" 두 번째 CSV 파일을 찾을 수 없습니다: {files['second']}")
        has_error = True

    print("-" * 45)

    # 3. 세 번째 CSV 파일 처리 (Server-Lambda 통신, Lambda 저장)
    if os.path.exists(files["third"]):
        df3 = pd.read_csv(files["third"])
        
        if 'lambda_network_delay' in df3.columns:
            server_lambda_comm_avg = df3['lambda_network_delay'].mean()
            total_delay += server_lambda_comm_avg
            print(f"  server-lambda 통신 딜레이: {server_lambda_comm_avg:.4f} 초")
        else:
            print(" 세 번째 CSV에 'lambda_network_delay' 컬럼이 없습니다.")
            has_error = True
            
        if 'processing_duration' in df3.columns:
            lambda_save_avg = df3['processing_duration'].mean()
            total_delay += lambda_save_avg
            print(f"  lambda 저장 딜레이       : {lambda_save_avg:.4f} 초")
        else:
            print(" 세 번째 CSV에 'processing_duration' 컬럼이 없습니다.")
            has_error = True
    else:
        print(f" 세 번째 CSV 파일을 찾을 수 없습니다: {files['third']}")
        has_error = True

    print("-" * 45)
    
    # 전체 딜레이 합산 출력
    if not has_error:
        print(f" {cam_id.upper()} 전체 시스템 딜레이  : {total_delay:.4f} 초")
    else:
        print(f" 일부 데이터 누락으로 인해 {cam_id.upper()} 총 딜레이를 정확히 계산할 수 없습니다.")
        
    print("=========================================\n")


def main():
    for cam_id, files in CSV_PATHS.items():
        analyze_camera_delay(cam_id, files)

if __name__ == "__main__":
    main()