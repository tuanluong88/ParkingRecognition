import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

def prepare_custom_calibration_data(source_dir, target_dir, num_samples=100):
    """
    학습 데이터셋에서 무작위로 이미지를 선택하여 캘리브레이션 폴더로 복사합니다.
    """
    # 1. 대상 디렉토리 생성
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # 2. 지원하는 이미지 확장자 정의
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    # 3. 소스 디렉토리에서 이미지 파일 목록 가져오기
    print(f"'{source_path}'에서 이미지 파일을 검색 중...")
    image_files = [
        f for f in source_path.rglob('*') 
        if f.suffix.lower() in valid_extensions
    ]

    if not image_files:
        print("오류: 소스 폴더에서 이미지 파일을 찾을 수 없습니다.")
        return

    # 4. 샘플링 개수 결정
    num_to_sample = min(len(image_files), num_samples)
    selected_files = random.sample(image_files, num_to_sample)

    print(f"총 {len(image_files)}개 중 {num_to_sample}개의 이미지를 선택하여 복사를 시작합니다...")

    # 5. 파일 복사 (진행 표시줄 포함)
    success_count = 0
    for file_path in tqdm(selected_files, desc="Copying images"):
        try:
            # 파일 이름 중복 방지를 위해 원본 경로를 포함한 이름을 생성하거나 그대로 사용
            dest_file = target_path / file_path.name
            shutil.copy2(file_path, dest_file)
            success_count += 1
        except Exception as e:
            print(f"파일 복사 실패: {file_path.name} - {e}")

    print(f"\n완료! {success_count}개의 이미지가 '{target_dir}' 폴더에 준비되었습니다.")
    print("이제 이 폴더를 Mobilint 컴파일러의 캘리브레이션 경로로 사용하세요.")

if __name__ == "__main__":
    # --- 설정 영역 ---
    # 실제 학습 데이터가 저장된 최상위 폴더 경로
    MY_TRAIN_DATA_DIR = "./test/images" 
    
    # 캘리브레이션 이미지를 저장할 대상 폴더 경로
    CALIB_DEST_DIR = "./calib_images"
    
    # 추출할 이미지 개수 (보통 100~500장 정도면 충분합니다)
    SAMPLE_COUNT = 100 
    # ----------------

    prepare_custom_calibration_data(MY_TRAIN_DATA_DIR, CALIB_DEST_DIR, SAMPLE_COUNT)