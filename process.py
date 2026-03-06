import os
import subprocess
import shutil
import time
import random
import numpy as np
import librosa
from spleeter.separator import Separator

def extract_features(file_path, output_path):
    """보컬 파일에서 MFCC 특징을 추출하여 저장 (매칭 속도 최적화)"""
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        np.save(output_path, mfcc_scaled)
        return True
    except Exception as e:
        print(f"❌ 특징 추출 실패: {e}")
        return False

def auto_pipeline(song_list_file):
    db_dir = "vocal_db"
    feature_dir = "vocal_features" # 숫자 데이터 저장 폴더
    temp_dir = "temp_download"
    
    for d in [db_dir, feature_dir, temp_dir]:
        if not os.path.exists(d): os.makedirs(d)

    # Spleeter 초기화 (한 번만 로드해서 속도 향상)
    separator = Separator('spleeter:2stems')

    with open(song_list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if '|' not in line: continue
        song_title, url = [x.strip() for x in line.split('|')]
        
        final_vocal_path = os.path.join(db_dir, f"{song_title}_vocal.wav")
        feature_path = os.path.join(feature_dir, f"{song_title}.npy")

        # [중복 방지] 이미 결과물이 있다면 건너뜀
        if os.path.exists(final_vocal_path) and os.path.exists(feature_path):
            print(f"⏩ 건너뜀: {song_title} (이미 존재)")
            continue

        print(f"🎵 처리 중: {song_title}...")
        
        try:
            # 주소 세탁
            clean_url = url.split('&list=')[0].split('?list=')[0]
            audio_file = os.path.join(temp_dir, f"{song_title}.wav")

            # [최종병기] 쿠키 없이 차단을 뚫는 최신 옵션 조합
            download_cmd = [
                'yt-dlp',
                '-x', '--audio-format', 'wav',
                '--no-playlist',
                # 현재 가장 차단을 잘 피하는 VR 전용 클라이언트 설정
                '--extractor-args', 'youtube:player_client=android_vr,web',
                '--force-overwrites',
                '-o', audio_file,
                clean_url
            ]
            
            # 101 에러 등을 방지하기 위해 실시간 로그 확인용으로 수정
            subprocess.run(download_cmd, check=True)

            # 2. 보컬 분리
            separator.separate_to_file(audio_file, temp_dir)
            spleeter_out = os.path.join(temp_dir, song_title, "vocals.wav")
            
            if os.path.exists(spleeter_out):
                # 보컬 파일 이동
                os.replace(spleeter_out, final_vocal_path)
                
                # 3. 특징 추출 (나중에 매칭할 때 사용)
                extract_features(final_vocal_path, feature_path)
                print(f"✅ 완료: {song_title}")
            
            # 4. 정리: 곡마다 생성된 임시 폴더 삭제
            shutil.rmtree(os.path.join(temp_dir, song_title), ignore_errors=True)
            if os.path.exists(audio_file): os.remove(audio_file)

            # 유튜브 차단 방지를 위한 짧은 휴식 (전공자의 매너)
            time.sleep(random.uniform(2, 5))

        except Exception as e:
            print(f"❌ '{song_title}' 처리 중 에러 발생: {e}")
            with open("error_log.txt", "a") as err_f:
                err_f.write(f"{song_title}: {str(e)}\n")
            continue
        except subprocess.CalledProcessError as e:
            print(f"❌ 진짜 에러 원인: {e}")

if __name__ == "__main__":
    import os

    # 1. 현재 파이썬이 어디를 바라보고 있는지 출력
    current_path = os.getcwd()
    print(f"📍 현재 파이썬 실행 위치: {current_path}")

    # 2. songs.txt 파일이 진짜 이 폴더에 있는지 확인
    target_file = "songs.txt"
    if os.path.exists(target_file):
        print(f"✅ '{target_file}' 파일을 찾았습니다! 분석을 시작합니다.")
        # 파일 내용이 비어있는지 확인
        with open(target_file, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                print("❌ 파일은 찾았는데 내용이 텅 비어있습니다. 내용을 채워주세요!")
            else:
                print(f"📝 읽어온 내용:\n{content}")
                # 이제 자동화 함수 실행
                auto_pipeline(target_file)
    else:
        print(f"❌ '{target_file}' 파일이 현재 폴더에 없습니다!")
        print(f"💡 해결책: {current_path} 폴더 안에 '{target_file}'을 넣어주세요.")
