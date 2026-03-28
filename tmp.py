import os
import subprocess
import shutil
import time
import random
from spleeter.separator import Separator

def download_and_extract_vocal(song_title, url):
    db_dir = "tmp"         # 최종 보컬 파일 저장소
    temp_dir = "temp_download"   # 임시 작업 폴더
    
    for d in [db_dir, temp_dir]:
        if not os.path.exists(d): os.makedirs(d)

    # 1. 유튜브 음원 다운로드 (yt-dlp)
    print(f"🎵 '{song_title}' 다운로드 중...")
    audio_file = os.path.join(temp_dir, f"{song_title}.wav")
    clean_url = url.split('&list=')[0].split('?list=')[0]

    download_cmd = [
        'yt-dlp', '-x', '--audio-format', 'wav', '--no-playlist',
        '--extractor-args', 'youtube:player_client=android_vr,web',
        '--force-overwrites', '-o', audio_file, clean_url
    ]
    
    try:
        subprocess.run(download_cmd, check=True)
        
        # 2. 보컬 분리 (Spleeter 2stems)
        print(f"✂️ '{song_title}' 보컬 분리 중 (Spleeter)...")
        separator = Separator('spleeter:2stems')
        separator.separate_to_file(audio_file, temp_dir)
        
        # Spleeter는 기본적으로 '파일명/vocals.wav' 경로에 결과물을 생성함
        spleeter_out = os.path.join(temp_dir, song_title, "vocals.wav")
        final_vocal_path = os.path.join(db_dir, f"{song_title}_vocal.wav")

        if os.path.exists(spleeter_out):
            # 최종 보컬 파일 이동 및 이름 변경
            if os.path.exists(final_vocal_path): os.remove(final_vocal_path)
            os.rename(spleeter_out, final_vocal_path)
            print(f"✅ 추출 완료: {final_vocal_path}")
        
        # 3. 임시 파일 정리
        shutil.rmtree(os.path.join(temp_dir, song_title), ignore_errors=True)
        if os.path.exists(audio_file): os.remove(audio_file)
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    # 테스트용: 곡 제목과 유튜브 주소를 입력하세요
    test_title = "주저하는"
    test_url = "https://youtu.be/40vynd0KsHg?list=RD40vynd0KsHg"
    
    if "http" in test_url:
        download_and_extract_vocal(test_title, test_url)
    else:
        print("💡 test_url에 실제 유튜브 주소를 넣고 실행해주세요.")
