import os
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity

def get_user_voice_features(file_path):
    """사용자 녹음 파일에서 특징 추출"""
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def find_my_match(user_wav_path, feature_dir="vocal_features"):
    # 1. 사용자 특징 추출
    print("🎙️ 내 목소리 분석 중...")
    user_feat = get_user_voice_features(user_wav_path).reshape(1, -1)
    
    results = []
    
    # 2. DB 내의 모든 곡과 비교 (Cosine Similarity)
    for feat_file in os.listdir(feature_dir):
        if not feat_file.endswith('.npy'): continue
        
        song_feat = np.load(os.path.join(feature_dir, feat_file)).reshape(1, -1)
        
        # 코사인 유사도 계산 (1에 가까울수록 비슷함)
        similarity = cosine_similarity(user_feat, song_feat)[0][0]
        song_name = feat_file.replace('.npy', '')
        results.append((song_name, similarity))
    
    # 3. 점수 높은 순으로 정렬
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\n🏆 --- 매칭 결과 분석 ---")
    for i, (name, score) in enumerate(results[:5]): # 상위 5위까지 출력
        print(f"{i+1}위: {name} (유사도: {score*100:.2f}%)")

if __name__ == "__main__":
    # 본인이 미리 녹음해둔 wav 파일 경로를 넣으세요.
    # (녹음 환경이 없다면 스마트폰으로 녹음해서 옮겨도 됩니다!)
    my_voice = "my.m4a" 
    
    if os.path.exists(my_voice):
        find_my_match(my_voice)
    else:
        print(f"❌ {my_voice} 파일이 없습니다. 녹음 파일을 준비해 주세요!")
