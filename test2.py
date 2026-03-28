import numpy as np
import librosa
import pesto
import torch
import matplotlib.pyplot as plt

def final_vocal_analysis_engine(audio_path, ref_npz_path):
    print(f"🚀 [System] 분석 엔진 가동: {audio_path}")
    
    # 1. PESTO 딥러닝 기반 피치 추출 및 텐서 연산
    y, sr = librosa.load(audio_path, sr=16000)
    y_tensor = torch.from_numpy(y).float().to("cuda" if torch.cuda.is_available() else "cpu")
    
    pitch_info = pesto.predict(y_tensor, sr)
    u_pitch_hz = pitch_info[1].cpu().numpy()
    u_conf = pitch_info[2].cpu().numpy()

    # 2. HMM 기반 데이터 정제 및 수치 안정화
    u_pitch_semi = 12 * np.log2(np.where(u_pitch_hz > 0, u_pitch_hz, 1e-9) / 440.0) + 69
    
    # 전이 확률 제어 (급격한 음정 튀기 방지)
    for i in range(1, len(u_pitch_semi)):
        if abs(u_pitch_semi[i] - u_pitch_semi[i-1]) > 4 and u_conf[i] < 0.5:
            u_pitch_semi[i] = np.nan
    u_pitch_semi[u_conf < 0.3] = np.nan # 신뢰도 하위 30% 컷

    # 3. 원곡 데이터 로드 및 0값 예외 처리
    ref_data = np.load(ref_npz_path)
    r_pitch_hz = ref_data['pitch_series']
    r_pitch_hz_clean = np.where(r_pitch_hz > 10, r_pitch_hz, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        r_pitch_semi = 12 * np.log2(r_pitch_hz_clean / 440.0) + 69

    # 4. Onset 기반 초기 동기화
    u_valid = np.where(~np.isnan(u_pitch_semi))[0]
    r_valid = np.where(~np.isnan(r_pitch_semi))[0]
    if len(u_valid) == 0 or len(r_valid) == 0: return 0, 0, 0
    
    u_final = u_pitch_semi[u_valid[0]:]
    r_final = r_pitch_semi[r_valid[0]:]
    min_l = min(len(u_final), len(r_final))
    u_final, r_final = u_final[:min_l], r_final[:min_l]

    # 5. [핵심] 세그먼트별 박자 후보정 및 정량 오차 산출
    seg_size = 12 # 0.3초 단위
    search_range = 15 # 약 0.5초 탐색 범위
    hop_time = 512 / 16000 # 프레임 시간
    
    pitch_errors = []
    time_diffs_sec = []

    for i in range(search_range, min_l - seg_size - search_range, seg_size):
        u_chunk = u_final[i : i + seg_size]
        r_chunk = r_final[i : i + seg_size]
        
        # 런타임 에러 방지 (All-NaN 체크)
        if np.isnan(u_chunk).all() or np.isnan(r_chunk).all(): continue
        
        best_mae = float('inf')
        best_shift = 0
        
        for shift in range(-search_range, search_range + 1):
            r_seg = r_final[i+shift : i+shift+seg_size]
            if np.isnan(r_seg).all(): continue
            
            diff = np.abs(u_chunk - r_seg)
            mae = np.nanmean(np.minimum(diff % 12, 12 - (diff % 12)))
            if mae < best_mae:
                best_mae = mae; best_shift = shift
        
        pitch_errors.append(best_mae)
        time_diffs_sec.append(abs(best_shift) * hop_time)

    # 6. 최종 점수 환산 (Normalizing)
    if pitch_errors:
        avg_mae = np.mean(pitch_errors)
        avg_time_err = np.mean(time_diffs_sec)
        
        # 음정 정확도: 1.2 semi 오차 시 0점 (Linear Penalty)
        p_acc = round(max(0, 100 - (avg_mae * 83.3)), 2)
        
        # 박자 정확도: 평균 0.5초 지연 시 0점
        max_t = search_range * hop_time
        r_acc = round(max(0, 100 - (avg_time_err / max_t * 100)), 2)
    else:
        p_acc, r_acc, avg_time_err = 0, 0, 0

    # 7. 시각화 및 리포트 출력
    plt.figure(figsize=(12, 5))
    plt.plot(r_final, label='Original Ref', alpha=0.3, linestyle='--')
    plt.plot(u_final, label='My Vocal (Adaptive Sync)', color='red', linewidth=1)
    plt.title(f"Final Hybrid Analysis: Pitch {p_acc}% | Rhythm {r_acc}%")
    plt.ylabel("Semitone"); plt.legend(); plt.grid(True); plt.show()

    return p_acc, r_acc, avg_time_err

# 🚀 결과 출력부
if __name__ == "__main__":
    p_acc, r_acc, t_err = final_vocal_analysis_engine("my_vocal.m4a", "./vocal_features/주저하는 연인들을 위해.npz")
    
    print("\n" + "="*40)
    print(f"📊 가창 분석 최종 성적표")
    print(f"1. 음정 정확도(보정됨): {p_acc}%")
    print(f"2. 박자 정확도(지연감): {r_acc}%")
    print(f"3. 평균 박자 지연 시간: {t_err:.3f}초")
    print("="*40)
