import numpy as np
import librosa
import pesto
import torch
import matplotlib.pyplot as plt

def final_vocal_analysis_engine(audio_path, ref_npz_path):
    print(f"🚀 [System] 상관계수 패턴 매칭 엔진 가동: {audio_path}")
    
    # 1. 오디오 로드 및 PESTO 추출 (16kHz 통일)
    y, sr = librosa.load(audio_path, sr=16000)
    y_tensor = torch.from_numpy(y).float().to("cuda" if torch.cuda.is_available() else "cpu")
    
    pitch_info = pesto.predict(y_tensor, sr)
    u_pitch_hz = pitch_info[1].cpu().numpy()
    u_conf = pitch_info[2].cpu().numpy()

    # 2. 세미톤 변환 및 노이즈 필터링
    u_pitch_semi = 12 * np.log2(np.where(u_pitch_hz > 0, u_pitch_hz, 1e-9) / 440.0) + 69
    u_pitch_semi[u_conf < 0.4] = np.nan # 신뢰도 기준 상향

    # 3. 원곡 데이터 로드
    ref_data = np.load(ref_npz_path)
    r_pitch_hz = ref_data['pitch_series']
    r_pitch_hz_clean = np.where(r_pitch_hz > 10, r_pitch_hz, np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        r_pitch_semi = 12 * np.log2(r_pitch_hz_clean / 440.0) + 69

    # 4. 동기화
    u_valid = np.where(~np.isnan(u_pitch_semi))[0]
    r_valid = np.where(~np.isnan(r_pitch_semi))[0]
    if len(u_valid) == 0 or len(r_valid) == 0: return 0, 0, 0
    
    u_final = u_pitch_semi[u_valid[0]:]
    r_final = r_pitch_semi[r_valid[0]:]
    min_l = min(len(u_final), len(r_final))
    u_final, r_final = u_final[:min_l], r_final[:min_l]

    # 5. [핵심 수정] 상관계수 기반 세그먼트 매칭
    seg_size = 12 
    search_range = 15 
    hop_time = 512 / 16000 
    
    correlations = []
    time_diffs_sec = []

    for i in range(search_range, min_l - seg_size - search_range, seg_size):
        u_chunk = u_final[i : i + seg_size]
        if np.isnan(u_chunk).all(): continue
        
        best_corr = -1.0 # 상관계수 최솟값에서 시작
        best_shift = 0
        
        for shift in range(-search_range, search_range + 1):
            r_seg = r_final[i+shift : i+shift+seg_size]
            if np.isnan(r_seg).all(): continue
            
            # 유효 데이터 필터링
            mask = ~np.isnan(u_chunk) & ~np.isnan(r_seg)
            if np.sum(mask) < (seg_size / 2): continue
            
            # 피어슨 상관계수 산출 (패턴 유사도 측정)
            # 절대값이 달라도 모양이 같으면 1.0에 수렴함
            corr_val = np.corrcoef(u_chunk[mask], r_seg[mask])[0, 1]
            
            if not np.isnan(corr_val) and corr_val > best_corr:
                best_corr = corr_val; best_shift = shift
        
        if best_corr > -1.0:
            correlations.append(best_corr)
            time_diffs_sec.append(abs(best_shift) * hop_time)

    # 6. 최종 점수 환산 (상관계수 -> 백분율)
    if correlations:
        avg_corr = np.mean(correlations)
        avg_time_err = np.mean(time_diffs_sec)
        
        # 음정 정확도: 상관계수 1.0이면 100점, 0 이하면 0점
        p_acc = round(max(0, avg_corr * 100), 2)
        
        # 박자 정확도
        max_t = search_range * hop_time
        r_acc = round(max(0, 100 - (avg_time_err / max_t * 100)), 2)
    else:
        p_acc, r_acc, avg_time_err = 0, 0, 0

    # 7. 시각화
    plt.figure(figsize=(12, 6))
    plt.plot(r_final, label='Reference (Original)', alpha=0.3, linestyle='--')
    plt.plot(u_final, label='User Vocal', color='red', linewidth=1)
    
    result_text = (f"PITCH ACC (Pattern): {p_acc}%\n"
                   f"RHYTHM ACC: {r_acc}%\n"
                   f"AVG DELAY: {avg_time_err:.3f}s")
    
    plt.gca().text(0.05, 0.95, result_text, transform=plt.gca().transAxes, 
                   fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title(f"Correlation-based Analysis Report", fontsize=15)
    plt.ylabel("Pitch (Semitone)")
    plt.grid(True); plt.legend(loc='upper right'); plt.show()

    return p_acc, r_acc, avg_time_err

if __name__ == "__main__":
    # 16k로 다시 뽑은 .npz 파일을 사용하세요
    final_vocal_analysis_engine("my_vocal.m4a", "./vocal_features/주저하는 연인들을 위해.npz")
