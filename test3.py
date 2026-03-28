import numpy as np
import librosa
import pesto
import torch
import matplotlib.pyplot as plt

def final_vocal_analysis_engine(audio_path, ref_npz_path):
    print(f"🚀 [System] 엔드 트림 최적화 엔진 가동: {audio_path}")
    
    # 1. 오디오 로드 및 PESTO 피치 추출 (16kHz)
    y, sr = librosa.load(audio_path, sr=16000)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_tensor = torch.from_numpy(y).float().to(device)
    
    pitch_info = pesto.predict(y_tensor, sr)
    u_pitch_hz, u_conf = pitch_info[1].cpu().numpy(), pitch_info[2].cpu().numpy()

    # 2. 세미톤 변환 및 [튜닝] Cutoff 0.33 적용
    u_pitch_semi = 12 * np.log2(np.where(u_pitch_hz > 0, u_pitch_hz, 1e-9) / 440.0) + 69
    u_pitch_semi[u_conf < 0.33] = np.nan 

    # 3. 원곡 데이터 로드 및 전처리
    ref_data = np.load(ref_npz_path)
    r_pitch_hz = ref_data['pitch_series']
    with np.errstate(divide='ignore', invalid='ignore'):
        r_pitch_semi = 12 * np.log2(np.where(r_pitch_hz > 10, r_pitch_hz, np.nan) / 440.0) + 69

    # 4. 동기화 및 길이 정렬
    u_valid, r_valid = np.where(~np.isnan(u_pitch_semi))[0], np.where(~np.isnan(r_pitch_semi))[0]
    if len(u_valid) == 0 or len(r_valid) == 0: return 0, 0
    
    u_final = u_pitch_semi[u_valid[0]:]
    r_final = r_pitch_semi[r_valid[0]:]
    min_l = min(len(u_final), len(r_final))
    u_final, r_final = u_final[:min_l], r_final[:min_l]

    # 5. [추가] 가이드 음원 종료 시점 감지 (End-point Trimming)
    r_active_indices = np.where(~np.isnan(r_final))[0]
    last_r_idx = r_active_indices[-1] if len(r_active_indices) > 0 else min_l

    # 6. 세그먼트 파라미터
    seg_size, search_range, hop_time = 12, 30, 512 / 16000 
    correlations, rhythm_scores = [], []

    # 7. 정밀 루프 분석 (가이드 종료 시점까지만 실행)
    # 가이드가 끝난 뒤의 유저 가창 데이터는 분석 분모에서 제외
    loop_limit = min(min_l - seg_size - search_range, last_r_idx - seg_size)

    for i in range(search_range, loop_limit, seg_size):
        u_chunk = u_final[i : i + seg_size]
        if np.isnan(u_chunk).all(): continue 
        
        best_corr, best_shift = -1.0, 0
        for shift in range(-search_range, search_range + 1):
            r_seg = r_final[i+shift : i+shift+seg_size]
            mask = ~np.isnan(u_chunk) & ~np.isnan(r_seg)
            if np.sum(mask) < (seg_size * 0.3): continue
            
            corr_val = np.corrcoef(u_chunk[mask], r_seg[mask])[0, 1]
            if not np.isnan(corr_val) and corr_val > best_corr:
                best_corr, best_shift = corr_val, shift
        
        if best_corr > -1.0:
            correlations.append(best_corr)
            time_offset = abs(best_shift) * hop_time
            
            # [튜닝] 리듬 관용도 모델 (Deadzone 0.1s, k=1.0)
            if time_offset <= 0.1:
                seg_r_score = 100.0
            else:
                seg_r_score = 100.0 * np.exp(-1.0 * (time_offset - 0.1))
            rhythm_scores.append(max(0, seg_r_score))

    # 8. 최종 결과 산출
    if correlations:
        avg_corr = np.mean(correlations)
        # 상관계수의 제곱을 취해 고득점 허들을 높임 (0.9 -> 81점)
        pitch_acc = round((avg_corr ** 2) * 100, 2)
    else:
        pitch_acc = 0
    rhythm_acc = round(np.mean(rhythm_scores), 2) if rhythm_scores else 0

    # 9. 시각화 (색상 및 스타일 개선)
    plt.figure(figsize=(14, 7), facecolor='#F8F9FA')
    r_plot = r_final[:last_r_idx]
    u_plot = u_final[:last_r_idx]
    plt.plot(r_plot, label='Original Reference', alpha=0.5, linestyle='--', color='deepskyblue')
    plt.plot(u_plot, label='My Optimized Vocal', color='crimson', linewidth=1.5)
    
    res_text = (f"★ GRADUATION PROJECT: FINAL ★\n"
                f"-----------------------------\n"
                f"PITCH ACC  : {pitch_acc}%\n"
                f"RHYTHM ACC : {rhythm_acc}%")
    
    plt.gca().text(0.02, 0.98, res_text, transform=plt.gca().transAxes, 
                   fontsize=12, family='monospace', fontweight='bold',
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.title(f"Vocal Performance Analysis Engine", fontsize=16, pad=20)
    plt.ylabel("Pitch (Semitone)"); plt.xlabel("Time Frames")
    plt.grid(True, linestyle=':', alpha=0.6); plt.legend(loc='upper right')
    plt.tight_layout(); plt.show()

    return pitch_acc, rhythm_acc

if __name__ == "__main__":
    final_vocal_analysis_engine("my_vocal.m4a", "./vocal_features/주저하는 연인들을 위해.npz")
