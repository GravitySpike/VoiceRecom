import numpy as np
import librosa

def analyze_vocal_stability(audio_path, target_note):
    """
    특정 가이드 음(target_note)에 대한 사용자의 가창 안정성을 분석합니다.
    """
    try:
        # 1. 음성 로드 및 기본 주파수(F0) 추출
        y, sr = librosa.load(audio_path, sr=22050)
        
        # 가이드 음을 주파수(Hz)로 변환
        target_hz = librosa.note_to_hz(target_note)
        
        # pYIN 알고리즘으로 정밀 피치 추출
        f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                                     fmin=librosa.note_to_hz('C2'), 
                                                     fmax=librosa.note_to_hz('C7'))
        
        # 유효한 음성 구간만 필터링
        valid_pitches = f0[voiced_flag & (f0 > 0)]
        
        if len(valid_pitches) == 0:
            return {"status": "fail", "reason": "음성이 감지되지 않음"}

        # 2. 오차 및 안정성 계산
        # 가이드 음과의 평균 편차 (Cents 단위)
        avg_pitch = np.mean(valid_pitches)
        pitch_error_cents = 1200 * np.log2(avg_pitch / target_hz)
        
        # 음정의 떨림(표준편차) - 이 값이 크면 한계에 도달한 것
        pitch_std = np.std(valid_pitches)
        
        # 3. 한계 판정 로직
        # 음정 오차가 50센트(반음의 절반) 이상이거나 떨림이 심하면 '한계'로 간주
        is_stable = abs(pitch_error_cents) < 50 and pitch_std < 5.0
        
        return {
            "status": "success" if is_stable else "limit",
            "note": target_note,
            "error_cents": round(pitch_error_cents, 2),
            "stability": round(pitch_std, 2)
        }

    except Exception as e:
        print(f"❌ 분석 실패: {e}")
        return None

# 실행 예시
# result = analyze_vocal_stability("user_record_C4.wav", "C4")
# print(f"결과: {result['status']} (오차: {result['error_cents']} cents, 불안정도: {result['stability']})")
