import pesto
import inspect # 함수 내부 정보를 들여다보는 표준 라이브러리

print("--- 1. PESTO 모듈 구성 요소 ---")
print(dir(pesto))

# predict가 모듈인지 함수인지 확인
print("\n--- 2. pesto.predict 타입 확인 ---")
print(type(pesto.predict))

# 함수의 입력 인자(Parameters)와 반환값 구조 확인
print("\n--- 3. pesto.predict 상세 정보 (인자 리스트) ---")
try:
    # inspect.signature는 함수의 파라미터 명칭과 기본값을 보여줍니다.
    sig = inspect.signature(pesto.predict)
    print(sig)
except Exception as e:
    print(f"인자 정보를 가져올 수 없습니다: {e}")

# 도움말(Docstring) 출력 - 개발자가 적어놓은 사용 설명서
print("\n--- 4. pesto.predict 도움말 ---")
print(pesto.predict.__doc__)
