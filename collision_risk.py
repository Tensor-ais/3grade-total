# -*- coding: utf-8 -*-
"""
객체 간 거리 변화율을 이용한 충돌 위험 예측 시스템
(Collision Risk Prediction System using Rate of Range Change)

핵심 아이디어
------------
두 이동 객체(예: 항공기, 차량, 드론 등) 사이의 '거리'와 '거리 변화율(=서로 가까워지거나
멀어지는 속도)'을 실시간으로 계산하여, 앞으로 얼마 뒤에 두 객체가 가장 가까워지는지
(TTC, Time To Closest Approach)와 그때의 최소 거리(DCPA, Distance at Closest
Point of Approach)를 예측하고, 이를 바탕으로 충돌 위험 등급을 판단한다.
이 원리는 실제 항공기의 공중충돌방지장치(TCAS)나 선박의 CPA/TCPA 계산과 동일한 개념이다.

물리적 원리
------------
객체 A, B의 위치벡터를 rA(t), rB(t), 속도벡터를 vA, vB (등속 가정)라 하면

    상대위치  r(t)   = rB(t) - rA(t)
    상대속도  v_rel  = vB - vA
    거리      D(t)   = |r(t)|
    거리변화율 dD/dt = (r(t) · v_rel) / D(t)   <- 음수면 가까워지는 중, 양수면 멀어지는 중

    TTC(최근접 시각)  = -(r(t) · v_rel) / |v_rel|^2   (v_rel != 0)
    DCPA(최근접 거리) = |r(t) + v_rel * TTC|

이 두 값(TTC, DCPA)과 현재 거리 D(t)를 조합해 아래처럼 4단계 위험 등급으로 분류한다.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 한글 폰트 자동 선택 (설치된 폰트 중 사용 가능한 것을 순서대로 시도)
_KOREAN_FONTS = ["Malgun Gothic", "AppleGothic", "NanumGothic",
                 "Noto Sans CJK KR", "Noto Sans KR", "Noto Sans CJK JP"]
_available = {f.name for f in font_manager.fontManager.ttflist}
for _font in _KOREAN_FONTS:
    if _font in _available:
        matplotlib.rcParams['font.family'] = _font
        break
matplotlib.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------------
# 1. 위험 등급 판단 기준 (임계값은 시나리오에 맞게 조정 가능)
# ------------------------------------------------------------------
SAFE_DISTANCE = 500.0   # m, 이 거리보다 멀면 기본적으로 안전
DANGER_DCPA   = 50.0    # m, 최근접 거리가 이보다 작으면 위험
CAUTION_DCPA  = 150.0   # m, 최근접 거리가 이보다 작으면 주의
TTC_LIMIT     = 20.0    # s, 이 시간 안에 최근접 상황이 온다면 긴급으로 취급


class MovingObject:
    """등속 직선 운동을 하는 객체 (위치·속도 벡터를 가짐)"""

    def __init__(self, name, position, velocity):
        self.name = name
        self.position = np.array(position, dtype=float)  # [x, y] (m)
        self.velocity = np.array(velocity, dtype=float)  # [vx, vy] (m/s)

    def position_at(self, t):
        return self.position + self.velocity * t


def compute_range_rate(objA, objB, t):
    """시각 t에서의 거리 D(t)와 거리 변화율 dD/dt를 계산"""
    r = objB.position_at(t) - objA.position_at(t)
    v_rel = objB.velocity - objA.velocity
    D = np.linalg.norm(r)
    if D < 1e-6:
        return 0.0, 0.0
    dD_dt = np.dot(r, v_rel) / D
    return D, dD_dt


def compute_ttc_dcpa(objA, objB, t0=0.0):
    """현재 시각 t0 기준으로 TTC(최근접까지 남은 시간)와 DCPA(최근접 거리)를 계산"""
    r0 = objB.position_at(t0) - objA.position_at(t0)
    v_rel = objB.velocity - objA.velocity
    v2 = np.dot(v_rel, v_rel)
    if v2 < 1e-9:  # 상대속도가 거의 0이면 거리 변화 없음
        return None, np.linalg.norm(r0)
    ttc = -np.dot(r0, v_rel) / v2
    if ttc < 0:  # 이미 최근접 지점을 지났음 (더 이상 가까워지지 않음)
        return None, np.linalg.norm(r0)
    r_at_ttc = r0 + v_rel * ttc
    dcpa = np.linalg.norm(r_at_ttc)
    return ttc, dcpa


def classify_risk(D, dD_dt, ttc, dcpa):
    """현재 거리, 거리 변화율, TTC, DCPA를 종합해 위험 등급을 4단계로 분류"""
    if dD_dt >= 0:
        return "SAFE", "서로 멀어지고 있음"
    if D <= DANGER_DCPA:
        return "DANGER", "이미 근접 위험 거리 이내"
    if ttc is not None and ttc <= TTC_LIMIT and dcpa <= DANGER_DCPA:
        return "DANGER", f"{ttc:.1f}초 후 최근접 거리 {dcpa:.1f}m로 위험"
    if ttc is not None and dcpa <= CAUTION_DCPA:
        return "WARNING", f"{ttc:.1f}초 후 최근접 거리 {dcpa:.1f}m로 근접 예상"
    if D <= SAFE_DISTANCE:
        return "CAUTION", "접근 중이나 아직 여유 있음"
    return "SAFE", "충분히 먼 거리에서 접근 중"


def simulate(objA, objB, t_end=60.0, dt=0.5):
    """0초부터 t_end초까지 dt 간격으로 시뮬레이션하며 결과를 기록"""
    times, distances, rates, risk_levels = [], [], [], []
    for t in np.arange(0, t_end + dt, dt):
        D, dD_dt = compute_range_rate(objA, objB, t)
        ttc, dcpa = compute_ttc_dcpa(objA, objB, t)
        risk, reason = classify_risk(D, dD_dt, ttc, dcpa)
        times.append(t)
        distances.append(D)
        rates.append(dD_dt)
        risk_levels.append(risk)
    return {
        "t": np.array(times),
        "D": np.array(distances),
        "rate": np.array(rates),
        "risk": risk_levels,
    }


def print_summary(name, objA, objB, result):
    print(f"\n===== 시나리오: {name} =====")
    ttc0, dcpa0 = compute_ttc_dcpa(objA, objB, 0.0)
    D0, rate0 = compute_range_rate(objA, objB, 0.0)
    print(f"초기 거리 D0     : {D0:.1f} m")
    print(f"초기 거리 변화율 : {rate0:.2f} m/s ({'접근 중' if rate0 < 0 else '이격 중'})")
    if ttc0 is not None:
        print(f"예상 최근접 시각 TTC : {ttc0:.1f} s")
        print(f"예상 최근접 거리 DCPA: {dcpa0:.1f} m")
    else:
        print("최근접 지점을 향해 접근하고 있지 않음 (또는 이미 지남)")

    # 위험 등급이 바뀌는 시점만 출력
    prev = None
    for t, risk in zip(result["t"], result["risk"]):
        if risk != prev:
            print(f"  t={t:5.1f}s -> 위험 등급: {risk}")
            prev = risk


def plot_scenario(name, result, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    color_map = {"SAFE": "#2ecc71",
                 "CAUTION": "#f1c40f",
                 "WARNING": "#e67e22",
                 "DANGER": "#e74c3c"}
    colors = [color_map[r] for r in result["risk"]]

    ax1.scatter(result["t"], result["D"], c=colors, s=12)
    ax1.plot(result["t"], result["D"], color="gray", alpha=0.3, linewidth=1)
    ax1.axhline(DANGER_DCPA, color="#e74c3c", linestyle="--", linewidth=0.8, label="위험 임계거리")
    ax1.axhline(CAUTION_DCPA, color="#e67e22", linestyle="--", linewidth=0.8, label="주의 임계거리")
    ax1.set_ylabel("거리 D(t) [m]")
    ax1.set_title(f"[{name}] 거리 및 위험 등급 변화")
    ax1.legend(loc="upper right", fontsize=8)

    ax2.plot(result["t"], result["rate"], color="#3498db")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylabel("거리 변화율 dD/dt [m/s]")
    ax2.set_xlabel("시간 t [s]")

    # 범례용 더미 산점도
    for risk, c in color_map.items():
        ax1.scatter([], [], c=c, label=risk)
    ax1.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(filename, dpi=130)
    plt.close()
    print(f"그래프 저장 완료: {filename}")


if __name__ == "__main__":
    # 시나리오 1: 안전 - 두 객체가 서로 다른 방향으로 멀어짐
    A1 = MovingObject("A", position=[0, 0], velocity=[10, 0])
    B1 = MovingObject("B", position=[0, 500], velocity=[0, 10])
    r1 = simulate(A1, B1)
    print_summary("SAFE - 발산 경로", A1, B1, r1)
    plot_scenario("SAFE - 발산 경로", r1, "scenario1_safe.png")

    # 시나리오 2: 주의/경고 - 두 객체가 비스듬히 교차하되 완전히 정면충돌은 아님(근접 통과)
    A2 = MovingObject("A", position=[-300, -160], velocity=[15, 0])
    B2 = MovingObject("B", position=[40, -300], velocity=[0, 12])
    r2 = simulate(A2, B2)
    print_summary("WARNING - 교차 경로", A2, B2, r2)
    plot_scenario("WARNING - 교차 경로", r2, "scenario2_warning.png")

    # 시나리오 3: 위험 - 두 객체가 정면으로 접근 (거의 정면 충돌 코스)
    A3 = MovingObject("A", position=[-400, 2], velocity=[20, 0])
    B3 = MovingObject("B", position=[400, -2], velocity=[-20, 0])
    r3 = simulate(A3, B3)
    print_summary("DANGER - 정면 접근", A3, B3, r3)
    plot_scenario("DANGER - 정면 접근", r3, "scenario3_danger.png")
