"""
환승연애2 - 3개 케이스 비교 분석 코드
===========================================
케이스 1: 나연-희두 (X 재결합 성공)
케이스 2: 해은-현규 (환승 성공)
케이스 3: 정규민 (실패 - 혼자 나옴)

분석 목표: 각 케이스의 차이점을 정량적으로 규명
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# STEP 1: 데이터 로드
# =============================================================================

print("="*70)
print("STEP 1: 데이터 로드")
print("="*70)

# 기본 데이터 로드
char_profile = pd.read_csv('/mnt/user-data/uploads/character_profile.csv')
message_log = pd.read_csv('/mnt/user-data/uploads/message_log.csv', encoding='utf-8-sig')
episode_log = pd.read_csv('/mnt/user-data/uploads/episode_event_log.csv')
haeeun_gyumin = pd.read_csv('/mnt/user-data/uploads/환연2_해은규민_pair_ai.csv', encoding='utf-8-sig')
nayeon_hidu = pd.read_csv('/mnt/user-data/uploads/환연2_희두나연_pair_ai.csv', encoding='utf-8-sig')

# 주요 인물 정보 확인
print("\n주요 인물 프로필:")
target_people = char_profile[char_profile['name'].isin(['성해은', '정규민', '정현규', '이나연', '남희두'])]
print(target_people[['name', 'mbti', 'relationship_months', 'breakup_months', 'entry_episode']])

print("\n✓ 데이터 로드 완료")

# =============================================================================
# STEP 2: 3개 케이스 정의 및 결과 변수 설정
# =============================================================================

print("\n" + "="*70)
print("STEP 2: 3개 케이스 정의")
print("="*70)

# 케이스 정의
cases = {
    'Case1_Reunion': {
        'name': '나연-희두 (X 재결합)',
        'people': ['이나연', '남희두'],
        'result': 'X_REUNION',
        'description': 'X 파트너끼리 다시 만남',
        'color': '#2ecc71'  # 녹색
    },
    'Case2_Transfer': {
        'name': '해은-현규 (환승 성공)',
        'people': ['성해은', '정현규'],
        'result': 'NEW_RELATIONSHIP',
        'description': '새로운 사람과 관계 시작',
        'color': '#3498db'  # 파란색
    },
    'Case3_Failed': {
        'name': '정규민 (실패)',
        'people': ['정규민'],
        'result': 'ALONE',
        'description': '아무도 선택 못함',
        'color': '#e74c3c'  # 빨간색
    }
}

for case_id, case_info in cases.items():
    print(f"\n[{case_id}] {case_info['name']}")
    print(f"  인물: {', '.join(case_info['people'])}")
    print(f"  결과: {case_info['result']}")
    print(f"  설명: {case_info['description']}")

print("\n✓ 케이스 정의 완료")

# =============================================================================
# STEP 3: 분석 지표 설계 (주인님이 직접 수정 가능)
# =============================================================================

print("\n" + "="*70)
print("STEP 3: 분석 지표 설계")
print("="*70)

# 분석할 지표 리스트 (주인님이 추가/삭제 가능)
metrics = {
    '1_문자_발송_패턴': {
        'description': '속마음 문자를 누구에게 얼마나 보냈는가',
        'method': 'message_log 분석'
    },
    '2_미련도_점수': {
        'description': '미련 관련 키워드 빈도',
        'method': 'text mining (보고싶, 그리워, 후회 등)'
    },
    '3_감정_변화': {
        'description': '긍정/부정 감정 비율 및 변화',
        'method': 'sentiment 분석'
    },
    '4_관심_집중도': {
        'description': 'attention 카테고리 발화 빈도',
        'method': 'category 분석'
    },
    '5_MBTI_조합': {
        'description': 'MBTI 유형 및 J-P 조합',
        'method': '프로필 데이터 매칭'
    },
    '6_투입_타이밍': {
        'description': '메기 투입 시점 및 영향도',
        'method': 'entry_episode 분석'
    }
}

for metric_id, metric_info in metrics.items():
    print(f"\n[{metric_id}]")
    print(f"  지표: {metric_info['description']}")
    print(f"  방법: {metric_info['method']}")

print("\n✓ 지표 설계 완료")

# =============================================================================
# STEP 4: 지표별 분석 함수 (주인님이 직접 실행)
# =============================================================================

print("\n" + "="*70)
print("STEP 4: 분석 함수 정의")
print("="*70)

# ---------------------------------------------------------------------------
# 함수 1: 속마음 문자 패턴 분석
# ---------------------------------------------------------------------------

def analyze_message_pattern(person_name, message_df):
    """
    특정 인물의 속마음 문자 발송 패턴 분석
    
    Args:
        person_name: 분석 대상 이름 (예: '성해은')
        message_df: message_log 데이터프레임
    
    Returns:
        dict: 발송 통계
    """
    person_messages = message_df[
        (message_df['name'] == person_name) & 
        (message_df['message_txt'].notna())
    ]
    
    # 수신자별 분류
    recipients = person_messages['message'].value_counts().to_dict()
    
    result = {
        'total_sent': len(person_messages),
        'recipients': recipients,
        'unique_recipients': len(recipients)
    }
    
    return result

# 사용 예시
print("\n[함수 1: 속마음 문자 패턴]")
haeeun_msg = analyze_message_pattern('성해은', message_log)
print(f"성해은 문자 발송: {haeeun_msg['total_sent']}회")
print(f"수신자: {haeeun_msg['recipients']}")

# ---------------------------------------------------------------------------
# 함수 2: 미련도 점수 계산
# ---------------------------------------------------------------------------

def calculate_regret_score(person_name, conversation_df):
    """
    미련도 점수 계산 (0-100점)
    
    Args:
        person_name: 분석 대상 이름
        conversation_df: 대화 데이터프레임 (환연2_xxx_pair_ai.csv)
    
    Returns:
        dict: 미련도 점수 및 세부 내용
    """
    # 미련 키워드 정의 (주인님이 추가 가능)
    regret_keywords = ['보고 싶', '그리', '못 잊', '후회', '미련', '아쉬', '돌아', '다시']
    
    # 해당 인물의 발화만 필터링
    person_text = conversation_df[conversation_df['speaker'] == person_name]['text'].fillna('').str.lower()
    
    # 키워드별 빈도 계산
    keyword_counts = {}
    total_count = 0
    
    for keyword in regret_keywords:
        count = person_text.str.contains(keyword, na=False).sum()
        if count > 0:
            keyword_counts[keyword] = count
            total_count += count
    
    # 미련도 점수 = (키워드 총 빈도 / 총 발화 수) * 100
    total_utterances = len(person_text)
    regret_score = (total_count / total_utterances * 100) if total_utterances > 0 else 0
    
    result = {
        'score': round(regret_score, 2),
        'total_keywords': total_count,
        'keyword_detail': keyword_counts,
        'total_utterances': total_utterances
    }
    
    return result

# 사용 예시
print("\n[함수 2: 미련도 점수]")
haeeun_regret = calculate_regret_score('성해은', haeeun_gyumin)
print(f"성해은 미련도: {haeeun_regret['score']}점")
print(f"키워드 상세: {haeeun_regret['keyword_detail']}")

# ---------------------------------------------------------------------------
# 함수 3: 감정 분포 분석
# ---------------------------------------------------------------------------

def analyze_sentiment_distribution(person_name, conversation_df):
    """
    감정 분포 분석 (positive/negative/neutral 비율)
    
    Args:
        person_name: 분석 대상 이름
        conversation_df: 대화 데이터프레임
    
    Returns:
        dict: 감정 분포 통계
    """
    person_sentiment = conversation_df[conversation_df['speaker'] == person_name]['sentiment']
    
    # 감정 카테고리 통합
    sentiment_map = {
        '중립': 'neutral',
        '복합적': 'mixed',
        '궁금': 'neutral',
        '짜증': 'negative',
        '후회': 'negative'
    }
    person_sentiment = person_sentiment.replace(sentiment_map)
    
    # 비율 계산
    total = len(person_sentiment)
    distribution = person_sentiment.value_counts()
    
    result = {
        'positive_ratio': (distribution.get('positive', 0) / total * 100) if total > 0 else 0,
        'negative_ratio': (distribution.get('negative', 0) / total * 100) if total > 0 else 0,
        'neutral_ratio': (distribution.get('neutral', 0) / total * 100) if total > 0 else 0,
        'distribution': distribution.to_dict()
    }
    
    return result

# 사용 예시
print("\n[함수 3: 감정 분포]")
haeeun_sentiment = analyze_sentiment_distribution('성해은', haeeun_gyumin)
print(f"성해은 감정 비율:")
print(f"  긍정: {haeeun_sentiment['positive_ratio']:.1f}%")
print(f"  부정: {haeeun_sentiment['negative_ratio']:.1f}%")
print(f"  중립: {haeeun_sentiment['neutral_ratio']:.1f}%")

# ---------------------------------------------------------------------------
# 함수 4: Attention 집중도 분석
# ---------------------------------------------------------------------------

def analyze_attention_focus(person_name, conversation_df):
    """
    상대방에 대한 관심/집착 표현 빈도 분석
    
    Args:
        person_name: 분석 대상 이름
        conversation_df: 대화 데이터프레임
    
    Returns:
        dict: attention 통계
    """
    person_conv = conversation_df[conversation_df['speaker'] == person_name]
    attention_utterances = person_conv[person_conv['category'] == 'attention']
    
    total_utterances = len(person_conv)
    attention_count = len(attention_utterances)
    
    result = {
        'attention_count': attention_count,
        'total_utterances': total_utterances,
        'attention_ratio': (attention_count / total_utterances * 100) if total_utterances > 0 else 0
    }
    
    return result

# 사용 예시
print("\n[함수 4: Attention 집중도]")
haeeun_attention = analyze_attention_focus('성해은', haeeun_gyumin)
print(f"성해은 attention: {haeeun_attention['attention_count']}회 ({haeeun_attention['attention_ratio']:.1f}%)")

print("\n✓ 분석 함수 정의 완료")

# =============================================================================
# STEP 5: 3개 케이스 종합 비교 (주인님이 실행)
# =============================================================================

print("\n" + "="*70)
print("STEP 5: 3개 케이스 종합 비교 실행")
print("="*70)

# 결과 저장용 데이터프레임 초기화
comparison_data = []

# Case 1: 나연-희두 분석
print("\n[Case 1: 나연-희두 분석 중...]")
nayeon_data = {
    'case': 'X 재결합',
    'person': '이나연',
    'mbti': 'ESFP',
    'message_to_x': len(message_log[(message_log['name']=='이나연') & (message_log['message']=='남희두') & (message_log['message_txt'].notna())]),
    'regret_score': calculate_regret_score('이나연', nayeon_hidu)['score'],
    'negative_ratio': analyze_sentiment_distribution('이나연', nayeon_hidu)['negative_ratio'],
    'attention_ratio': analyze_attention_focus('이나연', nayeon_hidu)['attention_ratio'],
    'relationship_months': 31,
    'result_color': cases['Case1_Reunion']['color']
}
comparison_data.append(nayeon_data)

hidu_data = {
    'case': 'X 재결합',
    'person': '남희두',
    'mbti': 'INTP',
    'message_to_x': len(message_log[(message_log['name']=='남희두') & (message_log['message']=='이나연') & (message_log['message_txt'].notna())]),
    'regret_score': calculate_regret_score('남희두', nayeon_hidu)['score'],
    'negative_ratio': analyze_sentiment_distribution('남희두', nayeon_hidu)['negative_ratio'],
    'attention_ratio': analyze_attention_focus('남희두', nayeon_hidu)['attention_ratio'],
    'relationship_months': 31,
    'result_color': cases['Case1_Reunion']['color']
}
comparison_data.append(hidu_data)

# Case 2: 해은-현규 분석
print("[Case 2: 해은 분석 중...]")
# 주의: 현규 대화 데이터가 없으므로 해은 데이터만 분석
haeeun_data = {
    'case': '환승 성공',
    'person': '성해은',
    'mbti': 'INFP',
    'message_to_x': len(message_log[(message_log['name']=='성해은') & (message_log['message']=='정규민') & (message_log['message_txt'].notna())]),
    'message_to_new': len(message_log[(message_log['name']=='성해은') & (message_log['message']=='정현규') & (message_log['message_txt'].notna())]),
    'regret_score': calculate_regret_score('성해은', haeeun_gyumin)['score'],
    'negative_ratio': analyze_sentiment_distribution('성해은', haeeun_gyumin)['negative_ratio'],
    'attention_ratio': analyze_attention_focus('성해은', haeeun_gyumin)['attention_ratio'],
    'relationship_months': 76,
    'result_color': cases['Case2_Transfer']['color']
}
comparison_data.append(haeeun_data)

# Case 3: 정규민 분석
print("[Case 3: 정규민 분석 중...]")
gyumin_data = {
    'case': '실패',
    'person': '정규민',
    'mbti': 'ENTJ',
    'message_to_x': len(message_log[(message_log['name']=='정규민') & (message_log['message']=='성해은') & (message_log['message_txt'].notna())]),
    'regret_score': calculate_regret_score('정규민', haeeun_gyumin)['score'],
    'negative_ratio': analyze_sentiment_distribution('정규민', haeeun_gyumin)['negative_ratio'],
    'attention_ratio': analyze_attention_focus('정규민', haeeun_gyumin)['attention_ratio'],
    'relationship_months': 76,
    'result_color': cases['Case3_Failed']['color']
}
comparison_data.append(gyumin_data)

# 데이터프레임 생성
df_comparison = pd.DataFrame(comparison_data)

print("\n✓ 종합 비교 데이터 생성 완료")
print("\n비교 테이블:")
print(df_comparison[['case', 'person', 'mbti', 'message_to_x', 'regret_score', 'negative_ratio']])

# =============================================================================
# STEP 6: 시각화 (주인님이 커스터마이징 가능)
# =============================================================================

print("\n" + "="*70)
print("STEP 6: 시각화 생성")
print("="*70)

# 시각화 1: 3개 케이스 종합 비교 대시보드
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1-1. 문자 발송 비교
ax1 = fig.add_subplot(gs[0, 0])
people_order = ['이나연', '남희두', '성해은', '정규민']
message_counts = [nayeon_data['message_to_x'], hidu_data['message_to_x'], 
                  haeeun_data['message_to_x'], gyumin_data['message_to_x']]
colors = [nayeon_data['result_color'], hidu_data['result_color'],
          haeeun_data['result_color'], gyumin_data['result_color']]

bars = ax1.bar(people_order, message_counts, color=colors, alpha=0.8, edgecolor='black')
ax1.set_title('Messages to X-Partner', fontweight='bold', fontsize=11)
ax1.set_ylabel('Count')
for bar, count in zip(bars, message_counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             str(count), ha='center', fontweight='bold')

# 1-2. 미련도 점수 비교
ax2 = fig.add_subplot(gs[0, 1])
regret_scores = [nayeon_data['regret_score'], hidu_data['regret_score'],
                 haeeun_data['regret_score'], gyumin_data['regret_score']]

bars = ax2.bar(people_order, regret_scores, color=colors, alpha=0.8, edgecolor='black')
ax2.set_title('Regret Score', fontweight='bold', fontsize=11)
ax2.set_ylabel('Score')
for bar, score in zip(bars, regret_scores):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
             f'{score:.1f}', ha='center', fontweight='bold')

# 1-3. 부정 감정 비율
ax3 = fig.add_subplot(gs[0, 2])
negative_ratios = [nayeon_data['negative_ratio'], hidu_data['negative_ratio'],
                   haeeun_data['negative_ratio'], gyumin_data['negative_ratio']]

bars = ax3.bar(people_order, negative_ratios, color=colors, alpha=0.8, edgecolor='black')
ax3.set_title('Negative Emotion Ratio', fontweight='bold', fontsize=11)
ax3.set_ylabel('Percentage (%)')
for bar, ratio in zip(bars, negative_ratios):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{ratio:.1f}%', ha='center', fontweight='bold')

# 2-1. Attention 집중도
ax4 = fig.add_subplot(gs[1, 0])
attention_ratios = [nayeon_data['attention_ratio'], hidu_data['attention_ratio'],
                    haeeun_data['attention_ratio'], gyumin_data['attention_ratio']]

bars = ax4.bar(people_order, attention_ratios, color=colors, alpha=0.8, edgecolor='black')
ax4.set_title('Attention Focus Ratio', fontweight='bold', fontsize=11)
ax4.set_ylabel('Percentage (%)')
for bar, ratio in zip(bars, attention_ratios):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{ratio:.1f}%', ha='center', fontweight='bold')

# 2-2. 케이스별 그룹 비교 (레이더 차트)
ax5 = fig.add_subplot(gs[1, 1:], projection='polar')

categories = ['Message\nto X', 'Regret\nScore', 'Negative\nRatio', 'Attention\nRatio']
N = len(categories)

# 정규화 (0-100 스케일)
def normalize(values, max_val=None):
    if max_val is None:
        max_val = max(values)
    return [(v / max_val * 100) if max_val > 0 else 0 for v in values]

# 케이스별 평균값 계산
case1_values = [
    np.mean([nayeon_data['message_to_x'], hidu_data['message_to_x']]),
    np.mean([nayeon_data['regret_score'], hidu_data['regret_score']]),
    np.mean([nayeon_data['negative_ratio'], hidu_data['negative_ratio']]),
    np.mean([nayeon_data['attention_ratio'], hidu_data['attention_ratio']])
]

case2_values = [
    haeeun_data['message_to_x'],
    haeeun_data['regret_score'],
    haeeun_data['negative_ratio'],
    haeeun_data['attention_ratio']
]

case3_values = [
    gyumin_data['message_to_x'],
    gyumin_data['regret_score'],
    gyumin_data['negative_ratio'],
    gyumin_data['attention_ratio']
]

# 정규화
max_vals = [10, 15, 50, 50]  # 각 지표의 최대값
case1_norm = [v/m*100 for v, m in zip(case1_values, max_vals)]
case2_norm = [v/m*100 for v, m in zip(case2_values, max_vals)]
case3_norm = [v/m*100 for v, m in zip(case3_values, max_vals)]

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
case1_norm += case1_norm[:1]
case2_norm += case2_norm[:1]
case3_norm += case3_norm[:1]
angles += angles[:1]

ax5.plot(angles, case1_norm, 'o-', linewidth=2, label='X Reunion', color=cases['Case1_Reunion']['color'])
ax5.fill(angles, case1_norm, alpha=0.25, color=cases['Case1_Reunion']['color'])

ax5.plot(angles, case2_norm, 'o-', linewidth=2, label='Transfer', color=cases['Case2_Transfer']['color'])
ax5.fill(angles, case2_norm, alpha=0.25, color=cases['Case2_Transfer']['color'])

ax5.plot(angles, case3_norm, 'o-', linewidth=2, label='Failed', color=cases['Case3_Failed']['color'])
ax5.fill(angles, case3_norm, alpha=0.25, color=cases['Case3_Failed']['color'])

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(categories, fontsize=9)
ax5.set_ylim(0, 100)
ax5.set_title('Case Comparison (Radar)', fontweight='bold', fontsize=11, pad=20)
ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax5.grid(True)

# 3. 케이스별 요약 텍스트
ax6 = fig.add_subplot(gs[2, :])
ax6.axis('off')

summary_text = f"""
3-CASE COMPARISON SUMMARY

Case 1 (X Reunion): 나연-희두
• Message Balance: 6 vs 7 (균형)
• Regret Score: {np.mean([nayeon_data['regret_score'], hidu_data['regret_score']]):.1f} (중간)
• MBTI: ESFP-INTP (P-P 조합)
• Result: SUCCESS - 서로에 대한 균형 잡힌 관심

Case 2 (Transfer): 성해은 → 정현규
• Message to X: {haeeun_data['message_to_x']} (높음), to New: {haeeun_data.get('message_to_new', 0)} (낮음)
• Regret Score: {haeeun_data['regret_score']:.1f} (최고치)
• MBTI: INFP (Fi 주기능 - 과거 이상화)
• Result: SUCCESS - X는 포기하고 새 인연 선택

Case 3 (Failed): 정규민
• Message to X: {gyumin_data['message_to_x']} (0회!)
• Regret Score: {gyumin_data['regret_score']:.1f} (낮음)
• MBTI: ENTJ (Te 주기능 - 합리적 종료)
• Result: FAILED - 해은에게 관심 없음, 다른 사람과도 실패

KEY INSIGHT:
균형 잡힌 관심(Case1) > 일방적 미련(Case2) > 무관심(Case3)
"""

ax6.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('3-CASE ANALYSIS: X Reunion vs Transfer vs Failed', 
             fontsize=14, fontweight='bold', y=0.98)

plt.savefig('/home/claude/07_three_case_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ 저장: 07_three_case_comparison.png")

print("\n" + "="*70)
print("모든 분석 완료!")
print("="*70)

# =============================================================================
# 주인님을 위한 추가 분석 아이디어
# =============================================================================

print("\n" + "="*70)
print("추가 분석 아이디어 (주인님이 직접 시도해보세요!)")
print("="*70)

analysis_ideas = """
1. 성해은의 문자 전환 분석
   → 정규민에게 보낸 10회 vs 정현규에게 보낸 4회
   → 언제부터 정현규로 관심이 옮겨갔는가?
   → 코드: message_log[(message_log['name']=='성해은')].groupby('day')['message'].value_counts()

2. 회차별 감정 변화 타임라인
   → 에피소드 로그와 대화 데이터를 조인
   → 주요 이벤트(X룸 공개, 메기 투입) 전후 감정 변화
   → 코드: pd.merge(episode_log, haeeun_gyumin, how='cross')로 시계열 분석

3. 정규민의 관심 분산 패턴
   → 정규민이 이나연(9회), 김지수(5회)에게 보낸 문자 내용 분석
   → 왜 성해은만 제외했는가?
   → 코드: message_log[message_log['name']=='정규민']['message_txt'].tolist()

4. MBTI J-P 조합과 문자 비대칭도 상관관계
   → J 성향(규민, 현규)의 문자 발송 패턴 비교
   → P 성향(해은, 나연, 희두)의 문자 발송 패턴 비교
   → 코드: char_profile.merge(문자통계, on='name')

5. 정현규 투입(15회) 이후 성해은의 변화
   → 15회 이전 vs 이후 대화 감정 변화
   → "정규민" 언급 빈도 감소 여부
   → 코드: haeeun_gyumin['text'].str.contains('규민').sum() (회차별 분할)
"""

print(analysis_ideas)

print("\n주인님, 위 코드를 복사해서 직접 실행해보세요!")
print("궁금한 점이나 에러 나면 언제든 물어보세요!")
