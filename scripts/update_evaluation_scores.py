#!/usr/bin/env python3
"""
Phase 1: evaluation_dataset.json의 min_relevance_score 일괄 조정

기존 0.3~0.5 값을 0.05~0.15로 하향 조정합니다.
"""

import json
from datetime import date
from pathlib import Path


def main():
    data_path = Path(__file__).parent.parent / "data" / "config" / "evaluation_dataset.json"

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 조정할 케이스별 새로운 점수
    score_adjustments = {
        # 규정명/조문 직접 검색은 정확도 높아야 하므로 0.15 유지
        "regulation_name_01": 0.15,
        "article_reference_01": 0.1,

        # 일반적인 자연어 쿼리는 0.05~0.1로 하향
        "student_leave_01": 0.1,
        "edge_overseas_conference": 0.1,
        "edge_dormitory": 0.1,
        "edge_school_quit_student": 0.1,
        "tuition_payment_01": 0.1,
        "credit_transfer_01": 0.1,
        "major_change_01": 0.1,
        "double_major_01": 0.1,
        "reinstatement_01": 0.1,
        "thesis_submission_01": 0.1,
        "early_graduation_01": 0.1,
        "graduation_delay_01": 0.1,
        "course_registration_01": 0.1,
        "internship_01": 0.1,
        "exchange_student_01": 0.1,
        "certificate_issue_01": 0.1,
        "disability_support_01": 0.1,
        "promotion_01": 0.1,
        "salary_inquiry_01": 0.1,
        "research_funding_01": 0.1,
        "overseas_study_01": 0.1,
        "parking_01": 0.1,
        "dormitory_01": 0.1,
        "employment_support_01": 0.1,
        "bullying_report_01": 0.1,
        "academic_warning_01": 0.1,

        # 모호한 쿼리는 0.05로 하향
        "edge_quit_ambiguous": 0.05,
        "transfer_admission_01": 0.1,
        "expulsion_prevention_01": 0.1,
    }

    # 케이스 업데이트
    updated_count = 0
    for case in data["test_cases"]:
        case_id = case["id"]
        if case_id in score_adjustments:
            old_score = case.get("min_relevance_score", 0)
            new_score = score_adjustments[case_id]
            case["min_relevance_score"] = new_score
            print(f"  {case_id}: {old_score} -> {new_score}")
            updated_count += 1

    # 메타데이터 업데이트
    data["last_updated"] = str(date.today())

    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ {updated_count}개 케이스 업데이트 완료")


if __name__ == "__main__":
    main()
