#!/usr/bin/env python
"""
Dynamic Test Query Generator for RAG System Evaluation.

Generates diverse test queries using LLM for each persona type,
ensuring different queries on every run for comprehensive testing.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Persona definitions with query generation guidance
PERSONAS = {
    "student": {
        "label": "학생",
        "description": "학부생 또는 대학원생",
        "topics": [
            "휴학/복학", "졸업/학위", "장학금/등록금", "전과/편입", 
            "학사경고/성적", "수강신청/학점", "기숙사/생활관", "동아리/학생회"
        ],
        "styles": ["직접적 질문", "간접적 의도", "감정 표현", "불만 표현"],
    },
    "faculty": {
        "label": "교원",
        "description": "교수, 강사, 연구원",
        "topics": [
            "연구년/안식년", "승진/재임용", "강의/책임시수", "연구윤리",
            "해외파견/학회", "휴직/병가", "퇴직/명예퇴직", "겸직/겸임"
        ],
        "styles": ["직접적 질문", "간접적 의도", "업무 관련", "제도 문의"],
    },
    "staff": {
        "label": "직원",
        "description": "일반 행정직원",
        "topics": [
            "휴가/연가", "퇴직/퇴직금", "육아휴직/육아", "승진/전보",
            "복무/근무", "인사/평가", "수당/급여"
        ],
        "styles": ["직접적 질문", "간접적 의도", "행정 절차"],
    },
    "common": {
        "label": "공통",
        "description": "모든 대학 구성원",
        "topics": [
            "성희롱/성폭력 신고", "인권센터/고충처리", "연구윤리 위반",
            "주차/시설", "도서관/학술정보", "장애학생 지원", "갑질/괴롭힘"
        ],
        "styles": ["직접적 질문", "신고/문의", "정보 요청"],
    },
}

# Query generation prompt template
QUERY_GENERATION_PROMPT = """당신은 대학 구성원의 다양한 질문을 생성하는 전문가입니다.

## 역할
{persona_label} ({persona_desc}) 관점에서 대학 규정에 대한 질문을 생성하세요.

## 생성 조건
1. **주제**: {topic}
2. **스타일**: {style}
3. **다양성 키워드**: {diversity_seed}
4. **생성 개수**: {count}개

## 스타일 가이드
- 직접적 질문: "~하려면 어떻게 해야 하나요?"
- 간접적 의도: "~하기 싫어", "~하고 싶어"
- 감정 표현: "너무 힘들어", "어떡하지"
- 불만 표현: "왜 이렇게 복잡해", "불공평해"

## 출력 형식 (JSON 배열)
[
  {{"query": "질문 내용", "category": "카테고리", "intent_hint": "의도 힌트"}},
  ...
]

중요: JSON 배열만 출력하세요. 설명이나 다른 텍스트는 포함하지 마세요."""


def setup_llm():
    """Initialize LLM client."""
    from src.rag.infrastructure.llm_adapter import LLMClientAdapter
    
    try:
        return LLMClientAdapter(provider="ollama")
    except Exception as e:
        print(f"Warning: LLM init failed ({e}). Using fallback templates.")
        return None


def load_existing_queries(dataset_path: Optional[str] = None) -> set:
    """Load existing queries from evaluation dataset to avoid duplicates."""
    path = Path(dataset_path or "data/config/evaluation_dataset.json")
    if not path.exists():
        return set()
    
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {tc["query"].lower().strip() for tc in data.get("test_cases", [])}
    except Exception:
        return set()


def generate_diversity_seed() -> str:
    """Generate a random seed phrase for query diversity."""
    adjectives = ["긴급한", "복잡한", "특수한", "일반적인", "예외적인", "임시", "정규"]
    situations = ["상황", "경우", "조건", "사유", "사례"]
    emotions = ["걱정되는", "궁금한", "답답한", "급한", "중요한"]
    
    return f"{random.choice(adjectives)} {random.choice(situations)}, {random.choice(emotions)} 마음"


def generate_queries_with_llm(
    llm_client,
    persona: str,
    count: int = 5,
    existing_queries: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Generate queries using LLM for a specific persona."""
    if persona not in PERSONAS:
        raise ValueError(f"Unknown persona: {persona}")
    
    persona_info = PERSONAS[persona]
    topic = random.choice(persona_info["topics"])
    style = random.choice(persona_info["styles"])
    diversity_seed = generate_diversity_seed()
    
    prompt = QUERY_GENERATION_PROMPT.format(
        persona_label=persona_info["label"],
        persona_desc=persona_info["description"],
        topic=topic,
        style=style,
        diversity_seed=diversity_seed,
        count=count,
    )
    
    try:
        response = llm_client.generate(
            system_prompt="대학 규정 질문 생성기. JSON 배열만 출력.",
            user_message=prompt,
            temperature=0.8,  # Higher temperature for diversity
        )
        
        # Extract JSON from response
        import re
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            print(f"  Warning: Could not parse JSON from LLM response")
            return []
        
        queries = json.loads(json_match.group())
        
        # Filter out existing queries
        if existing_queries:
            queries = [
                q for q in queries 
                if q.get("query", "").lower().strip() not in existing_queries
            ]
        
        # Add metadata
        for q in queries:
            q["persona"] = persona
            q["generated_at"] = datetime.now().isoformat()
            q["topic"] = topic
            q["style"] = style
        
        return queries
        
    except Exception as e:
        print(f"  Error generating queries: {e}")
        return []


def generate_fallback_queries(persona: str, count: int = 5) -> List[Dict[str, Any]]:
    """Generate queries using templates when LLM is unavailable."""
    if persona not in PERSONAS:
        return []
    
    persona_info = PERSONAS[persona]
    templates = {
        "student": [
            "{topic} 어떻게 해야 해?",
            "{topic} 관련 규정 알려줘",
            "나 {topic} 하고 싶은데",
            "{topic} 때문에 너무 힘들어",
        ],
        "faculty": [
            "{topic} 신청하려면?",
            "{topic} 자격 요건이 뭐야?",
            "{topic} 절차 알려줘",
            "{topic} 하고 싶어",
        ],
        "staff": [
            "{topic} 규정 알려줘",
            "{topic} 어떻게 신청해?",
            "{topic} 쓰고 싶어",
        ],
        "common": [
            "{topic} 어떻게 해?",
            "{topic} 어디에 문의해야 해?",
            "{topic} 신고하고 싶어",
        ],
    }
    
    queries = []
    persona_templates = templates.get(persona, templates["common"])
    
    for _ in range(count):
        topic = random.choice(persona_info["topics"])
        template = random.choice(persona_templates)
        query_text = template.format(topic=topic)
        
        queries.append({
            "query": query_text,
            "persona": persona,
            "category": topic,
            "generated_at": datetime.now().isoformat(),
            "method": "fallback_template",
        })
    
    return queries


def save_queries(queries: List[Dict[str, Any]], output_path: Optional[str] = None):
    """Save generated queries to JSON file."""
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"data/output/generated_queries_{timestamp}.json"
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        "version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "total_queries": len(queries),
        "queries": queries,
    }
    
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="동적 테스트 쿼리 생성기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 학생 페르소나로 5개 쿼리 생성
  uv run python scripts/generate_test_queries.py --persona student --count 5
  
  # 모든 페르소나로 각 3개씩 생성
  uv run python scripts/generate_test_queries.py --all --count 3
  
  # 결과 파일 지정
  uv run python scripts/generate_test_queries.py --all -o data/output/my_queries.json
        """,
    )
    parser.add_argument(
        "--persona", "-p",
        choices=list(PERSONAS.keys()),
        help="쿼리를 생성할 페르소나 (student, faculty, staff, common)",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="모든 페르소나에 대해 쿼리 생성",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=5,
        help="페르소나당 생성할 쿼리 수 (기본값: 5)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="출력 파일 경로",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="기존 쿼리와의 중복 검사 비활성화",
    )
    
    args = parser.parse_args()
    
    if not args.persona and not args.all:
        parser.print_help()
        print("\n오류: --persona 또는 --all 옵션을 지정하세요.")
        sys.exit(1)
    
    # Setup
    print("🚀 동적 테스트 쿼리 생성기 시작")
    llm_client = setup_llm()
    existing_queries = set() if args.no_dedup else load_existing_queries()
    
    if existing_queries:
        print(f"📋 기존 쿼리 {len(existing_queries)}개 로드 (중복 방지)")
    
    # Generate queries
    all_queries = []
    personas_to_process = list(PERSONAS.keys()) if args.all else [args.persona]
    
    for persona in personas_to_process:
        print(f"\n👤 {PERSONAS[persona]['label']} 페르소나 쿼리 생성 중...")
        
        queries = []
        if llm_client:
            queries = generate_queries_with_llm(
                llm_client, persona, args.count, existing_queries
            )
        
        # Fallback to templates if LLM failed or returned empty
        if not queries:
            print("   ⚠️ LLM 생성 실패, 템플릿 사용")
            queries = generate_fallback_queries(persona, args.count)
        
        print(f"   ✅ {len(queries)}개 쿼리 생성 완료")
        all_queries.extend(queries)
        
        # Update existing queries set to prevent duplicates across personas
        existing_queries.update(q["query"].lower().strip() for q in queries)
    
    # Save results
    if all_queries:
        output_path = save_queries(all_queries, args.output)
        print(f"\n💾 결과 저장: {output_path}")
        print(f"📊 총 {len(all_queries)}개 쿼리 생성 완료")
        
        # Print sample
        print("\n📝 생성된 쿼리 샘플:")
        for q in all_queries[:5]:
            print(f"   - [{q.get('persona', 'unknown')}] {q['query']}")
        if len(all_queries) > 5:
            print(f"   ... 외 {len(all_queries) - 5}개")
    else:
        print("\n⚠️ 생성된 쿼리가 없습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
