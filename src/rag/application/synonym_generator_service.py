"""
Synonym Generator Service for Regulation RAG System.

Uses LLM to generate synonym candidates for terms in university regulations.
Provides CRUD operations for managing synonyms in synonyms.json.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..domain.repositories import ILLMClient


class SynonymGeneratorService:
    """LLM 기반 동의어 생성 및 관리 서비스"""

    DEFAULT_SYNONYMS_PATH = Path("data/config/synonyms.json")

    # LLM 프롬프트 (한국어)
    SYSTEM_PROMPT = """당신은 대학교 규정 문서에서 사용되는 용어의 동의어를 찾아주는 전문가입니다.
사용자가 제시하는 용어에 대해 대학 규정 맥락에서 사용될 수 있는 동의어, 유사어, 관련 표현을 제안해주세요.

규칙:
1. 대학 규정, 학사 행정, 인사 관리 맥락에 맞는 용어만 제안
2. 공식적인 표현과 일상적인 표현을 모두 포함
3. 최대 10개까지만 제안
4. 각 동의어는 쉼표로 구분하여 한 줄로 출력
5. 번호나 설명 없이 동의어만 출력

예시 입력: 휴학
예시 출력: 휴학원, 휴학 신청, 학업 중단, 학교 쉬다, 일반휴학, 군휴학"""

    USER_PROMPT_TEMPLATE = """다음 용어의 동의어를 대학 규정 맥락에서 찾아주세요.

용어: {term}
맥락: {context}

동의어 목록 (쉼표로 구분):"""

    def __init__(
        self,
        llm_client: Optional[ILLMClient] = None,
        synonyms_path: Optional[Path] = None,
    ):
        """
        Initialize SynonymGeneratorService.

        Args:
            llm_client: LLM 클라이언트 (None이면 동의어 생성 불가, CRUD만 가능)
            synonyms_path: synonyms.json 파일 경로
        """
        self.llm_client = llm_client
        self.synonyms_path = synonyms_path or self.DEFAULT_SYNONYMS_PATH

    def generate_synonyms(
        self,
        term: str,
        context: str = "대학 규정",
        exclude_existing: bool = True,
    ) -> list[str]:
        """
        LLM을 사용하여 주어진 용어의 동의어 후보를 생성합니다.

        Args:
            term: 동의어를 생성할 기준 용어
            context: 용어가 사용되는 맥락 (기본: 대학 규정)
            exclude_existing: 이미 등록된 동의어 제외 여부

        Returns:
            동의어 후보 리스트 (최대 10개)

        Raises:
            RuntimeError: LLM 클라이언트가 설정되지 않은 경우
        """
        if not self.llm_client:
            raise RuntimeError("LLM 클라이언트가 설정되지 않았습니다.")

        user_message = self.USER_PROMPT_TEMPLATE.format(term=term, context=context)

        response = self.llm_client.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_message=user_message,
            temperature=0.3,  # Slightly creative but mostly deterministic
        )

        # Parse response: split by comma and clean up
        candidates = []
        for item in response.split(","):
            cleaned = item.strip()
            if cleaned and cleaned != term:
                candidates.append(cleaned)

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)

        # Optionally exclude existing synonyms
        if exclude_existing:
            existing = self.get_synonyms(term)
            unique_candidates = [c for c in unique_candidates if c not in existing]

        return unique_candidates[:10]

    def load_synonyms(self) -> dict:
        """
        synonyms.json 파일을 로드합니다.

        Returns:
            동의어 사전 전체 데이터 (version, description, terms 포함)
        """
        if not self.synonyms_path.exists():
            return {
                "version": "1.0.0",
                "description": "대학 규정 검색용 동의어 사전",
                "last_updated": datetime.now().strftime("%Y-%m-%d"),
                "terms": {},
            }

        with open(self.synonyms_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_synonyms(self, data: dict) -> None:
        """
        동의어 데이터를 synonyms.json에 저장합니다.
        version과 last_updated를 자동으로 갱신합니다.

        Args:
            data: 저장할 동의어 사전 데이터
        """
        # Update metadata
        data["last_updated"] = datetime.now().strftime("%Y-%m-%d")

        # Ensure parent directory exists
        self.synonyms_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.synonyms_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def get_synonyms(self, term: str) -> list[str]:
        """
        특정 용어의 동의어 목록을 반환합니다.

        Args:
            term: 조회할 용어

        Returns:
            동의어 리스트 (없으면 빈 리스트)
        """
        data = self.load_synonyms()
        return data.get("terms", {}).get(term, [])

    def list_terms(self) -> list[str]:
        """
        등록된 모든 용어 목록을 반환합니다.

        Returns:
            용어 리스트
        """
        data = self.load_synonyms()
        return list(data.get("terms", {}).keys())

    def add_synonym(self, term: str, synonym: str) -> bool:
        """
        특정 용어에 동의어를 추가합니다.

        Args:
            term: 기준 용어
            synonym: 추가할 동의어

        Returns:
            True if added, False if already exists
        """
        data = self.load_synonyms()
        terms = data.setdefault("terms", {})

        if term not in terms:
            terms[term] = []

        if synonym in terms[term]:
            return False

        terms[term].append(synonym)
        self.save_synonyms(data)
        return True

    def add_synonyms(self, term: str, synonyms: list[str]) -> int:
        """
        특정 용어에 여러 동의어를 한번에 추가합니다.

        Args:
            term: 기준 용어
            synonyms: 추가할 동의어 리스트

        Returns:
            실제로 추가된 동의어 개수
        """
        data = self.load_synonyms()
        terms = data.setdefault("terms", {})

        if term not in terms:
            terms[term] = []

        added_count = 0
        for synonym in synonyms:
            if synonym not in terms[term]:
                terms[term].append(synonym)
                added_count += 1

        if added_count > 0:
            self.save_synonyms(data)

        return added_count

    def remove_synonym(self, term: str, synonym: str) -> bool:
        """
        특정 용어에서 동의어를 제거합니다.

        Args:
            term: 기준 용어
            synonym: 제거할 동의어

        Returns:
            True if removed, False if not found
        """
        data = self.load_synonyms()
        terms = data.get("terms", {})

        if term not in terms:
            return False

        if synonym not in terms[term]:
            return False

        terms[term].remove(synonym)

        # Remove term entry if no synonyms left
        if not terms[term]:
            del terms[term]

        self.save_synonyms(data)
        return True

    def remove_term(self, term: str) -> bool:
        """
        용어와 모든 동의어를 제거합니다.

        Args:
            term: 제거할 용어

        Returns:
            True if removed, False if not found
        """
        data = self.load_synonyms()
        terms = data.get("terms", {})

        if term not in terms:
            return False

        del terms[term]
        self.save_synonyms(data)
        return True
