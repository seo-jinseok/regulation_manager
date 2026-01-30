"""
Flip-the-RAG Generator

규정 문서에서 답변을 추출하고, LLM을 사용하여 질문을 생성하는 방식으로
질문-정답 쌍을 자동으로 생성합니다.
"""

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FlipTheRAGGenerator:
    """
    Flip-the-RAG 방식으로 질문-정답 쌍 생성

    Workflow:
    1. 규정 문서에서 의미 있는 청크(답변 후보) 추출
    2. LLM으로 청크에 대한 질문 생성
    3. 질문-정답 쌍 저장 및 중복 제거
    """

    # 질문 생성을 위한 카테고리와 난이도
    CATEGORIES = [
        "졸업",
        "휴학",
        "복학",
        "장학금",
        "등록",
        "성적",
        "교과과정",
        "교환학생",
        "규정해석",
        "신청절차",
        "기간",
        "서류",
        "자격",
    ]

    DIFFICULTY_LEVELS = ["초급", "중급", "고급"]

    QUERY_TYPES = [
        "정확한 쿼리",
        "구어체 쿼리",
        "모호한 쿼리",
        "오타 포함 쿼리",
        "영문 혼용 쿼리",
        "복합 질문",
        "문맥 의존 질문",
    ]

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 500,
    ):
        """
        Flip-the-RAG Generator 초기화

        Args:
            model_name: 사용할 LLM 모델명
            temperature: 생성 온도 (0.0 ~ 1.0)
            max_tokens: 최대 토큰 수
        """
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.pairs: list[dict[str, Any]] = []

    def extract_answer_candidates(
        self,
        regulation_text: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> list[str]:
        """
        규정 텍스트에서 답변 후보 추출

        Args:
            regulation_text: 규정 전체 텍스트
            chunk_size: 청크 크기
            overlap: 오버랩 크기

        Returns:
            답변 후보 청크 리스트
        """
        chunks = []
        start = 0
        text_length = len(regulation_text)

        while start < text_length:
            end = start + chunk_size
            chunk = regulation_text[start:end].strip()

            # 의미 있는 청크만 저장 (최소 100자)
            if len(chunk) >= 100:
                # 문장 경계에서 자르기
                sentences = chunk.split(". ")
                if len(sentences) > 1 and end < text_length:
                    # 마지막 불완전한 문장 제외
                    chunk = ". ".join(sentences[:-1]) + "."

                chunks.append(chunk)

            start = end - overlap

        return chunks

    def generate_questions_for_answer(
        self,
        answer_text: str,
        source: str,
        num_questions: int = 3,
    ) -> list[dict[str, Any]]:
        """
        답변 텍스트에 대한 질문 생성

        Args:
            answer_text: 답변 텍스트
            source: 출처 (규정명, 조항 등)
            num_questions: 생성할 질문 수

        Returns:
            질문-정답 쌍 리스트
        """
        prompt = f"""
다음 규정 내용에 대해 자연스러운 한국어 질문을 {num_questions}개 생성하세요.

규정 내용:
{answer_text}

요구사항:
1. 질문은 실제 학생이 묻을 법한 자연스러운 한국어여야 합니다
2. 다양한 난이도를 포함하세요 (초급: 단순 사실 확인, 중급: 이해 필요, 고급: 종합적 사고)
3. 질문은 구체적이어야 합니다
4. 질문마다 적절한 카테고리를 지정하세요: {", ".join(self.CATEGORIES)}
5. 질문 유형을 다양화하세요: {", ".join(self.QUERY_TYPES[:3])}

JSON 형식으로 응답하세요:
{{
    "questions": [
        {{
            "query": "질문 내용",
            "category": "카테고리",
            "difficulty": "난이도",
            "query_type": "질문 유형",
            "keywords": ["키워드1", "키워드2"]
        }}
    ]
}}
"""

        try:
            response = self.llm.invoke(prompt)
            result = json.loads(response.content)

            pairs = []
            for q_data in result.get("questions", []):
                pair = {
                    "id": f"gt_{uuid.uuid4().hex[:8]}",
                    "query": q_data["query"],
                    "answer": answer_text,
                    "context": [source],
                    "category": q_data.get("category", "규정해석"),
                    "difficulty": q_data.get("difficulty", "중급"),
                    "query_type": q_data.get("query_type", "정확한 쿼리"),
                    "metadata": {
                        "source": source,
                        "keywords": q_data.get("keywords", []),
                        "generated_by": "flip-the-rag",
                    },
                }
                pairs.append(pair)

            return pairs

        except Exception as e:
            logger.error(f"질문 생성 실패: {e}")
            return []

    def generate_from_regulation_file(
        self,
        regulation_path: Path,
        target_pairs: int = 100,
    ) -> list[dict[str, Any]]:
        """
        규정 파일에서 질문-정답 쌍 생성

        Args:
            regulation_path: 규정 파일 경로
            target_pairs: 목표 질문-정답 쌍 수

        Returns:
            생성된 질문-정답 쌍 리스트
        """
        # 규정 파일 읽기
        try:
            with open(regulation_path, "r", encoding="utf-8") as f:
                regulation_text = f.read()
        except Exception as e:
            logger.error(f"규정 파일 읽기 실패 ({regulation_path}): {e}")
            return []

        # 답변 후보 추출
        chunks = self.extract_answer_candidates(regulation_text)
        source = regulation_path.stem

        # 질문 생성
        all_pairs = []
        questions_per_chunk = max(1, target_pairs // len(chunks))

        for chunk in tqdm(chunks, desc=f"Generating from {source}"):
            pairs = self.generate_questions_for_answer(
                chunk,
                source=source,
                num_questions=questions_per_chunk,
            )
            all_pairs.extend(pairs)

            if len(all_pairs) >= target_pairs:
                break

        return all_pairs[:target_pairs]

    def remove_duplicates(self, similarity_threshold: float = 0.8) -> None:
        """
        중복 질문 제거 (간단한 문자열 기반)

        Args:
            similarity_threshold: 유사도 임계값
        """
        unique_pairs = []
        seen_queries = set()

        for pair in self.pairs:
            query = pair["query"].lower().strip()

            # 정규화 (공백, 특수문자 제거)
            normalized = "".join(c for c in query if c.isalnum() or c.isspace())

            # 단순 중복 체크
            is_duplicate = False
            for seen in seen_queries:
                if normalized == seen:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_pairs.append(pair)
                seen_queries.add(normalized)

        removed = len(self.pairs) - len(unique_pairs)
        logger.info(f"중복 제거: {removed}개 제거됨")

        self.pairs = unique_pairs

    def generate(
        self,
        regulation_dir: Path,
        target_pairs: int = 300,
    ) -> list[dict[str, Any]]:
        """
        규정 디렉토리에서 전체 질문-정답 쌍 생성

        Args:
            regulation_dir: 규정 파일이 있는 디렉토리
            target_pairs: 목표 총 질문-정답 쌍 수

        Returns:
            생성된 질문-정답 쌍 리스트
        """
        # 규정 파일 찾기
        regulation_files = list(regulation_dir.glob("*.txt")) + list(
            regulation_dir.glob("*.md")
        )

        if not regulation_files:
            logger.warning(f"규정 파일을 찾을 수 없음: {regulation_dir}")
            return []

        logger.info(f"발견된 규정 파일: {len(regulation_files)}개")

        # 파일별 할당
        pairs_per_file = target_pairs // len(regulation_files)

        # 각 파일에서 질문-정답 쌍 생성
        for reg_file in regulation_files:
            pairs = self.generate_from_regulation_file(
                reg_file,
                target_pairs=pairs_per_file,
            )
            self.pairs.extend(pairs)

        # 중복 제거
        self.remove_duplicates()

        logger.info(f"총 생성된 질문-정답 쌍: {len(self.pairs)}개")

        return self.pairs

    def save(
        self,
        output_path: Path,
        dataset_id: str = "rag_gt_v1.0",
    ) -> None:
        """
        질문-정답 쌍을 JSONL 형식으로 저장

        Args:
            output_path: 출력 파일 경로
            dataset_id: 데이터셋 ID
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = {
            "dataset_id": dataset_id,
            "total_pairs": len(self.pairs),
            "created_at": Path(__file__).stat().st_mtime,
            "pairs": self.pairs,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        logger.info(f"저장 완료: {output_path}")


def main():
    """테스트 메인 함수"""
    generator = FlipTheRAGGenerator()

    # 규정 디렉토리 경로
    regulation_dir = Path(
        "/Users/truestone/Dropbox/repo/University/regulation_manager/data/processed/regulations"
    )

    if regulation_dir.exists():
        generator.generate(regulation_dir, target_pairs=300)
        generator.save(Path("data/ground_truth/train/flip_the_rag.jsonl"))
    else:
        logger.error(f"규정 디렉토리를 찾을 수 없음: {regulation_dir}")


if __name__ == "__main__":
    main()
