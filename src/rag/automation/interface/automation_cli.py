"""
Automation CLI for RAG Testing.

Click-based command-line interface for automated RAG testing,
multi-turn simulation, and report generation.

Clean Architecture: Interface layer handles user interaction.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0")
def regulation():
    """
    RAG Regulation Testing Automation CLI.

    Automated testing framework for RAG-based regulation Q&A system.
    """
    pass


@regulation.command()
@click.option(
    "--session-id",
    "-s",
    required=True,
    help="Unique session identifier for the test run.",
)
@click.option(
    "--tests-per-persona",
    "-n",
    default=3,
    type=int,
    help="Number of test cases to generate per persona (default: 3).",
)
@click.option(
    "--difficulty",
    "-d",
    type=click.Choice(["easy", "medium", "hard"]),
    help="Filter tests by difficulty level.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("./test_results"),
    help="Output directory for test results (default: ./test_results).",
)
@click.option(
    "--db-path",
    type=click.Path(path_type=Path),
    default=Path("data/chroma_db"),
    help="ChromaDB database path (default: data/chroma_db).",
)
@click.option(
    "--provider",
    type=str,
    default=None,
    help="LLM provider (default: from .env or config).",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="LLM model name (default: from .env or config).",
)
@click.option(
    "--base-url",
    type=str,
    default=None,
    help="LLM base URL for local servers (default: from .env or config).",
)
@click.option(
    "--no-rerank",
    is_flag=True,
    help="Disable BGE reranker for faster testing.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Generate tests without executing them.",
)
@click.option(
    "--parallel",
    "-p",
    is_flag=True,
    help="Enable parallel test execution for faster evaluation.",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=None,
    help="Number of parallel workers (default: CPU count).",
)
@click.option(
    "--rate-limit",
    "-r",
    type=float,
    default=5.0,
    help="Rate limit for LLM API calls in requests per second (default: 5.0).",
)
@click.option(
    "--html-report",
    is_flag=True,
    help="Generate interactive HTML report with visualizations.",
)
@click.option(
    "--vary-queries/--no-vary-queries",
    default=True,
    help="Use LLM to generate diverse queries (default: True). Disable for template-based queries.",
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducible query generation. Same seed produces identical test cases.",
)
def test(
    session_id: str,
    tests_per_persona: int,
    difficulty: Optional[str],
    output_dir: Path,
    db_path: Path,
    provider: Optional[str],
    model: Optional[str],
    base_url: Optional[str],
    no_rerank: bool,
    dry_run: bool,
    parallel: bool,
    workers: Optional[int],
    rate_limit: float,
    html_report: bool,
    vary_queries: bool,
    seed: Optional[int],
):
    """
    Run automated RAG testing session.

    Generates test cases, executes queries, evaluates quality,
    and produces comprehensive reports.
    """
    click.echo(f"🧪 Starting test session: {session_id}")
    click.echo(f"📊 Tests per persona: {tests_per_persona}")
    click.echo(f"📂 Database: {db_path}")

    try:
        # Import dependencies
        from ...application.search_usecase import SearchUseCase
        from ...config import get_config
        from ...infrastructure.chroma_store import ChromaVectorStore
        from ...infrastructure.llm_adapter import LLMClientAdapter
        from ..application.execute_test_usecase import ExecuteTestUseCase
        from ..application.generate_test_usecase import GenerateTestUseCase
        from ..infrastructure.json_session_repository import JSONSessionRepository

        # Initialize RAG components
        click.echo("🔧 Initializing RAG system...")

        # Get default LLM settings
        config = get_config()
        llm_provider = provider or config.llm_provider
        llm_model = model or config.llm_model
        llm_base_url = base_url or config.llm_base_url

        # Initialize vector store
        store = ChromaVectorStore(persist_directory=str(db_path))

        # Initialize LLM client
        llm_client = LLMClientAdapter(
            provider=llm_provider,
            model=llm_model,
            base_url=llm_base_url,
        )

        # Initialize SearchUseCase
        search_usecase = SearchUseCase(
            store=store,
            llm_client=llm_client,
            use_reranker=not no_rerank,
        )

        click.echo(f"✅ RAG system ready (LLM: {llm_provider}/{llm_model})")

        # Initialize test use cases
        session_repo = JSONSessionRepository(output_dir)
        generate_usecase = GenerateTestUseCase(
            session_repository=session_repo,
            llm_client=llm_client,  # Pass LLM client for diverse query generation
        )
        execute_usecase = ExecuteTestUseCase(search_usecase=search_usecase)

        # Step 1: Generate test cases
        click.echo("📝 Generating test cases...")

        # Filter by difficulty if specified
        if difficulty:
            from ..domain.value_objects import DifficultyDistribution

            if difficulty == "easy":
                difficulty_dist = DifficultyDistribution(
                    easy_ratio=1.0, medium_ratio=0.0, hard_ratio=0.0
                )
            elif difficulty == "medium":
                difficulty_dist = DifficultyDistribution(
                    easy_ratio=0.0, medium_ratio=1.0, hard_ratio=0.0
                )
            else:  # hard
                difficulty_dist = DifficultyDistribution(
                    easy_ratio=0.0, medium_ratio=0.0, hard_ratio=1.0
                )
        else:
            difficulty_dist = None

        # Generate test cases (also for dry-run to enable report generation)
        session = generate_usecase.execute(
            session_id=session_id,
            tests_per_persona=tests_per_persona,
            difficulty_distribution=difficulty_dist,
            vary_queries=vary_queries,  # Pass vary_queries flag
            seed=seed,  # Pass seed for reproducibility
            metadata={
                "difficulty_filter": difficulty,
                "vary_queries": vary_queries,
                "seed": seed,
            },
        )

        click.echo(f"✅ Generated {len(session.test_cases)} test cases")

        if not dry_run:
            session = generate_usecase.execute(
                session_id=session_id,
                tests_per_persona=tests_per_persona,
                difficulty_distribution=difficulty_dist,
                vary_queries=vary_queries,  # Pass vary_queries flag
                seed=seed,  # Pass seed for reproducibility
                metadata={
                    "difficulty_filter": difficulty,
                    "vary_queries": vary_queries,
                    "seed": seed,
                },
            )

            click.echo(f"✅ Generated {len(session.test_cases)} test cases")

            # Step 2: Execute tests with actual RAG system
            if parallel:
                click.echo(
                    f"🚀 Executing tests in parallel (workers: {workers or 'auto'}, rate limit: {rate_limit} req/s)..."
                )

                # Progress tracking callback
                completed_tests = [0]

                def progress_callback(completed: int, total: int, result) -> None:
                    """Update progress display."""
                    completed_tests[0] = completed
                    sources_count = len(result.sources) if result.sources else 0
                    status = (
                        f"✅ {sources_count} sources, conf={result.confidence:.2f}"
                        if sources_count > 0
                        else "⚠️  No sources"
                    )
                    click.echo(
                        f"   [{completed}/{total}] {result.query[:50]}... → {status}"
                    )

                # Execute tests in parallel
                test_results = execute_usecase.batch_execute_test_cases(
                    test_cases=session.test_cases,
                    max_workers=workers,
                    rate_limit_per_second=rate_limit,
                    progress_callback=progress_callback,
                )
            else:
                click.echo("🚀 Executing tests sequentially...")

                test_results = []
                for idx, test_case in enumerate(session.test_cases, 1):
                    click.echo(
                        f"   [{idx}/{len(session.test_cases)}] {test_case.query[:50]}..."
                    )

                    try:
                        result = execute_usecase.execute_test_case(test_case)
                        test_results.append(result)

                        # Simple pass/fail based on whether we got results
                        if result.sources and len(result.sources) > 0:
                            click.echo(
                                f"      ✅ {len(result.sources)} sources, confidence={result.confidence:.2f}"
                            )
                        else:
                            click.echo("      ⚠️  No sources found")

                    except Exception as e:
                        logger.error(f"Test case {idx} failed: {e}")
                        # Create error result
                        from ..domain.entities import TestResult

                        error_result = TestResult(
                            test_case_id=f"error_{idx}",
                            query=test_case.query,
                            answer="",
                            sources=[],
                            confidence=0.0,
                            execution_time_ms=0,
                            rag_pipeline_log={"error": str(e)},
                            error_message=str(e),
                            passed=False,
                        )
                        test_results.append(error_result)
                        click.echo(f"      ❌ Error: {e}")

            # Calculate summary statistics
            passed = sum(
                1
                for r in test_results
                if r.sources and len(r.sources) > 0 and not r.error_message
            )
            failed = len(test_results) - passed

            # Save session with results
            session.test_results = test_results
            session.completed_at = datetime.now()
            session_repo.save(session)

            click.echo("\n📊 Execution Summary:")
            click.echo(f"   Total: {len(test_results)}")
            click.echo(f"   Passed: {passed}")
            click.echo(f"   Failed: {failed}")

        else:
            click.echo("🔍 Dry run: Tests would be generated but not executed")
            test_results = []

        # Step 3: Generate report
        click.echo("📊 Generating report...")

        from ..infrastructure.test_report_generator import TestReportGenerator

        report_generator = TestReportGenerator(output_dir=output_dir)

        # Generate HTML or Markdown report
        if html_report and test_results:
            report_path = report_generator.generate_html_report(
                session_id=session_id,
                test_results=test_results,
                metadata={
                    "total_test_cases": session.total_test_cases,
                    "session_duration": (
                        f"{session.duration_seconds:.0f}s"
                        if session.duration_seconds
                        else "N/A"
                    ),
                    "difficulty_filter": difficulty,
                    "tests_per_persona": tests_per_persona,
                },
            )
        else:
            report_path = report_generator.generate_report(
                session_id=session_id,
                test_results=test_results,
                metadata={
                    "total_test_cases": session.total_test_cases,
                    "session_duration": (
                        f"{session.duration_seconds:.0f}s"
                        if session.duration_seconds
                        else "N/A"
                    ),
                    "difficulty_filter": difficulty,
                    "tests_per_persona": tests_per_persona,
                },
            )

        click.echo(f"✅ Report generated: {report_path}")

        # Open HTML report in browser if requested
        if html_report:
            click.echo(
                f"💡 Open the report in your browser: file://{report_path.absolute()}"
            )

        click.echo(f"✨ Test session complete: {session_id}")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        logger.exception("Test execution failed")
        sys.exit(1)


@regulation.command()
@click.option(
    "--session-id",
    "-s",
    required=True,
    help="Session ID to generate report for.",
)
@click.option(
    "--results-dir",
    "-r",
    type=click.Path(path_type=Path),
    default=Path("./test_results"),
    help="Directory containing test results (default: ./test_results).",
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path for the report (default: auto-generated).",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["markdown", "json", "html"]),
    default="markdown",
    help="Report format (default: markdown).",
)
def report(
    session_id: str,
    results_dir: Path,
    output_file: Optional[Path],
    format: str,
):
    """
    Generate test report from results.

    Creates comprehensive markdown, JSON, or HTML reports
    from test execution results.

    HTML format includes interactive Chart.js visualizations.
    """
    click.echo(f"📊 Generating {format} report for session: {session_id}")

    try:
        from ..infrastructure.json_session_repository import JSONSessionRepository
        from ..infrastructure.test_report_generator import TestReportGenerator

        # Load test results
        session_repo = JSONSessionRepository(results_dir)
        session = session_repo.load(session_id)

        if not session:
            click.echo(f"❌ Session not found: {session_id}", err=True)
            sys.exit(1)

        # Initialize report generator
        report_generator = TestReportGenerator(output_dir=results_dir)

        # For demonstration, we'll create empty results
        # In production, would load actual TestResult objects
        from ..domain.entities import TestResult

        test_results: list[TestResult] = []

        # Generate report based on format
        if format == "html":
            report_path = report_generator.generate_html_report(
                session_id=session_id,
                test_results=test_results,
                metadata={
                    "total_test_cases": session.total_test_cases,
                    "session_duration": (
                        f"{session.duration_seconds:.0f}s"
                        if session.duration_seconds
                        else "N/A"
                    ),
                },
            )
        else:
            report_path = report_generator.generate_report(
                session_id=session_id,
                test_results=test_results,
                metadata={
                    "total_test_cases": session.total_test_cases,
                    "session_duration": (
                        f"{session.duration_seconds:.0f}s"
                        if session.duration_seconds
                        else "N/A"
                    ),
                },
            )

        click.echo(f"✅ Report generated: {report_path}")

        # Display report path
        click.echo(f"📄 Report saved to: {report_path}")

    except Exception as e:
        click.echo(f"❌ Error generating report: {e}", err=True)
        logger.exception("Report generation failed")
        sys.exit(1)


@regulation.command()
@click.option(
    "--query",
    "-q",
    required=True,
    help="Initial query for multi-turn conversation.",
)
@click.option(
    "--persona",
    "-p",
    type=click.Choice(
        [
            "freshman",
            "junior",
            "graduate",
            "new_professor",
            "professor",
            "new_staff",
            "staff_manager",
            "parent",
            "distressed_student",
            "dissatisfied_member",
        ]
    ),
    default="junior",
    help="User persona for the conversation (default: junior).",
)
@click.option(
    "--min-turns",
    "-m",
    default=3,
    type=int,
    help="Minimum number of turns to generate (default: 3).",
)
@click.option(
    "--max-turns",
    "-M",
    default=5,
    type=int,
    help="Maximum number of turns to generate (default: 5).",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("./multi_turn_results"),
    help="Output directory for results (default: ./multi_turn_results).",
)
def simulate(
    query: str,
    persona: str,
    min_turns: int,
    max_turns: int,
    output_dir: Path,
):
    """
    Simulate multi-turn conversation.

    Generates context-aware follow-up questions and
    tests RAG system's context management capabilities.
    """
    click.echo("💬 Starting multi-turn simulation")
    click.echo(f"👤 Persona: {persona}")
    click.echo(f"📝 Initial query: {query}")

    try:
        from ..domain.entities import PersonaType
        from ..infrastructure.llm_persona_generator import PersonaGenerator

        # Get persona
        persona_gen = PersonaGenerator()
        persona_enum = PersonaType(persona)
        persona_obj = persona_gen.get_persona(persona_enum)

        click.echo(f"🎭 Using persona: {persona_obj.name}")

        # Note: MultiTurnSimulator requires ExecuteTestUseCase
        # This would need proper RAG integration
        click.echo("⚠️  Multi-turn simulation requires RAG system integration")
        click.echo("💡 This is a placeholder for the simulation workflow")

        # Placeholder for simulation
        click.echo(f"📊 Results would be saved to: {output_dir}")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        logger.exception("Multi-turn simulation failed")
        sys.exit(1)


@regulation.command()
@click.option(
    "--session-id",
    "-s",
    required=True,
    help="Session ID to analyze failures for.",
)
@click.option(
    "--apply",
    is_flag=True,
    help="Apply suggested improvements automatically.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview improvements without applying.",
)
def improve(
    session_id: str,
    apply: bool,
    dry_run: bool,
):
    """
    Analyze failures and apply improvements.

    Performs 5-Why root cause analysis on test failures
    and applies automated improvements to intents/synonyms.
    """
    click.echo(f"🔍 Analyzing failures for session: {session_id}")

    if apply and dry_run:
        click.echo("⚠️  Cannot use --apply and --dry-run together", err=True)
        sys.exit(1)

    try:
        from ..infrastructure.json_session_repository import JSONSessionRepository

        # Load session
        session_repo = JSONSessionRepository()
        session = session_repo.load(session_id)

        if not session:
            click.echo(f"❌ Session not found: {session_id}", err=True)
            sys.exit(1)

        # Placeholder for failure analysis
        click.echo("⚠️  Failure analysis requires test execution results")
        click.echo("💡 Run 'regulation test run' first to generate results")

        if dry_run:
            click.echo("🔍 Dry run: Would preview improvements")
        elif apply:
            click.echo("🔧 Applying improvements...")
            # Would call apply_usecase.apply_improvements()
            click.echo("⚠️  Apply requires actual test results")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        logger.exception("Failure analysis failed")
        sys.exit(1)


@regulation.command()
@click.option(
    "--results-dir",
    "-r",
    type=click.Path(path_type=Path),
    default=Path("./test_results"),
    help="Directory containing test results (default: ./test_results).",
)
def list_sessions(results_dir: Path):
    """List all test sessions."""
    click.echo(f"📂 Listing sessions in: {results_dir}")

    try:
        from ..infrastructure.json_session_repository import JSONSessionRepository

        session_repo = JSONSessionRepository(results_dir)
        sessions = session_repo.list_all()

        if not sessions:
            click.echo("📭 No sessions found")
            return

        click.echo(f"\n📊 Found {len(sessions)} session(s):\n")

        for session in sessions:
            status = "✅ Completed" if session.is_completed else "🔄 In Progress"
            duration = (
                f"{session.duration_seconds:.0f}s"
                if session.duration_seconds
                else "N/A"
            )

            click.echo(f"📌 {session.session_id}")
            click.echo(f"   Status: {status}")
            click.echo(f"   Tests: {session.total_test_cases}")
            click.echo(f"   Duration: {duration}")
            click.echo(
                f"   Started: {session.started_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            click.echo("")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        logger.exception("List sessions failed")
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    regulation()


if __name__ == "__main__":
    main()
