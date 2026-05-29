"""Unified entrypoint with Typer CLI.

Usage:
    python main.py serve              # start FastAPI server
    python main.py ingest <type> <url>  # ingest a source
    python main.py search "query"     # search knowledge base
    python main.py chat               # interactive REPL
    python main.py review weekly      # run a review immediately
"""
from __future__ import annotations

import asyncio

import typer
from rich import print as rprint
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from agents import ChatAgent, get_agent
from models.database import init_db
from models.schemas import SourceType
from utils.logger import logger

app = typer.Typer(help="Personal Knowledge Management Agent CLI", no_args_is_help=True)
console = Console()


@app.command("serve")
def serve() -> None:
    """Start the FastAPI server (with web panel)."""
    from api.server import run

    run()


@app.command("ingest")
def cmd_ingest(
    source_type: str = typer.Argument(..., help="web|pdf|youtube|wechat|notion|rss|email"),
    target: str = typer.Argument(..., help="URL or local file path"),
) -> None:
    """Ingest a single source into the knowledge base."""

    async def _run() -> None:
        await init_db()
        result = await get_agent().ingest(SourceType(source_type), target)
        rprint(f"[green]✓[/green] Ingested: [bold]{result.title}[/bold]")
        rprint(f"  📂 Obsidian: {result.obsidian_path}")
        rprint(f"  🏷️  Tags: {', '.join(result.tags) or '-'}")
        rprint(f"  🔗 Links: {', '.join(result.links) or '-'}")
        rprint(f"  🧩 Chunks indexed: {result.chunks_indexed}")
        console.print(Markdown(result.summary or ""))

    asyncio.run(_run())


@app.command("search")
def cmd_search(
    query: str = typer.Argument(..., help="Natural language query"),
    k: int = typer.Option(5, "--k", "-k", help="Top-k results"),
) -> None:
    """Search the knowledge base (vector + keyword)."""
    from tools.search import hybrid_search

    async def _run() -> None:
        await init_db()
        hits = await hybrid_search(query, k=k)
        table = Table(title=f"Search results · '{query}'")
        table.add_column("#", justify="right")
        table.add_column("Title", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Source", overflow="fold")
        for i, h in enumerate(hits, 1):
            table.add_row(str(i), h.title, f"{h.score:.3f}", h.source)
        console.print(table)
        for h in hits:
            console.print(f"\n[bold]{h.title}[/bold]\n{h.snippet}\n")

    asyncio.run(_run())


@app.command("chat")
def cmd_chat() -> None:
    """Interactive REPL chat with the knowledge base."""

    async def _run() -> None:
        await init_db()
        agent = ChatAgent()
        rprint("[cyan]💬 PKM Chat (type 'exit' / 'quit' to leave, '/reset' to clear)[/cyan]")
        while True:
            try:
                msg = console.input("[bold green]you›[/bold green] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if msg.lower() in {"exit", "quit"}:
                break
            if msg == "/reset":
                agent.reset()
                rprint("[yellow]conversation reset[/yellow]")
                continue
            if not msg:
                continue
            resp = await agent.ask(msg)
            console.print(Markdown(resp.answer))

    asyncio.run(_run())


@app.command("review")
def cmd_review(period: str = typer.Argument("weekly", help="weekly|monthly")) -> None:
    """Generate a review report immediately."""

    async def _run() -> None:
        await init_db()
        resp = await get_agent().review(period)
        rprint(f"[green]✓[/green] Review saved to {resp.obsidian_path}")
        console.print(Markdown(resp.summary))

    asyncio.run(_run())


@app.command("init-db")
def cmd_init_db() -> None:
    """Initialise the SQLite database."""
    asyncio.run(init_db())
    rprint("[green]✓[/green] database ready")


@app.command("rss-add")
def cmd_rss_add(url: str = typer.Argument(...)) -> None:
    """Subscribe to an RSS feed."""
    from scrapers.rss_scraper import RSSScraper

    RSSScraper.add_subscription(url)
    rprint(f"[green]✓[/green] subscribed: {url}")


if __name__ == "__main__":
    try:
        app()
    except Exception as exc:  # pragma: no cover
        logger.exception(f"CLI crashed: {exc}")
        raise
