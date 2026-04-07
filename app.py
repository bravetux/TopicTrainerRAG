# =============================================================================
# Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
# Date   : 26 March 2026
# =============================================================================
"""TechTrainer AI — Streamlit UI."""
import json
import re
import uuid
import time
import logging
import threading
import streamlit as st
from src.tools.progress_db import init_db, read_progress
from src.agents.orchestrator import build_orchestrator
from src.agents.content_author_agent import content_author_agent
from src.config import PROGRESS_DB, BUILTIN_TOPICS
from src.tools.kb_manager import (
    load_all_topics,
    create_custom_topic,
    delete_topic,
    save_uploaded_file,
    list_topic_files,
    get_available_topic_ids,
    get_available_topics,
)
from src.tools.document_ingestion import index_technology
from src.tools.chat_history import save_exchange, load_history, clear_history as clear_chat_history
from src.tools.provider_manager import (
    get_effective_config,
    test_connection,
    load_settings,
    save_settings,
    write_env_values,
    PROVIDER_IDS,
    PROVIDER_LABELS,
)
from src.tools.embedding_manager import (
    EMBEDDING_PROVIDER_IDS,
    EMBEDDING_PROVIDER_LABELS,
    EMBEDDING_MODEL_DEFAULTS,
    get_embedding_config,
)

logger = logging.getLogger(__name__)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechTrainer AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state bootstrap ────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.connection_status = {"ok": None, "message": "Checking...", "latency_ms": None}

if st.session_state.get("connection_status", {}).get("ok") is None:
    st.session_state.connection_status = test_connection()

if "messages" not in st.session_state:
    st.session_state.messages = load_history()
if "orchestrator" not in st.session_state:
    init_db()
    st.session_state.orchestrator = build_orchestrator(st.session_state.session_id)

session_id = st.session_state.session_id
orchestrator = st.session_state.orchestrator


def _extract_json(text: str) -> dict | None:
    """Extract the first valid JSON object from a string, handling markdown fences."""
    if not text:
        return None
    # Fast path: the whole string is already valid JSON (direct tool returns)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    # Strip markdown code fences if present
    fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass
    # Greedy search for outermost { ... }
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


def _start_indexing(tid: str, reindex: bool = False) -> None:
    """Kick off a background indexing thread for a topic and store its state."""
    cancel_event = threading.Event()
    state = {
        "cancel": cancel_event,
        "reindex": reindex,
        "chunks": 0,
        "current_file": "",
        "file_index": 0,
        "total_files": 0,
        "done": False,
        "error": None,
    }

    def _on_file(name, idx, total):
        state["current_file"] = name
        state["file_index"] = idx
        state["total_files"] = total

    def _run():
        try:
            state["chunks"] = index_technology(
                tid,
                reindex=reindex,
                cancel_event=cancel_event,
                on_file_start=_on_file,
            )
        except Exception as exc:
            state["error"] = str(exc)
        finally:
            state["done"] = True

    t = threading.Thread(target=_run, daemon=True)
    state["thread"] = t
    st.session_state[f"idx_{tid}"] = state
    t.start()


def _render_index_progress(tid: str) -> None:
    """Render progress bar + Stop button while indexing; show result when done."""
    key = f"idx_{tid}"
    state = st.session_state.get(key)
    if state is None:
        return

    thread = state["thread"]

    if thread.is_alive():
        total = state["total_files"] or 1
        done_files = state["file_index"]
        pct = done_files / total
        label = f"Indexing {state['current_file'] or '…'}  ({done_files}/{total} files)"
        col_bar, col_stop = st.columns([4, 1])
        with col_bar:
            st.progress(pct, text=label)
        with col_stop:
            if st.button("⏹ Stop", key=f"stop_{tid}"):
                state["cancel"].set()
        time.sleep(0.4)
        st.rerun()
        return

    # Thread finished — show result then clear state
    del st.session_state[key]
    if state["cancel"].is_set():
        st.warning(f"Indexing stopped — {state['chunks']} chunks indexed before stop.")
    elif state["error"]:
        st.error(f"Indexing error: {state['error']}")
    elif state["chunks"] == 0:
        st.warning("No indexable documents found.")
    else:
        st.success(f"{'Re-indexed' if state['reindex'] else 'Indexed'}: {state['chunks']} chunks")


def _render_topic_card(topic: dict) -> None:
    """Render a single topic card with status, file list, upload zone, and index controls."""
    tid = topic["id"]
    status = topic["status"]
    is_builtin = topic.get("is_builtin", True)

    # Status header
    if status == "AVAILABLE":
        st.success(
            f"**{topic['display_name']}**{'  ' if is_builtin else '  `custom`'} — ✓ {topic['chunk_count']} chunks",
        )
    elif status == "PENDING":
        st.warning(f"**{topic['display_name']}**{'  ' if is_builtin else '  `custom`'} — ⏳ Files not indexed")
    else:
        st.error(f"**{topic['display_name']}**{'  ' if is_builtin else '  `custom`'} — ✗ No material")

    # File list
    files = list_topic_files(tid)
    if files:
        with st.expander(f"📂 {len(files)} file(s) in folder"):
            for fname in files:
                st.text(f"  📄 {fname}")

    if status == "DISABLED" and not files:
        st.caption("Upload documents or copy files to data/documents/" + tid + "/")

    # Upload widget
    auto_idx = st.checkbox("Auto-index on upload", value=True, key=f"auto_{tid}")
    uploaded = st.file_uploader(
        f"Upload for {topic['display_name']}",
        accept_multiple_files=True,
        key=f"upload_{tid}",
        type=["pdf", "docx", "pptx", "xlsx", "txt", "md"],
        label_visibility="collapsed",
    )

    if uploaded:
        # Deduplicate by comparing upload batch to last-processed batch to avoid
        # re-running on every Streamlit rerun while the file uploader holds files.
        batch_key = f"uploaded_batch_{tid}"
        current_batch = sorted(uf.name for uf in uploaded)
        if st.session_state.get(batch_key) != current_batch:
            st.session_state[batch_key] = current_batch
            saved_names = []
            for uf in uploaded:
                try:
                    save_uploaded_file(tid, uf.name, uf.getvalue())
                    saved_names.append(uf.name)
                except Exception as e:
                    st.warning(f"Skipped {uf.name}: {e}")

            if saved_names and auto_idx:
                _start_indexing(tid, reindex=False)
                st.rerun()
            elif saved_names:
                st.info(f"Saved {len(saved_names)} file(s). Click 'Index' to make searchable.")
                st.rerun()
    else:
        # Clear the batch key when the uploader is cleared
        st.session_state.pop(f"uploaded_batch_{tid}", None)

    # ── Indexing progress / stop / result ────────────────────────────────────
    _render_index_progress(tid)

    # Action buttons
    idx_key = f"idx_{tid}"
    b1, b2, b3 = st.columns([2, 2, 1])
    with b1:
        if status == "AVAILABLE" and idx_key not in st.session_state:
            if st.button("🔄 Re-index", key=f"reindex_{tid}"):
                _start_indexing(tid, reindex=True)
                st.rerun()
    with b2:
        if status in ("PENDING", "DISABLED") and topic["file_count"] > 0 and idx_key not in st.session_state:
            if st.button("⚡ Index", key=f"index_{tid}"):
                _start_indexing(tid, reindex=False)
                st.rerun()
    with b3:
        if not is_builtin:
            if st.button("🗑", key=f"del_{tid}", help=f"Delete {topic['display_name']}"):
                st.session_state[f"confirm_del_{tid}"] = True

    # Delete confirmation (custom topics only)
    if not is_builtin and st.session_state.get(f"confirm_del_{tid}"):
        st.warning(f"Delete '{topic['display_name']}'? Files on disk are kept.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, delete", key=f"yes_del_{tid}", type="primary"):
                try:
                    delete_topic(tid)
                    st.session_state.pop(f"confirm_del_{tid}", None)
                    st.success(f"Deleted: {topic['display_name']}")
                except ValueError as e:
                    st.error(str(e))
                st.rerun()
        with c2:
            if st.button("Cancel", key=f"no_del_{tid}"):
                st.session_state.pop(f"confirm_del_{tid}", None)
                st.rerun()


def _render_topic_grid(topics: list) -> None:
    """Render topics in a 2-column grid of cards."""
    for i in range(0, len(topics), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(topics):
                with col:
                    with st.container(border=True):
                        _render_topic_card(topics[i + j])


def call_agent(prompt: str) -> str:
    """Invoke orchestrator and return response string."""
    if st.session_state.get("agent_busy"):
        return "The assistant is still processing a previous request. Please wait a moment and try again."
    st.session_state["agent_busy"] = True
    try:
        result = orchestrator(
            prompt,
            invocation_state={"session_id": session_id},
        )
        return str(result)
    except Exception as exc:
        logger.error("Agent error: %s", exc)
        return f"Sorry, I encountered an error: {exc}"
    finally:
        st.session_state["agent_busy"] = False


def get_progress_data() -> dict:
    """Fetch progress data from SQLite."""
    try:
        raw = read_progress(PROGRESS_DB, session_id)
        return json.loads(raw)
    except Exception:
        return {"quiz_results": [], "technologies": {}}


# ── Sidebar — Progress Dashboard ───────────────────────────────────────────────
with st.sidebar:
    st.title("📊 My Progress")
    st.caption(f"Session: `{session_id[:8]}...`")
    st.divider()

    progress_data = get_progress_data()
    tech_stats = progress_data.get("technologies", {})

    if tech_stats:
        for tech, stats in sorted(tech_stats.items()):
            avg = stats.get("avg_score", 0)
            best = stats.get("best_score", 0)
            attempts = stats.get("attempts", 0)
            st.markdown(f"**{tech.replace('_', ' ').title()}**")
            st.progress(avg / 100, text=f"Avg: {avg:.0f}% | Best: {best}% | {attempts} attempt(s)")
            st.caption("")
    else:
        st.info("Take a quiz to start tracking progress!")

    total_quizzes = len(progress_data.get("quiz_results", []))
    st.metric("Total Quizzes Taken", total_quizzes)
    st.divider()
    st.divider()
    _status = st.session_state.get("connection_status", {"ok": None})
    _cfg = get_effective_config()
    _label = _cfg.get("provider_label", "AI Provider")
    if _status.get("ok") is True:
        st.markdown(
            f'<span style="color:#4ade80">● {_label} · Connected</span>',
            unsafe_allow_html=True,
        )
    elif _status.get("ok") is False:
        st.markdown(
            f'<span style="color:#ef4444">● {_label} · Disconnected</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span style="color:#94a3b8">● {_label} · Checking...</span>',
            unsafe_allow_html=True,
        )
    st.caption("TechTrainer AI v1.0")
    st.divider()
    st.caption("Developed by B.Vignesh Kumar\nic19939@gmail.com")


# ── Main header ────────────────────────────────────────────────────────────────
st.title("🎓 TechTrainer AI")
st.caption("Your intelligent training assistant for Skill Engineering")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_chat, tab_quiz, tab_path, tab_author, tab_kb, tab_settings = st.tabs(
    ["💬 Chat", "📝 Quiz", "🗺️ Learning Path", "✍️ Content Author", "📚 Knowledge Base", "⚙️ Settings"]
)


# ── TAB: Chat ──────────────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("Ask About Any Technology")
    st.caption("Explore your question with our Knowledge Base")

    # ── Prompt input (top) ───────────────────────────────────────────────────
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your question",
            placeholder="Type your question here… (Shift+Enter for new line)",
            height=80,
            label_visibility="collapsed",
        )
        col_ask, col_clear = st.columns([2, 1])
        with col_ask:
            send = st.form_submit_button("Ask AI", type="primary", use_container_width=True)
        with col_clear:
            clear = st.form_submit_button("🗑 Clear History", use_container_width=True)

    if clear:
        st.session_state.messages = []
        clear_chat_history()
        st.rerun()

    if send and user_input.strip():
        with st.spinner("Thinking..."):
            response = call_agent(user_input.strip())
        question = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": response})
        try:
            save_exchange(question, response)
        except Exception as _save_err:
            st.warning(f"Chat history could not be saved: {_save_err}")
        st.rerun()

    # ── Chat history (bottom) — latest expanded, rest folded ─────────────────
    pairs = []
    msgs = st.session_state.messages
    i = 0
    while i < len(msgs) - 1:
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant":
            pairs.append((msgs[i]["content"], msgs[i + 1]["content"]))
            i += 2
        else:
            i += 1

    if pairs:
        st.divider()
        st.caption(f"**Chat history** — {len(pairs)} exchange(s)")
        for idx, (q, a) in enumerate(reversed(pairs)):
            label = q[:80] + "…" if len(q) > 80 else q
            is_latest = (idx == 0)
            with st.expander(f"#{len(pairs) - idx}  {label}", expanded=is_latest):
                st.markdown(f"**You:** {q}")
                st.markdown("---")
                st.markdown(f"**Assistant:** {a}")


# ── TAB: Quiz ─────────────────────────────────────────────────────────────────
with tab_quiz:
    st.subheader("Test Your Knowledge")

    available_quiz_topics = get_available_topics()
    available_quiz_ids = [t["id"] for t in available_quiz_topics]
    available_quiz_names = {t["id"]: t["display_name"] for t in available_quiz_topics}

    col1, col2, col3 = st.columns(3)
    with col1:
        if not available_quiz_ids:
            st.warning("No topics available yet. Go to 📚 Knowledge Base to upload training documents.")
            quiz_tech = None
        else:
            quiz_tech = st.selectbox(
                "Technology",
                options=available_quiz_ids,
                format_func=lambda x: available_quiz_names.get(x, x),
                key="quiz_tech",
            )
    with col2:
        quiz_diff = st.selectbox("Difficulty", ["beginner", "intermediate", "advanced"], key="quiz_diff")
    with col3:
        quiz_n = st.selectbox("Questions", [5, 10, 15], key="quiz_n")

    if quiz_tech and st.button("🎯 Generate Quiz", type="primary"):
        with st.spinner(f"Generating {quiz_n} {quiz_diff} questions on {quiz_tech}..."):
            prompt = (
                f"Generate a {quiz_n}-question {quiz_diff} quiz on {quiz_tech}. "
                f"Use the quiz_agent tool with technology='{quiz_tech}', "
                f"difficulty='{quiz_diff}', num_questions={quiz_n}."
            )
            raw = call_agent(prompt)

        try:
            quiz_data = _extract_json(raw)
            if quiz_data and "questions" in quiz_data:
                st.session_state["current_quiz"] = quiz_data
                st.session_state["quiz_answers"] = {}
                st.session_state["quiz_submitted"] = False
            elif quiz_data and "error" in quiz_data:
                st.error(f"Quiz generation failed: {quiz_data['error']}")
            else:
                st.error("Could not parse quiz response. Please try again.")
                with st.expander("Debug — raw response"):
                    st.text(raw)
        except Exception as exc:
            st.error(f"Failed to parse quiz: {exc}")
            with st.expander("Debug — raw response"):
                st.text(raw)

    if "current_quiz" in st.session_state and not st.session_state.get("quiz_submitted"):
        quiz_data = st.session_state["current_quiz"]
        questions = quiz_data.get("questions", [])

        if questions:
            st.divider()
            with st.form("quiz_form"):
                for i, q in enumerate(questions):
                    st.markdown(f"**Q{i+1}. {q['question']}**")
                    st.radio(
                        f"q{i}", q["options"], key=f"quiz_q{i}", label_visibility="collapsed"
                    )
                    st.caption(f"Difficulty: {q.get('difficulty', '?')} | Topic: {q.get('topic', '?')}")
                    st.write("")

                submitted = st.form_submit_button("✅ Submit Answers", type="primary")

            if submitted:
                # Read answers from widget keys (correct pattern for forms)
                correct = sum(
                    1 for i, q in enumerate(questions)
                    if st.session_state.get(f"quiz_q{i}") == q["correct_answer"]
                )
                score = int((correct / len(questions)) * 100)
                st.session_state["quiz_submitted"] = True
                passing = score >= quiz_data.get("passing_score", 70)

                save_prompt = (
                    f"Save quiz result: session_id={session_id}, "
                    f"technology={quiz_data.get('technology', quiz_tech)}, "
                    f"difficulty={quiz_data.get('difficulty', quiz_diff)}, "
                    f"score={score}, total_questions={len(questions)}, correct_answers={correct}. "
                    f"Use the progress_agent tool."
                )
                call_agent(save_prompt)

                if passing:
                    st.success(f"🎉 You scored {score}% ({correct}/{len(questions)}) — PASSED!")
                else:
                    st.warning(f"📚 You scored {score}% ({correct}/{len(questions)}) — Keep studying!")

                st.divider()
                st.subheader("Answer Review")
                for i, q in enumerate(questions):
                    user_ans = st.session_state["quiz_answers"].get(i, "")
                    correct_ans = q["correct_answer"]
                    is_correct = user_ans == correct_ans
                    icon = "✅" if is_correct else "❌"
                    st.markdown(f"{icon} **Q{i+1}.** {q['question']}")
                    if not is_correct:
                        st.markdown(f"Your answer: ~~{user_ans}~~ | Correct: **{correct_ans}**")
                    st.info(f"📖 {q['explanation']}")

                if st.button("🔄 Take Another Quiz"):
                    st.session_state.pop("current_quiz", None)
                    st.session_state.pop("quiz_submitted", None)
                    st.rerun()


# ── TAB: Learning Path ────────────────────────────────────────────────────────
with tab_path:
    st.subheader("Your Personalised Learning Path")

    if st.button("🔄 Refresh My Path", type="primary"):
        with st.spinner("Analysing your progress and building recommendations..."):
            available_lp = [t["display_name"] for t in get_available_topics()]
            available_lp_str = ", ".join(available_lp) if available_lp else "none yet"
            raw = call_agent(
                f"Generate a learning path for me. My session_id is {session_id}. "
                f"Only recommend topics from this available list: {available_lp_str}. "
                "Use the learning_path_agent tool."
            )
        lp_data = _extract_json(raw)
        if lp_data:
            st.session_state["learning_path"] = lp_data
        else:
            st.error("Could not parse learning path. Try again.")
            with st.expander("Debug — raw response"):
                st.text(raw)

    if "learning_path" in st.session_state:
        lp = st.session_state["learning_path"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Level", lp.get("current_level", "beginner").title())
        col2.metric("Est. Hours Remaining", f"{lp.get('estimated_hours', 0):.1f}h")
        col3.metric("Next Milestone", lp.get("next_milestone", "—")[:40])

        st.divider()

        col_weak, col_strong = st.columns(2)
        with col_weak:
            st.markdown("### 📉 Areas to Improve")
            for area in lp.get("weak_areas", []):
                st.markdown(f"- {area}")
            if not lp.get("weak_areas"):
                st.success("No weak areas identified yet!")

        with col_strong:
            st.markdown("### 📈 Strong Areas")
            for area in lp.get("strong_areas", []):
                st.markdown(f"- ✅ {area}")
            if not lp.get("strong_areas"):
                st.info("Take some quizzes to build your profile.")

        st.divider()
        st.markdown("### 🗺️ Recommended Study Order")
        topics = lp.get("recommended_topics", [])
        progress_data = get_progress_data()
        completed_techs = set(progress_data.get("technologies", {}).keys())

        for i, topic in enumerate(topics, 1):
            tech_key = topic.split()[0].lower().replace(" ", "_")
            if tech_key in completed_techs:
                st.markdown(f"✅ **{i}. {topic}** *(completed)*")
            elif i == 1:
                st.markdown(f"▶️ **{i}. {topic}** ← *start here*")
            else:
                st.markdown(f"○ {i}. {topic}")


# ── TAB: Content Author ───────────────────────────────────────────────────────
with tab_author:
    st.subheader("Generate Training Modules")
    st.caption("AI will create a complete Markdown training module based on your existing training documents.")

    available_author_topics = sorted(get_available_topics(), key=lambda t: t["display_name"].lower())
    available_author_ids = [t["id"] for t in available_author_topics]
    available_author_names = {t["id"]: t["display_name"] for t in available_author_topics}

    with st.form("author_form"):
        title = st.text_input("Module Title", placeholder="e.g. Introduction to Selenium Locators")
        col1, col2 = st.columns(2)
        with col1:
            if not available_author_ids:
                st.warning("No topics available. Go to 📚 Knowledge Base first.")
                tech = None
            else:
                tech = st.selectbox(
                    "Technology",
                    options=available_author_ids,
                    format_func=lambda x: available_author_names.get(x, x),
                    key="author_tech",
                )
        with col2:
            diff = st.selectbox("Difficulty", ["beginner", "intermediate", "advanced"], key="author_diff")
        objectives = st.text_area(
            "Learning Objectives (one per line)",
            placeholder="Understand locator types\nWrite Page Object Model classes\nUse explicit waits",
        )
        generate_btn = st.form_submit_button("✨ Generate Module", type="primary")

    if generate_btn and title and tech:
        with st.spinner("Generating training module (this may take 30-60 seconds)..."):
            raw = content_author_agent(
                title=title,
                technology=tech,
                difficulty=diff,
                objectives=objectives,
            )

        module_data = _extract_json(raw)
        if module_data:
            st.session_state["generated_module"] = module_data
        else:
            st.error("Failed to parse generated module.")
            with st.expander("Debug — raw response"):
                st.text(raw)

    if "generated_module" in st.session_state:
        module = st.session_state["generated_module"]
        if "error" not in module:
            st.success(f"Module generated: **{module.get('title', 'Untitled')}**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Technology", module.get("technology", "?"))
            col2.metric("Difficulty", module.get("difficulty", "?"))
            col3.metric("Est. Duration", f"{module.get('duration_minutes', 0)} min")

            st.divider()
            st.subheader("Preview")
            content = module.get("content", "")
            st.markdown(content)

            st.divider()
            col_dl, col_path = st.columns(2)
            with col_dl:
                st.download_button(
                    "⬇️ Download as Markdown",
                    data=content,
                    file_name=f"{module.get('technology', 'module')}_{title.lower().replace(' ', '_')}.md",
                    mime="text/markdown",
                )
            with col_path:
                if saved_path := module.get("saved_to"):
                    st.info(f"Saved to: `{saved_path}`")
        else:
            st.error(f"Generation failed: {module['error']}")


# ── TAB: Knowledge Base ───────────────────────────────────────────────────────
with tab_kb:
    st.subheader("Manage Training Documents")
    st.caption(
        "Upload documents or copy files into data/documents/<technology>/ — "
        "then index to make them searchable in Chat, Quiz, and Learning Path."
    )

    # ── Load topics once for this render ────────────────────────────────────
    all_topics = load_all_topics()

    # ── Top action bar ──────────────────────────────────────────────────────
    col_refresh, col_index_all = st.columns([1, 1])
    with col_refresh:
        if st.button("🔄 Refresh Status", key="kb_refresh"):
            st.rerun()
    with col_index_all:
        idx_all = st.session_state.get("idx_all")
        if idx_all and idx_all["thread"].is_alive():
            r = idx_all["result"]
            total = r["total"] or 1
            pct = r["done_count"] / total
            label = f"Indexing {r['current'] or '…'}  ({r['done_count']}/{total} topics)"
            col_bar2, col_stop2 = st.columns([3, 1])
            with col_bar2:
                st.progress(pct, text=label)
            with col_stop2:
                if st.button("⏹ Stop All", key="stop_idx_all"):
                    idx_all["cancel"].set()
            time.sleep(0.4)
            st.rerun()
        elif idx_all:
            r = idx_all["result"]
            del st.session_state["idx_all"]
            if idx_all["cancel"].is_set():
                st.warning(f"Index All stopped — {r['chunks']} chunks across {r['done_count']} topic(s).")
            else:
                st.success(f"Indexed {r['done_count']} topic(s), {r['chunks']} chunks total.")
            st.rerun()
        else:
            if st.button("⚡ Index All", key="kb_index_all", type="primary"):
                to_index = [
                    t for t in all_topics
                    if t["status"] in ("PENDING", "DISABLED") and t["file_count"] > 0
                ]
                if to_index:
                    cancel_event = threading.Event()
                    result = {"chunks": 0, "done_count": 0, "total": len(to_index), "current": ""}

                    def _run_all(topics=to_index, c=cancel_event, r=result):
                        for t in topics:
                            if c.is_set():
                                break
                            r["current"] = t["display_name"]
                            try:
                                r["chunks"] += index_technology(t["id"], cancel_event=c)
                            except Exception:
                                pass
                            r["done_count"] += 1

                    thread = threading.Thread(target=_run_all, daemon=True)
                    st.session_state["idx_all"] = {
                        "thread": thread, "cancel": cancel_event, "result": result
                    }
                    thread.start()
                    st.rerun()
                else:
                    st.info("No topics with files to index.")

    # ── Summary metrics ─────────────────────────────────────────────────────
    available_count = sum(1 for t in all_topics if t["status"] == "AVAILABLE")
    pending_count = sum(1 for t in all_topics if t["status"] == "PENDING")
    disabled_count = sum(1 for t in all_topics if t["status"] == "DISABLED")
    m1, m2, m3 = st.columns(3)
    m1.metric("✓ Available", available_count)
    m2.metric("⏳ Pending", pending_count)
    m3.metric("✗ Disabled", disabled_count)

    st.divider()

    # ── Add Custom Topic ─────────────────────────────────────────────────────
    with st.expander("➕ Add Custom Topic"):
        with st.form("add_topic_form"):
            new_name = st.text_input("Topic Name *", placeholder="e.g. Kubernetes")
            new_desc = st.text_area(
                "Description (optional)",
                placeholder="Container orchestration platform training material",
            )
            if st.form_submit_button("Create Topic"):
                if new_name.strip():
                    try:
                        create_custom_topic(new_name.strip(), new_desc.strip())
                        st.success(f"Topic '{new_name}' created successfully.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                else:
                    st.error("Topic name is required.")

    st.divider()

    # ── Built-in Technologies ────────────────────────────────────────────────
    builtin = [t for t in all_topics if t.get("is_builtin", True)]
    custom = [t for t in all_topics if not t.get("is_builtin", True)]

    st.markdown("### 🔒 Built-in Technologies")
    _render_topic_grid(builtin)

    if custom:
        st.markdown("### ✏️ Custom Topics")
        _render_topic_grid(custom)


# ── TAB: Settings ─────────────────────────────────────────────────────────────
with tab_settings:
    st.subheader("⚙️ AI Provider Settings")
    st.caption(
        "Choose your AI provider and enter credentials. "
        "Credentials are saved to .env. Non-sensitive settings are saved to data/settings.json."
    )

    # ── Connection status card ───────────────────────────────────────────────
    _conn = st.session_state.get("connection_status", {"ok": None, "message": "Unknown", "latency_ms": None})
    _cfg_now = get_effective_config()
    _testable_providers = {"ollama", "lmstudio", "custom"}
    _can_test = _cfg_now.get("active_provider") in _testable_providers
    if _can_test:
        col_conn, col_test_btn = st.columns([4, 1])
    else:
        col_conn = st.container()
    with col_conn:
        if _conn.get("ok") is True:
            st.success(f"✓ Connected to **{_cfg_now['provider_label']}** · {_conn.get('message', '')}")
        elif _conn.get("ok") is False:
            st.error(f"✗ Cannot reach **{_cfg_now['provider_label']}**: {_conn.get('message', '')}")
        else:
            st.info(f"● **{_cfg_now['provider_label']}** — provider configured")
    if _can_test:
        with col_test_btn:
            st.write("")
            if st.button("🔌 Test Connection", key="settings_test_conn"):
                with st.spinner("Testing..."):
                    result = test_connection()
                    st.session_state["connection_status"] = result
                st.rerun()

    st.divider()

    # ── Provider selector ────────────────────────────────────────────────────
    st.markdown("### Provider")
    provider_names = [PROVIDER_LABELS[p] for p in PROVIDER_IDS]
    current_provider = _cfg_now.get("active_provider", "bedrock")
    current_idx = PROVIDER_IDS.index(current_provider) if current_provider in PROVIDER_IDS else 0
    selected_idx = st.radio(
        "Select Provider",
        range(len(PROVIDER_IDS)),
        format_func=lambda i: provider_names[i],
        index=current_idx,
        horizontal=True,
        label_visibility="collapsed",
        key="settings_provider_radio",
    )
    selected_provider = PROVIDER_IDS[selected_idx]

    # ── Embedding provider selector (outside form so conditional fields react immediately) ──
    st.divider()
    st.markdown("### Embedding Provider")
    st.caption(
        "Changing the embedding provider requires re-indexing all Knowledge Base collections "
        "because stored vector dimensions must match the query dimensions."
    )
    _emb_cfg = get_embedding_config()
    _emb_provider_now = _emb_cfg.get("embedding_provider", "bedrock")
    _emb_provider_names = [EMBEDDING_PROVIDER_LABELS[p] for p in EMBEDDING_PROVIDER_IDS]
    _emb_idx_now = EMBEDDING_PROVIDER_IDS.index(_emb_provider_now) if _emb_provider_now in EMBEDDING_PROVIDER_IDS else 0
    _emb_selected_idx = st.radio(
        "Embedding Provider",
        range(len(EMBEDDING_PROVIDER_IDS)),
        format_func=lambda i: _emb_provider_names[i],
        index=_emb_idx_now,
        horizontal=True,
        label_visibility="collapsed",
        key="s_emb_provider",
    )
    _emb_selected_provider = EMBEDDING_PROVIDER_IDS[_emb_selected_idx]
    _emb_default_model = EMBEDDING_MODEL_DEFAULTS[_emb_selected_provider]

    ec1, ec2 = st.columns(2)
    with ec1:
        _emb_model = st.text_input(
            "Embedding Model",
            value=_emb_cfg.get("embedding_model") or _emb_default_model,
            placeholder=_emb_default_model or "model name",
            key="s_emb_model",
        )
    with ec2:
        if _emb_selected_provider in ("ollama", "custom"):
            _emb_base_url = st.text_input(
                "Base URL",
                value=_emb_cfg.get("embedding_base_url", ""),
                placeholder="http://localhost:11434" if _emb_selected_provider == "ollama" else "https://...",
                key="s_emb_base_url",
            )
        else:
            _emb_base_url = _emb_cfg.get("embedding_base_url", "")
    if _emb_selected_provider in ("openai", "custom"):
        _emb_api_key = st.text_input(
            "Embedding API Key",
            value="",
            type="password",
            placeholder="Leave blank to keep existing",
            key="s_emb_api_key",
        )
    else:
        _emb_api_key = ""

    st.divider()
    st.markdown(f"### {provider_names[selected_idx]} Credentials")

    # Initialize all form variables from current config before the form
    _api_key = _cfg_now.get("llm_api_key", "")
    _base_url = _cfg_now.get("llm_base_url", "")
    _model_name = _cfg_now.get("llm_model", "")
    _aws_region = _cfg_now.get("aws_region", "us-east-1")
    _bedrock_model = _cfg_now.get("bedrock_model_id", "us.anthropic.claude-sonnet-4-20250514-v1:0")
    _guardrail_id = _cfg_now.get("bedrock_guardrail_id", "")
    # Never pre-fill secrets — user must re-enter
    _aws_access_key = ""
    _aws_secret_key = ""

    with st.form("settings_form"):
        if selected_provider == "bedrock":
            c1, c2 = st.columns(2)
            with c1:
                _aws_region = st.text_input("AWS Region", value=_aws_region, key="s_region")
                _aws_access_key = st.text_input("Access Key ID", value="", type="password", key="s_ak",
                                                 placeholder="Leave blank to keep existing")
                _guardrail_id = st.text_input("Guardrail ID (optional)", value=_guardrail_id, key="s_gid")
            with c2:
                _bedrock_model = st.text_input("Model ID", value=_bedrock_model, key="s_model_id")
                _aws_secret_key = st.text_input("Secret Access Key", value="", type="password", key="s_sk",
                                                 placeholder="Leave blank to keep existing")

        elif selected_provider == "ollama":
            _base_url = st.text_input("Base URL", value=_base_url or "http://localhost:11434", key="s_base")
            st.caption("Ollama does not require an API key.")

        elif selected_provider == "lmstudio":
            c1, c2 = st.columns(2)
            with c1:
                _base_url = st.text_input("Base URL", value=_base_url or "http://localhost:1234/v1", key="s_base")
            with c2:
                _model_name = st.text_input("Model Name (optional)", value=_model_name,
                                             placeholder="Leave blank to use loaded model", key="s_model")

        elif selected_provider == "openrouter":
            c1, c2 = st.columns(2)
            with c1:
                _api_key = st.text_input("API Key", value="", type="password", key="s_apikey",
                                          placeholder="sk-or-...")
            with c2:
                _model_name = st.text_input("Model", value=_model_name or "anthropic/claude-3-sonnet", key="s_model")

        elif selected_provider == "gemini":
            c1, c2 = st.columns(2)
            with c1:
                _api_key = st.text_input("API Key", value="", type="password", key="s_apikey",
                                          placeholder="AIza...")
            with c2:
                _model_name = st.text_input("Model", value=_model_name or "gemini-1.5-pro", key="s_model")

        elif selected_provider == "openai":
            c1, c2 = st.columns(2)
            with c1:
                _api_key = st.text_input("API Key", value="", type="password", key="s_apikey",
                                          placeholder="sk-...")
                _base_url = st.text_input("Base URL (optional — for Enterprise/Azure)",
                                           value=_base_url, key="s_base")
            with c2:
                _model_name = st.text_input("Model", value=_model_name or "gpt-4o", key="s_model")

        elif selected_provider == "custom":
            c1, c2 = st.columns(2)
            with c1:
                _base_url = st.text_input("Base URL *", value=_base_url,
                                           placeholder="https://my-api.example.com/v1", key="s_base")
                _api_key = st.text_input("API Key (optional)", value="", type="password", key="s_apikey")
            with c2:
                _model_name = st.text_input("Model Name (optional)", value=_model_name, key="s_model")

        st.divider()
        st.markdown("### Model Parameters")
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            _temperature = st.number_input(
                "Temperature", min_value=0.0, max_value=1.0, step=0.05,
                value=float(_cfg_now.get("temperature", 0.3)), key="s_temp",
            )
        with pc2:
            _top_p = st.number_input(
                "Top P", min_value=0.0, max_value=1.0, step=0.05,
                value=float(_cfg_now.get("top_p", 0.9)), key="s_top_p",
            )
        with pc3:
            _max_tokens = st.number_input(
                "Max Tokens", min_value=256, max_value=32768, step=256,
                value=int(_cfg_now.get("max_tokens", 32768)), key="s_max_tokens",
            )

        save_btn = st.form_submit_button("💾 Save to .env & settings.json", type="primary")

    # ── Save handler ─────────────────────────────────────────────────────────
    if save_btn:
        env_updates: dict = {"ACTIVE_PROVIDER": selected_provider}
        if selected_provider == "bedrock":
            env_updates["AWS_REGION"] = _aws_region
            env_updates["BEDROCK_MODEL_ID"] = _bedrock_model
            env_updates["BEDROCK_GUARDRAIL_ID"] = _guardrail_id
            if _aws_access_key:
                env_updates["AWS_ACCESS_KEY_ID"] = _aws_access_key
            if _aws_secret_key:
                env_updates["AWS_SECRET_ACCESS_KEY"] = _aws_secret_key
        else:
            env_updates["LLM_BASE_URL"] = _base_url
            env_updates["LLM_MODEL"] = _model_name
            if _api_key:
                env_updates["LLM_API_KEY"] = _api_key

        env_updates["AGENT_TEMPERATURE"] = str(_temperature)
        env_updates["AGENT_TOP_P"] = str(_top_p)
        env_updates["AGENT_MAX_TOKENS"] = str(int(_max_tokens))
        env_updates["EMBEDDING_PROVIDER"] = _emb_selected_provider
        env_updates["EMBEDDING_MODEL"] = _emb_model
        env_updates["EMBEDDING_BASE_URL"] = _emb_base_url
        if _emb_api_key:
            env_updates["EMBEDDING_API_KEY"] = _emb_api_key

        try:
            write_env_values(env_updates)
        except Exception as e:
            st.error(f"Cannot write to .env — check file permissions: {e}")

        save_settings({
            "active_provider": selected_provider,
            "model_name": _model_name,
            "base_url": _base_url,
            "temperature": _temperature,
            "top_p": _top_p,
            "max_tokens": int(_max_tokens),
            "embedding_provider": _emb_selected_provider,
            "embedding_model": _emb_model,
            "embedding_base_url": _emb_base_url,
        })
        st.success("Settings saved.")
        st.info("Click **Apply & Restart Session** below for provider changes to take effect.")

    # ── Apply & Restart ───────────────────────────────────────────────────────
    if st.button("↺ Apply & Restart Session", key="settings_restart"):
        for _k in list(st.session_state.keys()):
            del st.session_state[_k]
        st.rerun()

    st.caption(
        "Credentials are written to .env (gitignored). "
        "Non-sensitive config is written to data/settings.json. "
        "Values already in .env are pre-loaded in fields above. "
        "Password fields are always blank — re-enter only if you want to change them."
    )

    # ── Topic Classification ──────────────────────────────────────────────────
    st.divider()
    st.subheader("🗂️ Topic Classification")
    st.caption(
        "Control which topics appear under Built-in Technologies or Custom Topics "
        "in the Knowledge Base tab. Demoted built-in topics become deletable."
    )

    _cls_settings_cache = load_settings()
    _saved_overrides = _cls_settings_cache.get("topic_classifications", {})
    _all_topics_cls = load_all_topics()
    _builtin_ids = {t["id"] for t in BUILTIN_TOPICS}
    _cls_options = ["Built-in", "Custom"]
    _new_overrides: dict = {}

    for _topic in _all_topics_cls:
        _tid = _topic["id"]
        _default_class = "builtin" if _tid in _builtin_ids else "custom"
        _current_class = _saved_overrides.get(_tid, _default_class)
        _current_label = "Built-in" if _current_class == "builtin" else "Custom"

        _col_name, _col_select = st.columns([3, 1])
        with _col_name:
            st.write(_topic["display_name"])
        with _col_select:
            _selected = st.selectbox(
                label="classification",
                options=_cls_options,
                index=_cls_options.index(_current_label),
                key=f"cls_{_tid}",
                label_visibility="collapsed",
            )
        _resolved = "builtin" if _selected == "Built-in" else "custom"
        if _resolved != _default_class:
            _new_overrides[_tid] = _resolved

    if st.button("💾 Save Classifications", key="save_classifications"):
        try:
            _cls_settings_cache["topic_classifications"] = _new_overrides
            save_settings(_cls_settings_cache)
            st.success("Topic classifications saved.")
            st.rerun()
        except Exception as _e:
            st.error(f"Could not save classifications: {_e}")
