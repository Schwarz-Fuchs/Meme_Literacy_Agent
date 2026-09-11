import os
import streamlit as st
from agent import run
from chat_store import load_conversations, new_conversation, save_conversations, touch


st.set_page_config(page_title="Meme Literacy Agent", page_icon="🧠", layout="wide")
st.title("🧠 Meme Literacy Agent")
st.caption("把抽象处境转成 meme 的笑点结构，再检索、重排和反思。")


def initialise_conversations() -> None:
    if "conversations" in st.session_state:
        return
    conversations = load_conversations()
    if not conversations:
        conversation = new_conversation()
        conversations[conversation["id"]] = conversation
        save_conversations(conversations)
    st.session_state.conversations = conversations
    st.session_state.active_conversation_id = max(
        conversations,
        key=lambda conversation_id: conversations[conversation_id]["updated_at"],
    )


def persist() -> None:
    save_conversations(st.session_state.conversations)


initialise_conversations()

with st.sidebar:
    st.subheader("对话")
    if st.button("＋ 新建对话", use_container_width=True):
        conversation = new_conversation()
        st.session_state.conversations[conversation["id"]] = conversation
        st.session_state.active_conversation_id = conversation["id"]
        persist()
        st.rerun()

    ordered_ids = sorted(
        st.session_state.conversations,
        key=lambda conversation_id: st.session_state.conversations[conversation_id]["updated_at"],
        reverse=True,
    )
    active_id = st.session_state.active_conversation_id
    selected_id = st.selectbox(
        "历史对话",
        ordered_ids,
        index=ordered_ids.index(active_id),
        format_func=lambda conversation_id: st.session_state.conversations[conversation_id]["title"],
        label_visibility="collapsed",
    )
    if selected_id != active_id:
        st.session_state.active_conversation_id = selected_id
        st.rerun()

    if len(ordered_ids) > 1 and st.button("删除当前对话", use_container_width=True):
        del st.session_state.conversations[active_id]
        st.session_state.active_conversation_id = ordered_ids[1] if ordered_ids[0] == active_id else ordered_ids[0]
        persist()
        st.rerun()

    st.divider()
    st.subheader("上下文控制")
    context_user_turns = st.slider(
        "历史用户输入数量",
        min_value=0,
        max_value=10,
        value=3,
        help="0 表示只发送当前输入。大于 0 时，会发送最近 N 条用户输入，以及其后的 Agent 回复。",
    )
    if context_user_turns == 0:
        st.caption("本轮只使用当前输入，不读取历史。")
    else:
        st.caption(f"本轮使用最近 {context_user_turns} 条用户输入作为短期记忆。")

    st.divider()
    st.subheader("运行状态")
    if os.getenv("OPENAI_API_KEY"):
        st.success(f"LLM 已启用：{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
    else:
        st.warning("未发现 OPENAI_API_KEY：意图理解将使用规则回退，但图片检索仍会运行。")
    st.markdown("**LLM真的能懂meme吗**")

active_chat = st.session_state.conversations[st.session_state.active_conversation_id]
history = active_chat["messages"]

for message in history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("描述一个你想用 meme 回应的场景……")
if prompt:
    history.append({"role": "user", "content": prompt})
    if active_chat["title"] == "新对话":
        active_chat["title"] = prompt.strip().replace("\n", " ")[:24] or "新对话"
    touch(active_chat)
    persist()
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Agent 正在理解笑点、检索并重排……"):
            result = run(
                prompt,
                history[:-1],
                context_user_turns=context_user_turns,
            )
        st.write(result["answer"])
        with st.expander("查看 Agent 思考轨迹（课程演示）"):
            st.json(result["intent"])
            for step in result["trace"]:
                st.write(f"- {step}")
        if result["ranked"]:
            columns = st.columns(3)
            for index, meme in enumerate(result["ranked"]):
                with columns[index % 3]:
                    st.image(meme["thumbnail"] or meme["url"], use_container_width=True)
                    st.caption(f"{meme['title'][:90]}\n\n分数 {meme['score']} · {meme['why']}")
                    st.link_button("打开来源", meme["url"])
    history.append({"role": "assistant", "content": result["answer"]})
    touch(active_chat)
    persist()
