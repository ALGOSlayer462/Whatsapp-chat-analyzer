import hashlib

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import helper
import ml_features as ml
import preprocessor

# ------------------------------------------------------------------ page setup
st.set_page_config(page_title="WhatsApp Chat Analyzer", page_icon="💬", layout="wide")

GREEN, TEAL, BLUE, YELLOW, PINK = "#25D366", "#128C7E", "#34B7F1", "#FFD166", "#EF476F"
PALETTE = [GREEN, BLUE, YELLOW, PINK, TEAL, "#B388FF", "#FF9F1C", "#2EC4B6"]

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1250px;}
    .hero {
        background: linear-gradient(135deg, #075E54 0%, #128C7E 55%, #25D366 100%);
        padding: 28px 32px; border-radius: 18px; margin-bottom: 22px;
        box-shadow: 0 8px 30px rgba(37,211,102,.18);
    }
    .hero h1 {margin: 0; font-size: 2.1rem; color: #fff;}
    .hero p {margin: 6px 0 0 0; color: #d9fdd3; font-size: 1rem;}
    .kpi {
        background: #111B21; border: 1px solid #1f2c34; border-left: 5px solid var(--c);
        border-radius: 14px; padding: 16px 18px; transition: transform .15s ease;
    }
    .kpi:hover {transform: translateY(-3px);}
    .kpi .label {color: #8696A0; font-size: .85rem; letter-spacing: .04em; text-transform: uppercase;}
    .kpi .value {color: #E9EDEF; font-size: 2rem; font-weight: 700; line-height: 1.25;}
    .kpi .icon {float: right; font-size: 1.6rem; opacity: .9;}
    .insight {
        background: #111B21; border: 1px dashed #2a3942; border-radius: 12px;
        padding: 12px 16px; color: #c5ced3; font-size: .95rem; height: 100%;
    }
    .insight b {color: #25D366;}
    .stTabs [data-baseweb="tab-list"] {gap: 6px;}
    .stTabs [data-baseweb="tab"] {
        background: #111B21; border-radius: 10px 10px 0 0; padding: 8px 18px;
    }
    .stTabs [aria-selected="true"] {background: #1f2c34; color: #25D366;}
    </style>
    """,
    unsafe_allow_html=True,
)


def style(fig, height=380):
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color="#E9EDEF"), legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="#1f2c34")
    fig.update_yaxes(gridcolor="#1f2c34")
    return fig


def kpi(col, label, value, icon, color):
    col.markdown(
        f'<div class="kpi" style="--c:{color}"><span class="icon">{icon}</span>'
        f'<div class="label">{label}</div><div class="value">{value:,}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner="Chat process ho rahi hai...")
def load(data: str):
    return preprocessor.preprocess(data)


@st.cache_resource(show_spinner="ML models train ho rahe hain (pehli baar ~10-20 sec)...")
def get_models():
    return ml.train_all()


@st.cache_data(show_spinner="Messages par sentiment / toxicity / spam models chal rahe hain...")
def run_ml(_models, user, key, _df):
    return ml.analyze_chat(_models, user, _df)


@st.cache_resource(show_spinner="Next-word model ban raha hai...")
def get_next_word(user, key, _df):
    return ml.build_next_word_model(user, _df)


def model_card(m, class_names):
    mt = m["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{mt['accuracy']:.1%}")
    c2.metric("Precision", f"{mt['precision']:.1%}")
    c3.metric("Recall", f"{mt['recall']:.1%}")
    c4.metric("F1-score", f"{mt['f1']:.1%}")
    st.caption(f"{mt['average']} average • train {mt['train_size']:,} / hold-out test {mt['test_size']:,} samples "
               f"• hand-written seed examples: {mt['seed_count']:,}")
    cm = pd.DataFrame(mt["confusion"], index=[f"Actual {c}" for c in class_names],
                      columns=[f"Pred {c}" for c in class_names])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale=["#0B141A", TEAL, GREEN], aspect="auto")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(style(fig, 260), width="stretch")


SENT_COLORS = {"Positive": GREEN, "Neutral": "#8696A0", "Negative": PINK}

# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("## 💬 Chat Analyzer")
st.sidebar.caption("WhatsApp → Export chat → Without media → .txt upload karo")
uploaded_file = st.sidebar.file_uploader("Chat file (.txt)", type=["txt"])

if uploaded_file is None:
    st.markdown(
        '<div class="hero"><h1>💬 WhatsApp Chat Analyzer</h1>'
        "<p>Apni chat ke patterns dekho: kaun kitna bolta hai, kab active hai, kaun se emoji aur words sabse zyada.</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    c1.info("**1.** WhatsApp mein chat kholo → ⋮ → More → Export chat → *Without media*")
    c2.info("**2.** Wo `.txt` file left sidebar mein upload karo")
    c3.info("**3.** User select karo, aur saara analysis turant dikhega")
    st.stop()

raw_text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
file_key = hashlib.md5(raw_text.encode("utf-8", errors="ignore")).hexdigest()
try:
    df = load(raw_text)
except ValueError as e:
    st.error(str(e))
    st.stop()

users = sorted(u for u in df["user"].unique() if u != "group_notification")
selected_user = st.sidebar.selectbox("👤 Analysis kiske liye", ["Overall"] + users)

ml_on = st.sidebar.toggle("🤖 ML Lab chalao", value=False,
                          help="Sentiment, Toxicity/Spam detection aur Next-word prediction (pehli baar models train hote hain)")

with st.sidebar.expander("Raw data preview"):
    st.caption(f"Total rows: {len(df):,}")
    st.dataframe(df.head(10), width="stretch")

# ------------------------------------------------------------------ header + KPIs
num_messages, words, num_media, num_links = helper.fetch_stats(selected_user, df)
if num_messages == 0:
    st.warning("Is selection ke liye koi message nahi mila.")
    st.stop()

st.markdown(
    f'<div class="hero"><h1>💬 {"Group / Chat Overview" if selected_user == "Overall" else selected_user}</h1>'
    f"<p>{df['only_date'].min():%d %b %Y} → {df['only_date'].max():%d %b %Y} "
    f"&nbsp;•&nbsp; {len(users)} participants</p></div>",
    unsafe_allow_html=True,
)

k1, k2, k3, k4 = st.columns(4)
kpi(k1, "Total Messages", num_messages, "💬", GREEN)
kpi(k2, "Total Words", words, "📝", BLUE)
kpi(k3, "Media Shared", num_media, "🖼️", YELLOW)
kpi(k4, "Links Shared", num_links, "🔗", PINK)

# quick insights
daily = helper.daily_timeline(selected_user, df)
heat = helper.activity_heatmap(selected_user, df)
peak = daily.loc[daily["message"].idxmax()]
busy_hour = heat.sum().idxmax() if not heat.empty else "-"
avg_day = num_messages / max(len(daily), 1)

st.write("")
i1, i2, i3 = st.columns(3)
i1.markdown(f'<div class="insight">🔥 Sabse busy din: <b>{peak["only_date"]:%d %b %Y}</b> ({int(peak["message"])} messages)</div>', unsafe_allow_html=True)
i2.markdown(f'<div class="insight">⏰ Sabse active ghanta: <b>{busy_hour}</b></div>', unsafe_allow_html=True)
i3.markdown(f'<div class="insight">📈 Average: <b>{avg_day:.1f}</b> messages / active din</div>', unsafe_allow_html=True)
st.write("")

# ------------------------------------------------------------------ tabs
tab_time, tab_act, tab_users, tab_words, tab_emoji, tab_ml = st.tabs(
    ["📅 Timeline", "🗓️ Activity", "👥 Users", "🔤 Words", "😀 Emoji", "🤖 ML Lab"]
)

with tab_time:
    st.subheader("Monthly Timeline")
    t = helper.monthly_timeline(selected_user, df)
    fig = px.area(t, x="time", y="message", markers=True, color_discrete_sequence=[GREEN])
    fig.update_xaxes(title=None); fig.update_yaxes(title="Messages")
    st.plotly_chart(style(fig), width="stretch")

    st.subheader("Daily Timeline")
    fig = px.line(daily, x="only_date", y="message", color_discrete_sequence=[BLUE])
    fig.update_xaxes(title=None, rangeslider_visible=True); fig.update_yaxes(title="Messages")
    st.plotly_chart(style(fig, 420), width="stretch")

with tab_act:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Most busy day")
        d = helper.week_activity_map(selected_user, df)
        fig = px.bar(x=d.index, y=d.values, color=d.values, color_continuous_scale=["#128C7E", GREEN])
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(title=None); fig.update_yaxes(title="Messages")
        st.plotly_chart(style(fig), width="stretch")
    with c2:
        st.subheader("Most busy month")
        m = helper.month_activity_map(selected_user, df)
        fig = px.bar(x=m.index, y=m.values, color=m.values, color_continuous_scale=["#8a5a00", YELLOW])
        fig.update_layout(coloraxis_showscale=False)
        fig.update_xaxes(title=None); fig.update_yaxes(title="Messages")
        st.plotly_chart(style(fig), width="stretch")

    st.subheader("Weekly Activity Map")
    if heat.empty:
        st.warning("Heatmap ke liye data available nahi hai.")
    else:
        fig = px.imshow(heat, aspect="auto", color_continuous_scale=["#0B141A", TEAL, GREEN],
                        labels=dict(x="Hour", y="Day", color="Messages"))
        st.plotly_chart(style(fig, 400), width="stretch")

with tab_users:
    if selected_user == "Overall":
        x, pct = helper.most_busy_users(df)
        c1, c2 = st.columns([3, 2])
        with c1:
            st.subheader("Most Busy Users")
            fig = px.bar(x=x.index, y=x.values, color=x.index, color_discrete_sequence=PALETTE)
            fig.update_layout(showlegend=False)
            fig.update_xaxes(title=None); fig.update_yaxes(title="Messages")
            st.plotly_chart(style(fig), width="stretch")
        with c2:
            st.subheader("Contribution %")
            st.dataframe(pct, width="stretch", hide_index=True,
                         column_config={"percent": st.column_config.ProgressColumn(
                             "percent", format="%.2f%%", min_value=0, max_value=100)})
    else:
        st.info("Users ka comparison dekhne ke liye sidebar mein **Overall** select karo.")

with tab_words:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Wordcloud")
        wc = helper.create_wordcloud(selected_user, df)
        if wc is None:
            st.warning("Wordcloud ke liye text messages nahi mile.")
        else:
            fig, ax = plt.subplots(facecolor="#0B141A")
            ax.imshow(wc.recolor(colormap="summer"), interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
    with c2:
        st.subheader("Most common words")
        mc = helper.most_common_words(selected_user, df)
        if mc.empty:
            st.warning("Common words ke liye data nahi hai.")
        else:
            fig = px.bar(mc, x="count", y="word", orientation="h", color="count",
                         color_continuous_scale=["#128C7E", GREEN])
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            fig.update_xaxes(title="Count"); fig.update_yaxes(title=None)
            st.plotly_chart(style(fig, 520), width="stretch")

with tab_emoji:
    edf = helper.emoji_helper(selected_user, df)
    if edf.empty:
        st.warning("Is selection mein koi emoji nahi mila.")
    else:
        c1, c2 = st.columns([2, 3])
        with c1:
            st.subheader("Emoji list")
            st.dataframe(edf, width="stretch", hide_index=True, height=420)
        with c2:
            st.subheader("Top emojis")
            top = edf.head(8)
            fig = go.Figure(go.Pie(labels=top["emoji"], values=top["count"], hole=0.5,
                                   textinfo="label+percent", textfont=dict(size=18),
                                   marker=dict(colors=PALETTE)))
            st.plotly_chart(style(fig, 420), width="stretch")


# ------------------------------------------------------------------ ML Lab
with tab_ml:
    if not ml_on:
        st.info("Left sidebar mein **🤖 ML Lab chalao** toggle ON karo. Pehli baar models train honge (~10-20 sec), phir sab fast hai.")
    else:
        models = get_models()
        res = run_ml(models, selected_user, file_key, df)
        if res.empty:
            st.warning("Is selection mein analyze karne layak text messages nahi mile.")
            st.stop()

        s_tab, t_tab, n_tab = st.tabs(["😊 Sentiment Analysis", "🚫 Toxicity & Spam", "🔮 Next-word Prediction"])

        # ---------------------------------------------------------- sentiment
        with s_tab:
            st.caption("Model: TF-IDF (word + character n-grams) + Logistic Regression, 3 classes: Negative / Neutral / Positive")
            counts = res["sentiment"].value_counts()
            total = len(res)
            m1, m2, m3 = st.columns(3)
            for col, name, icon in [(m1, "Positive", "😊"), (m2, "Neutral", "😐"), (m3, "Negative", "😞")]:
                n = int(counts.get(name, 0))
                col.metric(f"{icon} {name}", f"{n:,}", f"{n / total:.1%} of messages", delta_color="off")

            c1, c2 = st.columns([2, 3])
            with c1:
                st.subheader("Mood split")
                fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55,
                                       marker=dict(colors=[SENT_COLORS[x] for x in counts.index]),
                                       textinfo="label+percent"))
                st.plotly_chart(style(fig, 340), width="stretch")
            with c2:
                st.subheader("Sentiment timeline (monthly average)")
                tl = (res.groupby(["year", "month_num", "month"])["sentiment_score"].mean().reset_index())
                tl["time"] = tl["month"] + "-" + tl["year"].astype(str)
                fig = px.line(tl, x="time", y="sentiment_score", markers=True, color_discrete_sequence=[GREEN])
                fig.add_hline(y=0, line_dash="dot", line_color="#8696A0")
                fig.update_yaxes(title="Score (-1 negative ... +1 positive)", range=[-1, 1]); fig.update_xaxes(title=None)
                st.plotly_chart(style(fig, 340), width="stretch")

            if selected_user == "Overall":
                st.subheader("Kaun kitna positive hai? (average score per user)")
                per_user = res.groupby("user")["sentiment_score"].agg(["mean", "count"])
                per_user = per_user[per_user["count"] >= 10].sort_values("mean", ascending=False).reset_index()
                if per_user.empty:
                    st.caption("Kisi user ke 10 se zyada messages nahi hain.")
                else:
                    fig = px.bar(per_user, x="user", y="mean", color="mean", range_color=[-1, 1],
                                 color_continuous_scale=[PINK, "#8696A0", GREEN])
                    fig.update_layout(coloraxis_showscale=False)
                    fig.update_xaxes(title=None); fig.update_yaxes(title="Avg sentiment score")
                    st.plotly_chart(style(fig, 340), width="stretch")

            cols = ["user", "message", "sentiment_score"]
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("😊 Sabse positive messages")
                st.dataframe(res.nlargest(7, "sentiment_score")[cols].round(2), hide_index=True, width="stretch")
            with c2:
                st.subheader("😞 Sabse negative messages")
                st.dataframe(res.nsmallest(7, "sentiment_score")[cols].round(2), hide_index=True, width="stretch")

            st.subheader("✍️ Khud try karo")
            txt = st.text_input("Koi bhi message likho (Hinglish / English)", key="try_sent")
            if txt.strip():
                sc, lb = ml.predict_sentiment_texts(models, [txt])
                st.markdown(f"**{lb[0]}**  •  score = `{sc[0]:+.2f}`")
            with st.expander("📊 Model accuracy aur confusion matrix"):
                model_card(models["sentiment"], ["Negative", "Neutral", "Positive"])
                st.caption("Training data: 10,000 English tweets (NLTK twitter_samples) + SMS messages (neutral) + chhota hand-written Hinglish set.")

        # ---------------------------------------------------------- toxicity & spam
        with t_tab:
            st.caption("Models: TF-IDF + Logistic Regression (alag-alag model: Toxicity, Spam). "
                       "Threshold = kitni probability par message flag ho.")
            sc1, sc2 = st.columns(2)
            tox_th = sc1.slider("Toxicity threshold", 0.30, 0.95, 0.60, 0.05)
            spam_th = sc2.slider("Spam threshold", 0.30, 0.95, 0.70, 0.05)
            res["is_toxic"] = res["toxic_prob"] >= tox_th
            res["is_spam"] = res["spam_prob"] >= spam_th

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("🚫 Toxic messages", f"{int(res['is_toxic'].sum()):,}")
            k2.metric("Toxic %", f"{res['is_toxic'].mean():.2%}")
            k3.metric("📢 Spam messages", f"{int(res['is_spam'].sum()):,}")
            k4.metric("Spam %", f"{res['is_spam'].mean():.2%}")

            if selected_user == "Overall":
                g = res.groupby("user").agg(messages=("message", "size"), toxic=("is_toxic", "sum"), spam=("is_spam", "sum"))
                g = g[g["messages"] >= 10]
                if not g.empty:
                    g["Toxic %"] = g["toxic"] / g["messages"] * 100
                    g["Spam %"] = g["spam"] / g["messages"] * 100
                    long = g.reset_index().melt(id_vars="user", value_vars=["Toxic %", "Spam %"], var_name="type", value_name="percent")
                    st.subheader("User-wise flagged messages (%)")
                    fig = px.bar(long, x="user", y="percent", color="type", barmode="group",
                                 color_discrete_map={"Toxic %": PINK, "Spam %": YELLOW})
                    fig.update_xaxes(title=None)
                    st.plotly_chart(style(fig, 340), width="stretch")

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("🚫 Flagged toxic messages")
                tox = res[res["is_toxic"]].sort_values("toxic_prob", ascending=False)[["user", "message", "toxic_prob"]].head(25)
                if tox.empty:
                    st.success("Koi toxic message nahi mila 🎉")
                else:
                    st.dataframe(tox.round(2), hide_index=True, width="stretch")
            with c2:
                st.subheader("📢 Flagged spam messages")
                sp = res[res["is_spam"]].sort_values("spam_prob", ascending=False)[["user", "message", "spam_prob"]].head(25)
                if sp.empty:
                    st.success("Koi spam message nahi mila 🎉")
                else:
                    st.dataframe(sp.round(2), hide_index=True, width="stretch")

            st.subheader("✍️ Khud try karo")
            txt = st.text_input("Koi bhi message likho", key="try_tox")
            if txt.strip():
                pt = float(ml.predict_prob(models, "toxicity", [txt])[0])
                ps = float(ml.predict_prob(models, "spam", [txt])[0])
                st.write(f"Toxicity probability: **{pt:.0%}**"); st.progress(pt)
                st.write(f"Spam probability: **{ps:.0%}**"); st.progress(ps)
            with st.expander("📊 Model accuracy aur confusion matrix"):
                st.markdown("**Toxicity model**")
                model_card(models["toxicity"], ["Not toxic", "Toxic"])
                st.warning("Toxicity model ka training data chhota aur hand-written hai, isliye ye accuracy zyada dikh rahi hai "
                           "(train aur test sentences ek jaise templates se bane hain). Real chats par galtiyan hongi. "
                           "Behtar result ke liye Jigsaw jaisa bada dataset use karo.")
                st.markdown("**Spam model**  (SMS Spam Collection, 5,574 messages + Hinglish seed)")
                model_card(models["spam"], ["Not spam", "Spam"])

        # ---------------------------------------------------------- next word
        with n_tab:
            who = "poori chat" if selected_user == "Overall" else selected_user
            st.caption(f"Model: Trigram language model (stupid-backoff). Ye **{who}** ke messages se seekhta hai ki agla word kya aa sakta hai.")
            nw, nw_stats = get_next_word(selected_user, file_key, df)
            if nw is None:
                st.warning("Next-word model ke liye kam se kam 30 text messages chahiye.")
            else:
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("Training messages", f"{nw_stats['messages']:,}")
                a2.metric("Vocabulary", f"{nw_stats['vocab']:,}")
                a3.metric("Top-1 accuracy", f"{nw_stats['top1']:.1%}")
                a4.metric("Top-3 accuracy", f"{nw_stats['top3']:.1%}")
                st.caption(f"Accuracy hold-out 20% messages par ({nw_stats['tested_words']:,} words test kiye).")

                seed = st.text_input("Kuch type karo, model agla word suggest karega", value="", key="nw_seed",
                                     placeholder="jaise: kal, kya kar, main")
                preds = nw.predict(seed, k=6)
                if preds:
                    pdf = pd.DataFrame(preds, columns=["word", "probability"])
                    st.markdown("**Suggestions:** " + "  |  ".join(f"`{w}`" for w, _ in preds))
                    fig = px.bar(pdf, x="probability", y="word", orientation="h", color_discrete_sequence=[GREEN])
                    fig.update_layout(yaxis=dict(autorange="reversed"))
                    fig.update_xaxes(tickformat=".0%", title=None); fig.update_yaxes(title=None)
                    st.plotly_chart(style(fig, 300), width="stretch")

                st.subheader("🎲 Is style mein sentence generate karo")
                if st.button("Generate 3 sentences"):
                    for _ in range(3):
                        st.write("• " + nw.generate(seed))
                with st.expander("Ye kaise kaam karta hai?"):
                    st.markdown(
                        "- Har message ko words mein todte hain aur har 3 words ka combination ginte hain (trigram).\n"
                        "- Agla word = jo pichhle 2 words ke baad sabse zyada aaya.\n"
                        "- Agar wo combination kabhi nahi dikha to **backoff**: 1 word ke context par, phir overall frequency par (weight 0.4 se kam hota jaata hai).\n"
                        "- Isliye model har user ki **apni chatting style** seekhta hai."
                    )