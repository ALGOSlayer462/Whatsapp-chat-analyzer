import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import helper
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

try:
    df = load(uploaded_file.getvalue().decode("utf-8", errors="ignore"))
except ValueError as e:
    st.error(str(e))
    st.stop()

users = sorted(u for u in df["user"].unique() if u != "group_notification")
selected_user = st.sidebar.selectbox("👤 Analysis kiske liye", ["Overall"] + users)

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
tab_time, tab_act, tab_users, tab_words, tab_emoji = st.tabs(
    ["📅 Timeline", "🗓️ Activity", "👥 Users", "🔤 Words", "😀 Emoji"]
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