import os
from collections import Counter

import emoji
import pandas as pd
from urlextract import URLExtract
from wordcloud import WordCloud

extract = URLExtract()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def _load_stop_words():
    with open(os.path.join(BASE_DIR, 'stop_hinglish.txt'), encoding='utf-8') as f:
        return set(f.read().split())


STOP_WORDS = _load_stop_words()


def _is_media(msg):
    m = msg.strip().lower()
    return m == '<media omitted>' or (m.endswith(' omitted') and len(m.split()) <= 2)


def _is_deleted(msg):
    return msg.strip().lower() in ('this message was deleted', 'you deleted this message', 'null')


def _filter_user(selected_user, df):
    df = df[df['user'] != 'group_notification']
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df


def _text_df(selected_user, df):
    """Only real text messages (no media / deleted / notifications)."""
    df = _filter_user(selected_user, df)
    mask = ~(df['message'].apply(_is_media) | df['message'].apply(_is_deleted))
    return df[mask]


def fetch_stats(selected_user, df):
    df = _filter_user(selected_user, df)
    num_messages = df.shape[0]

    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media_messages = int(df['message'].apply(_is_media).sum())

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)


def most_busy_users(df):
    df = df[df['user'] != 'group_notification']
    counts = df['user'].value_counts()
    x = counts.head()
    percent_df = (counts / df.shape[0] * 100).round(2).reset_index()
    percent_df.columns = ['name', 'percent']
    return x, percent_df


def create_wordcloud(selected_user, df):
    temp = _text_df(selected_user, df)
    words = [w for msg in temp['message'] for w in msg.lower().split() if w not in STOP_WORDS]
    if not words:
        return None
    wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
    return wc.generate(" ".join(words))


def most_common_words(selected_user, df):
    temp = _text_df(selected_user, df)
    words = [w for msg in temp['message'] for w in msg.lower().split() if w not in STOP_WORDS]
    out = pd.DataFrame(Counter(words).most_common(20), columns=['word', 'count'])
    return out


def emoji_helper(selected_user, df):
    df = _filter_user(selected_user, df)
    emojis = []
    for message in df['message']:
        emojis.extend(e['emoji'] for e in emoji.emoji_list(message))
    return pd.DataFrame(Counter(emojis).most_common(), columns=['emoji', 'count'])


def monthly_timeline(selected_user, df):
    df = _filter_user(selected_user, df)
    timeline = (df.groupby(['year', 'month_num', 'month']).size()
                  .reset_index(name='message'))
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)
    return timeline


def daily_timeline(selected_user, df):
    df = _filter_user(selected_user, df)
    return df.groupby('only_date').size().reset_index(name='message')


def week_activity_map(selected_user, df):
    df = _filter_user(selected_user, df)
    return df['day_name'].value_counts()


def month_activity_map(selected_user, df):
    df = _filter_user(selected_user, df)
    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    df = _filter_user(selected_user, df)
    heat = df.pivot_table(index='day_name', columns='period',
                          values='message', aggfunc='count').fillna(0)
    heat = heat.reindex([d for d in DAY_ORDER if d in heat.index])
    return heat[sorted(heat.columns)]