import re
import pandas as pd

# Supports both formats of Android exports:
#   12/01/2023, 21:35 - Name: msg          (24-hour)
#   12/01/2023, 9:35 pm - Name: msg        (12-hour, newer phones; may use a narrow no-break space)
PATTERN = re.compile(
    r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})(?:[\s\u202f]?([AaPp][Mm]))?[\s\u202f]-[\s\u202f]'
)


def preprocess(data):
    data = data.replace('\u202f', ' ').replace('\ufeff', '')
    matches = list(PATTERN.finditer(data))
    if not matches:
        raise ValueError(
            "Chat format samajh nahi aaya. WhatsApp se 'Export chat (Without media)' "
            "wali .txt file (Android format) upload karo."
        )

    dates, raw_messages = [], []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(data)
        d, t, ampm = m.group(1), m.group(2), m.group(3)
        dates.append(f"{d} {t} {ampm}".strip() if ampm else f"{d} {t}")
        raw_messages.append(data[m.end():end])

    df = pd.DataFrame({'user_message': raw_messages, 'date_str': dates})
    df['date'] = pd.to_datetime(df['date_str'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date']).reset_index(drop=True)

    users, messages = [], []
    for message in df['user_message']:
        entry = re.match(r'^([^:\n]{1,60}?):\s(.*)$', message, re.S)
        if entry:
            users.append(entry.group(1))
            messages.append(entry.group(2).strip('\n'))
        else:
            users.append('group_notification')
            messages.append(message.strip('\n'))

    df['user'] = users
    df['message'] = messages
    df = df.drop(columns=['user_message', 'date_str'])

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['period'] = df['hour'].apply(lambda h: f"{h:02d}-{(h + 1) % 24:02d}")

    return df