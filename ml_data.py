"""
Chhota hand-written Hinglish/English seed data.
Bade public datasets (SMS spam, tweets) Hinglish nahi samajhte, isliye ye seed unke saath mila kar train hota hai.
Accuracy badhani ho to yahan aur examples add karo.
"""

# ------------------------------------------------------------ normal chat (ham / non-toxic)
NORMAL_CHAT = [
    "kal milte hain college mein", "aaj kya plan hai", "khana khaya kya", "ok theek hai",
    "meeting 5 baje hai", "assignment submit kar diya", "kal exam hai padh liya?", "reach ho gaya ghar",
    "bhai call kar na", "main 10 min mein aata hu", "notes bhej do please", "good morning everyone",
    "chalo phir baat karte hain", "project ka kaam kaha tak pahucha", "movie dekhne chalein?",
    "kya haal hai", "mai office ja raha hu", "photo bhej do yaar", "kal subah jaldi uthna hai",
    "birthday party kab hai", "see you tomorrow", "i will call you later", "where are you now",
    "did you finish the work", "let's meet at 6 pm", "send me the file please", "okay sounds good",
    "what time is the meeting", "i am reaching in 5 minutes", "thanks for the update",
    "paper kaisa gaya tera", "lunch pe chalein", "bus aa gayi hai", "network nahi aa raha",
    "wifi password kya hai", "kal holiday hai kya", "kitne baje aana hai", "teacher ne kya bola",
    "ghar pahunch ke message karna", "traffic bahut hai aaj", "cricket match dekha kal?",
    "aaj ka homework kya hai", "pls link bhejo notes ka", "ok done", "haan bilkul", "hmm dekhte hain",
]

# ------------------------------------------------------------ sentiment seed
POSITIVE_CHAT = [
    "bahut accha laga yaar 😍", "mast hai ekdum badhiya", "love you bro ❤️", "congratulations bhai party do 🎉",
    "tu best hai yaar", "kya baat hai zabardast", "maza aa gaya aaj to 😂", "thank you so much dost",
    "bohot khushi hui sunke", "great job well done 👏", "i am so happy today", "this is awesome 😍",
    "amazing work yaar", "so proud of you", "ekdum mast din tha", "lajawab performance tha tera",
    "wah kya khana tha maza aa gaya", "you are the best friend ever", "superb idea bhai", "bahut shukriya aapka 🙏",
    "dil khush kar diya", "full enjoy kiya aaj", "kamaal kar diya tune 🔥", "happy birthday dost 🎂 many happy returns",
    "mujhe ye bahut pasand aaya", "what a lovely day", "great news yaar bahut badhai", "sab kuch ekdum perfect raha",
]
NEGATIVE_CHAT = [
    "bahut bura laga yaar 😞", "mood kharab hai aaj", "i am so sad today 😢", "bohot dukh hua sunke",
    "kuch accha nahi lag raha", "ye bilkul bekaar tha", "worst day ever", "bahut gussa aa raha hai 😡",
    "tension ho rahi hai bahut", "main bahut thak gaya hu pareshan hu", "this is terrible", "i hate this so much",
    "sab kuch kharab ho gaya", "dil toot gaya mera 💔", "bahut bore ho raha hu", "kya bakwas movie thi time waste",
    "bahut takleef ho rahi hai", "i feel so lonely", "disappointed hu tujhse", "kitna ganda experience tha",
    "exam kharab gaya rona aa raha hai", "nobody cares about me", "bilkul maza nahi aaya", "ye service bahut ghatiya hai",
    "bahut irritate ho gaya hu", "such a horrible day", "udaas hu aaj", "sab log bahut pareshan kar rahe hain",
]

# ------------------------------------------------------------ toxicity seed (templates)
TOXIC_WORDS = ["idiot", "stupid", "moron", "loser", "dumb", "jerk", "trash", "pagal", "gadha", "kamina",
               "harami", "bewakoof", "nalayak", "ullu", "chutiya", "bakwas insaan", "pathetic fool"]
TOXIC_TEMPLATES = [
    "you are such a {w}", "tu bilkul {w} hai", "shut up you {w}", "{w} kahin ka", "kitna {w} hai tu",
    "tum log {w} ho", "get lost {w}", "nikal yaha se {w}", "i hate you {w}", "chup kar {w}",
    "tera dimag kharab hai {w}", "what a {w} you are", "{w} hai ye banda", "tu toh {w} nikla",
]
TOXIC_EXTRA = [
    "tujhse kuch nahi hoga nikamme", "teri aukat kya hai tu hai kya", "dafa ho ja yaha se", "go to hell",
    "nobody likes you, get lost", "you are worthless and useless", "tu hamesha fail hi hoga bewakoof",
    "shakal dekh apni", "tum jaise logon se nafrat hai mujhe", "stop talking you fool", "abe ullu ke pathe",
    "bakwas band kar warna dekh lunga", "you make me sick", "tu pagal hai kya, dimaag nahi hai tere paas",
]


# ------------------------------------------------------------ extra seeds (Hinglish positives/negatives, contrast data)
POSITIVE_CHAT += [
    "aap bahut achche insaan ho", "ye bahut accha hai yaar", "bahut badhiya kaam kiya", "shandar party thi kal",
    "kitna sundar hai ye 😍", "tera gaana mast tha", "bahut achha laga tumse milke", "ekdum jhakaas bhai",
    "tum sabse acche ho", "kya mast photo hai", "result achha aaya khushi hui", "aap great ho sir 🙏",
    "bahut pyaara hai ye", "tune toh dil jeet liya", "maza aa gaya yaar full on enjoy",
]
NEGATIVE_CHAT += [
    "ye bahut kharab hai", "kitna ganda khana tha", "bahut ghatiya movie thi", "mujhe bilkul pasand nahi aaya",
    "aaj ka din bekar tha", "bahut pareshan hu main", "mera mood off hai", "sab bekaar ho gaya yaar",
    "bahut bura hua tere saath", "rona aa raha hai mujhe 😭", "kuch bhi theek nahi chal raha",
]
NORMAL_CHAT += [
    "mujhse nahi hoga ye", "mujhe samajh nahi aaya", "ye kaam nahi ho raha", "i don't understand this",
    "pata nahi kya hoga", "abhi busy hu baad mein baat karte hain", "mujhe nahi pata yaar",
    "tu aa raha hai ya nahi", "tu kab aayega", "kal tu kahan tha", "tum log kab nikal rahe ho",
]
# normal links shared in chats (must NOT be flagged as spam)
SAFE_LINK_CHAT = [
    "check this https://youtube.com/watch?v=abc123", "ye video dekh https://youtu.be/xyz", "notes yaha hain https://drive.google.com/file/d/1abc",
    "my profile https://www.instagram.com/someone", "code yaha hai https://github.com/user/repo", "article padh le https://en.wikipedia.org/wiki/India",
    "join meeting https://zoom.us/j/123456", "linkedin pe connect karo https://www.linkedin.com/in/someone", "maps location https://maps.google.com/?q=xyz",
    "form bhar do https://forms.gle/abcd", "dekh ye reel https://www.instagram.com/reel/abc", "doc share kiya https://docs.google.com/document/d/1xyz",
]

# contrast data for toxicity: same sentence templates, but with nice words -> teaches the model that the WORD matters
NICE_WORDS = ["genius", "legend", "champion", "sweet", "star", "hero", "brilliant", "funny", "smart", "cutie",
              "superstar", "mast", "badhiya", "kamaal ka", "zabardast", "pyaara", "best insaan"]