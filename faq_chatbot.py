"""
FAQ Chatbot — CodVeda Level 3, Task 2
======================================
Topic : University Student Portal FAQs
NLP   : NLTK — tokenize, remove stopwords, lemmatize + WordNet synonym expansion
Match : TF-IDF vectorisation + Cosine Similarity
UI    : Terminal chat loop
"""

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
#  FAQ DATA  —  15 university portal questions with domain keyword hints
# ─────────────────────────────────────────────────────────────────────────────
FAQS = [
    {
        "question": "How do I register for courses?",
        "keywords": ["register", "enroll", "enrol", "sign up", "course", "class",
                     "subject", "add course", "choose subject"],
        "answer": (
            "Log in to the student portal, go to 'Course Registration', select your "
            "desired courses, and click 'Submit Registration' before the deadline."
        ),
    },
    {
        "question": "When is the tuition fee payment deadline?",
        "keywords": ["fee", "pay", "payment", "tuition", "school fees", "deadline",
                     "when pay", "how much", "billing"],
        "answer": (
            "Tuition fees must be paid by the end of the first two weeks of each semester. "
            "Late payment attracts a penalty fee."
        ),
    },
    {
        "question": "How can I check my exam results?",
        "keywords": ["result", "grade", "score", "mark", "exam result", "check result",
                     "see result", "view grade", "academic record"],
        "answer": (
            "Your results are published on the student portal under 'Academic Records > Results'. "
            "You will also receive an email notification when results are released."
        ),
    },
    {
        "question": "What is the process for deferring a semester?",
        "keywords": ["defer", "deferral", "postpone", "suspend", "take a break",
                     "leave of absence", "pause", "withdraw temporarily"],
        "answer": (
            "Submit a deferral request form to the Academic Affairs office at least 4 weeks "
            "before the semester begins. Attach supporting documents if deferring for medical reasons."
        ),
    },
    {
        "question": "How do I get my student ID card?",
        "keywords": ["id", "identity", "card", "student id", "id card",
                     "identification", "collect id", "where id"],
        "answer": (
            "New students collect their ID cards from the Student Affairs office after completing "
            "registration and uploading a passport photo to the portal."
        ),
    },
    {
        "question": "Can I change my course of study?",
        "keywords": ["change", "switch", "transfer", "programme", "department",
                     "faculty", "change course", "different programme"],
        "answer": (
            "Yes. You may apply for a change of programme in your first year by submitting a "
            "Change of Course form to the Academic Registry. Approval depends on available slots and CGPA."
        ),
    },
    {
        "question": "What is the minimum CGPA to avoid probation?",
        "keywords": ["cgpa", "gpa", "probation", "minimum", "withdrawal", "grade point",
                     "academic standing", "stay enrolled", "enrolled", "enrolment",
                     "remain", "keep studying", "stay in school", "what gpa",
                     "gpa needed", "grade needed", "minimum grade", "pass requirement"],
        "answer": (
            "Students must maintain a CGPA of at least 1.50. Falling below this for two consecutive "
            "semesters may result in withdrawal from the university."
        ),
    },
    {
        "question": "How do I apply for a scholarship?",
        "keywords": ["scholarship", "bursary", "financial aid", "award", "grant",
                     "funding", "sponsor", "free tuition"],
        "answer": (
            "Visit the Scholarships & Bursaries page on the portal, review available scholarships, "
            "and submit the required documents before the application deadline."
        ),
    },
    {
        "question": "How can I access the university library online?",
        "keywords": ["library", "ebook", "journal", "research", "book", "online library",
                     "past paper", "e-library", "read online"],
        "answer": (
            "Use your student ID and portal password to log in at library.university.edu. "
            "You will have access to e-journals, e-books, and past exam papers."
        ),
    },
    {
        "question": "What should I do if I miss an exam?",
        "keywords": ["miss", "skip", "absent", "missed exam", "supplementary",
                     "did not attend", "skipped exam", "could not write"],
        "answer": (
            "Notify your department immediately. You may apply for a supplementary exam by "
            "submitting a Missed Exam form with valid supporting evidence within 48 hours."
        ),
    },
    {
        "question": "How do I reset my portal password?",
        "keywords": ["password", "reset", "forgot", "login", "log in",
                     "locked out", "access portal", "cant login"],
        "answer": (
            "Click 'Forgot Password' on the login page and enter your student email. "
            "A reset link will be sent to your registered email address."
        ),
    },
    {
        "question": "Where can I find the academic calendar?",
        "keywords": ["calendar", "schedule", "timetable", "semester dates",
                     "holiday", "academic dates", "semester start", "exam period"],
        "answer": (
            "The academic calendar is available on the university's official website under "
            "'Academics > Calendar', and is also pinned on the student portal dashboard."
        ),
    },
    {
        "question": "How do I request an official transcript?",
        "keywords": ["transcript", "official document", "academic record",
                     "certificate", "letter", "request document", "send transcript"],
        "answer": (
            "Log in to the portal, go to 'Documents > Transcript Request', fill in the "
            "destination details, and pay the processing fee. Allow 5–7 working days."
        ),
    },
    {
        "question": "Is there a hostel on campus?",
        "keywords": ["hostel", "accommodation", "dorm", "dormitory", "room",
                     "lodge", "sleep", "stay", "where sleep", "on-campus housing"],
        "answer": (
            "Yes. Campus accommodation is available for eligible students. Apply via the portal "
            "under 'Accommodation' during the designated application window each session."
        ),
    },
    {
        "question": "Who do I contact for mental health support?",
        "keywords": ["mental health", "counselling", "counseling", "depressed", "anxious",
                     "anxiety", "stress", "overwhelmed", "struggling", "counsellor",
                     "sad", "depression", "feeling down", "need help", "emotional support",
                     "therapy", "psychologist", "wellbeing", "wellness"],
        "answer": (
            "The Student Wellness Centre offers free and confidential counselling. Book an "
            "appointment via the portal or walk in during open hours (Mon–Fri, 9am–4pm)."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
#  NLP PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def get_wordnet_synonyms(word: str) -> set:
    """Collect all lemma names for every WordNet synset of a word."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " ").lower())
    return synonyms


def preprocess(text: str, expand_synonyms: bool = False) -> str:
    """
    NLP pipeline
    ─────────────
    1. Lowercase
    2. Strip punctuation  (keep letters, digits, spaces)
    3. Tokenize           (NLTK word_tokenize)
    4. Remove stopwords   (NLTK English stopword list)
    5. Lemmatize          (WordNetLemmatizer)
    6. Expand synonyms    (optional — WordNet synsets)
    """
    text   = text.lower()
    text   = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    if expand_synonyms:
        expanded = list(tokens)
        for t in tokens:
            expanded.extend(get_wordnet_synonyms(t))
        tokens = expanded

    return " ".join(tokens)


def build_faq_document(faq: dict) -> str:
    """
    Create a rich text document per FAQ for TF-IDF:
      preprocessed question  +  domain keyword hints
    """
    base = preprocess(faq["question"], expand_synonyms=True)
    kws  = " ".join(lemmatizer.lemmatize(k.lower()) for k in faq.get("keywords", []))
    return f"{base} {kws}"


# Build TF-IDF index over all FAQ documents
faq_questions_raw = [f["question"] for f in FAQS]
faq_documents     = [build_faq_document(f) for f in FAQS]

vectorizer   = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(faq_documents)


# ─────────────────────────────────────────────────────────────────────────────
#  MATCHING
# ─────────────────────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.10   # minimum cosine similarity to return an answer


def find_best_match(user_query: str):
    """
    1. Preprocess the user query (with synonym expansion).
    2. Vectorise with the fitted TF-IDF model.
    3. Compute cosine similarity against all FAQ vectors.
    4. Return the best-matching question and answer if above threshold.
    """
    clean_query  = preprocess(user_query, expand_synonyms=True)
    query_vector = vectorizer.transform([clean_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    best_idx   = int(similarities.argmax())
    best_score = float(similarities[best_idx])

    if best_score < CONFIDENCE_THRESHOLD:
        return None, None, best_score

    return faq_questions_raw[best_idx], FAQS[best_idx]["answer"], best_score


# ─────────────────────────────────────────────────────────────────────────────
#  TERMINAL CHAT UI
# ─────────────────────────────────────────────────────────────────────────────
DIVIDER  = "─" * 55
BOT_NAME = "UniBot"

BANNER = f"""
{DIVIDER}
  🎓  {BOT_NAME} — University Student Portal FAQ
{DIVIDER}
  Ask me anything about registration, fees, exams,
  library access, accommodation, and more.

  Type  'help'   to see all available topics.
  Type  'quit'   or  'exit'  to leave.
{DIVIDER}
"""

HELP_TEXT = """
📚 Topics I can help with:
  • Course registration          • Tuition fee deadlines
  • Exam results & missed exams  • Student ID card
  • Scholarships & bursaries     • Password reset
  • Change of programme          • Transcript requests
  • Library online access        • Academic calendar
  • Campus accommodation         • Mental health support
  • Deferral / probation rules
"""


def chat():
    print(BANNER)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{BOT_NAME}: Goodbye! Take care. 👋\n")
            break

        if not user_input:
            continue

        lower = user_input.lower()

        if lower in ("quit", "exit"):
            print(f"\n{BOT_NAME}: Goodbye! Take care. 👋\n")
            break

        if lower == "help":
            print(HELP_TEXT)
            continue

        matched_q, answer, score = find_best_match(user_input)

        print()
        if answer:
            print(f"{BOT_NAME}: {answer}")
            print(f'  [Matched: "{matched_q}"  |  confidence: {score:.2f}]')
        else:
            print(f"{BOT_NAME}: Sorry, I couldn't find a good match for that question.")
            print( "  Try rephrasing, or type 'help' to see available topics.")
        print()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    chat()