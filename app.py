"""
app.py
------
Teacher's Dashboard for the Student Performance Prediction System (premium UI).

A teacher enters each student's details after the course, gets a predicted
final grade (G3), sees a performance category, and the app generates
PERSONALIZED IMPROVEMENT SUGGESTIONS for the student (based on the same
inputs the model used). Every record is saved to a SQLite database and can
be downloaded as a CSV report for the whole class.

Run:  streamlit run app.py
"""

import html
import sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

DB_PATH = "predictions.db"
MODEL_PATH = "model.pkl"
META_PATH = "model_meta.pkl"

# ------------------------------------------------------------------
# Load the trained model once (cached so Streamlit doesn't reload it
# on every interaction)
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model, meta


# ------------------------------------------------------------------
# Database helpers (SQLite - a simple file-based database, no server
# needed, perfect for small beginner projects)
# ------------------------------------------------------------------
COLUMNS = ["student_name", "age", "sex", "address", "internet", "romantic",
           "studytime", "failures", "absences", "medu", "fedu", "g1", "g2",
           "predicted_g3", "category", "suggestions", "created_at"]


def init_db():
    """Create the predictions table (old tables are migrated automatically)."""
    conn = sqlite3.connect(DB_PATH)

    # Check if this is an old table without the new columns
    existing = [row[1] for row in conn.execute("PRAGMA table_info(predictions)")]
    if existing and ("student_name" not in existing or "suggestions" not in existing):
        # Old database from version 1 -> migrate data into the new schema
        conn.execute("ALTER TABLE predictions RENAME TO predictions_old")
        _create_table(conn)
        old_cols = [r[1] for r in conn.execute("PRAGMA table_info(predictions_old)")]
        common = [c for c in COLUMNS if c in old_cols]
        cols = ", ".join(common)
        conn.execute(f"INSERT INTO predictions ({cols}) SELECT {cols} FROM predictions_old")
        conn.execute("DROP TABLE predictions_old")
    elif not existing:
        _create_table(conn)

    conn.commit()
    conn.close()


def _create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT,
            age INTEGER,
            sex TEXT,
            address TEXT,
            internet TEXT,
            romantic TEXT,
            studytime INTEGER,
            failures INTEGER,
            absences INTEGER,
            medu INTEGER,
            fedu INTEGER,
            g1 INTEGER,
            g2 INTEGER,
            predicted_g3 REAL,
            category TEXT,
            suggestions TEXT,
            created_at TEXT
        )
    """)


def save_prediction(record: dict):
    """Insert one prediction record into the database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO predictions
           (student_name, age, sex, address, internet, romantic, studytime,
            failures, absences, medu, fedu, g1, g2, predicted_g3, category,
            suggestions, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            record["student_name"], record["age"], record["sex"],
            record["address"], record["internet"], record["romantic"],
            record["studytime"], record["failures"], record["absences"],
            record["Medu"], record["Fedu"], record["G1"], record["G2"],
            record["predicted_g3"], record["category"], record["suggestions"],
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()


def load_history():
    """Read all past predictions from the database."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    columns = [d[0] for d in conn.execute("SELECT * FROM predictions LIMIT 0").description]
    conn.close()
    return pd.DataFrame(rows, columns=columns)


def grade_category(grade: float) -> str:
    """Convert a numeric grade (0-20) into a simple category."""
    if grade >= 16:
        return "Excellent"
    elif grade >= 13:
        return "Good"
    elif grade >= 10:
        return "Average"
    elif grade >= 7:
        return "Below Average"
    return "Poor"


# ------------------------------------------------------------------
# Personalized improvement suggestions
#
# Rule-based logic on the SAME inputs the model saw, so every suggestion
# maps to a real input feature and stays fully explainable.
# Each tip is (level, title, detail); level = risk | watch | good
# ------------------------------------------------------------------
def generate_suggestions(record: dict, prediction: float) -> list:
    tips = []

    if record["G2"] >= record["G1"] + 2:
        tips.append(("good", "Improving trend",
                     f"G2 ({record['G2']}) is well above G1 ({record['G1']}). Keep the current strategy."))
    elif record["G2"] < record["G1"] - 2:
        tips.append(("risk", "Grades are dropping",
                     f"G1 {record['G1']} to G2 {record['G2']}. Revise weak topics before the final exam."))

    if record["studytime"] <= 1:
        tips.append(("watch", "Study time is very low",
                     "Under 2 hours a week. Build a fixed daily schedule of at least 2-3 hours."))
    if record["absences"] > 10:
        tips.append(("watch", "High absences",
                     f"{record['absences']} days missed. Regular attendance directly affects grades."))

    if record["failures"] > 0:
        tips.append(("watch", "Past class failures",
                     f"{record['failures']} on record. Focus on fundamentals and ask teachers for help early."))

    if record["internet"] == "yes" and record["studytime"] <= 2:
        tips.append(("watch", "Possible distraction",
                     "Internet access with low study time. Try focused sessions with notifications off."))
    if record["romantic"] == "yes" and record["studytime"] <= 2:
        tips.append(("watch", "Balance personal time",
                     "Make sure personal commitments are not cutting into study hours."))

    if record["Medu"] <= 1 or record["Fedu"] <= 1:
        tips.append(("watch", "Limited home academic support",
                     "Use free online resources (NPTEL, recorded lectures) for self-study."))

    if prediction < 10:
        tips.append(("risk", "Below passing level",
                     f"Predicted {prediction:.1f}/20. This student needs mentoring and a written study plan."))
    elif prediction >= 16:
        tips.append(("good", "Excellent outlook",
                     f"Predicted {prediction:.1f}/20. Encourage the student to stay consistent."))

    if not tips:
        tips.append(("good", "Well balanced",
                     "Small, consistent increases in study time can lift the grade further."))
    return tips


CATEGORY_STYLE = {   # colour per performance category
    "Excellent": "#7FB59A", "Good": "#9BC9B0", "Average": "#D9A24A",
    "Below Average": "#D98A4A", "Poor": "#C0646E",
}
CATEGORY_ORDER = ["Excellent", "Good", "Average", "Below Average", "Poor"]


# ------------------------------------------------------------------
# Page setup + theme
# ------------------------------------------------------------------
st.set_page_config(page_title="Student Performance | Teacher's Dashboard",
                   page_icon="🎓", layout="wide",
                   initial_sidebar_state="expanded")

css_file = Path(__file__).parent / "styles.css"
st.markdown(f"<style>{css_file.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)

model, meta = load_artifacts()
init_db()


def page_header(title: str, subtitle: str):
    st.markdown(
        f"<div class='page-head'><h1>{title}</h1><p>{subtitle}</p>"
        "<div class='rule'></div></div>", unsafe_allow_html=True)


def section(title: str, note: str = ""):
    st.markdown(f"<div class='sec'>{title}</div>"
                + (f"<div class='sec-note'>{note}</div>" if note else ""),
                unsafe_allow_html=True)


# ----- Sidebar -----
with st.sidebar:
    st.markdown(
        "<div class='brand'>Student<br>Performance"
        "<small>Teacher's dashboard</small></div><div class='brand-rule'></div>",
        unsafe_allow_html=True)
    page = st.radio("Navigate", ["Predict", "Class report", "Model"],
                    label_visibility="collapsed")
    st.markdown("<div class='brand-rule' style='margin-top:2rem'></div>"
                "<div style='color:#9DB1A7;font-size:.8rem;font-weight:300'>"
                "Python, scikit-learn and Streamlit</div>", unsafe_allow_html=True)


# ==================================================================
# PAGE 1 - Predict
# ==================================================================
if page == "Predict":
    page_header("Predict a final grade",
                "Enter a student's details to forecast the final grade out of 20 "
                "and receive tailored guidance for the student.")

    with st.form("predict_form", border=False):
        section("Student", "Who are you assessing?")
        student_name = st.text_input("Student name", placeholder="e.g. Rahul Sharma")

        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 15, 30, 17)
            sex = st.selectbox("Sex", ["F", "M"])
        with c2:
            address = st.selectbox("Address", ["U", "R"], help="U = Urban, R = Rural")
            internet = st.selectbox("Internet access at home", ["yes", "no"])
        with c3:
            romantic = st.selectbox("In a relationship", ["no", "yes"])
            studytime = st.selectbox("Weekly study time", [1, 2, 3, 4],
                                     help="1: under 2h | 2: 2-5h | 3: 5-10h | 4: over 10h")

        section("Academic record")
        c4, c5, c6 = st.columns(3)
        with c4:
            failures = st.number_input("Past class failures", 0, 4, 0)
            absences = st.number_input("Absences", 0, 93, 0)
        with c5:
            g1 = st.number_input("G1 (first period grade)", 0, 20, 12)
            g2 = st.number_input("G2 (second period grade)", 0, 20, 12)
        with c6:
            medu = st.number_input("Mother's education (0-4)", 0, 4, 2,
                                   help="Type a level: 0 none, 1 primary, 2 5th-9th grade, 3 secondary, 4 higher education")
            fedu = st.number_input("Father's education (0-4)", 0, 4, 2,
                                   help="Type a level: 0 none, 1 primary, 2 5th-9th grade, 3 secondary, 4 higher education")

        submitted = st.form_submit_button("Predict performance", type="primary")

    if submitted:
        if not student_name.strip():
            st.error("Enter the student's name to continue.")
            st.stop()

        name = student_name.strip()
        input_df = pd.DataFrame([{
            "age": age, "sex": sex, "address": address, "internet": internet,
            "romantic": romantic, "studytime": studytime, "failures": failures,
            "absences": absences, "Medu": medu, "Fedu": fedu, "G1": g1, "G2": g2,
        }])
        prediction = min(max(float(model.predict(input_df)[0]), 0.0), 20.0)
        category = grade_category(prediction)
        color = CATEGORY_STYLE[category]

        record = {
            "student_name": name, "age": age, "sex": sex, "address": address,
            "internet": internet, "romantic": romantic, "studytime": studytime,
            "failures": failures, "absences": absences, "Medu": medu,
            "Fedu": fedu, "G1": g1, "G2": g2,
        }
        tips = generate_suggestions(record, prediction)

        # ----- Result plate -----
        pos = prediction / 20 * 100
        st.markdown(f"""
        <div class='plate'>
          <div class='who'>Predicted final grade for <b>{html.escape(name)}</b></div>
          <div class='score'>{prediction:.1f}<span>/ 20</span></div>
          <div class='verdict' style='color:{color}'>{category}</div>
          <div class='gauge'><i style='left:calc({pos:.1f}% - 1px)'></i></div>
          <div class='gauge-scale'><span>0</span><span>7</span><span>13</span><span>20</span></div>
        </div>""", unsafe_allow_html=True)

        # ----- Message + suggestions -----
        if prediction < 10:
            st.markdown("<div class='banner risk'><b>Needs immediate support.</b> "
                        "Review the areas below with the student.</div>", unsafe_allow_html=True)
        elif prediction >= 16:
            st.markdown("<div class='banner good'><b>Strong performance.</b> "
                        "Keep encouraging this student.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='banner watch'><b>Room to improve.</b> "
                        "Focus on the areas below.</div>", unsafe_allow_html=True)

        section(f"Guidance for {html.escape(name)}")
        st.markdown("".join(
            f"<div class='tip {lvl}'><div class='bar'></div><div>"
            f"<span class='t'>{html.escape(t)}</span>"
            f"<span class='d'>{html.escape(d)}</span></div></div>"
            for lvl, t, d in tips), unsafe_allow_html=True)

        save_prediction({
            **record, "predicted_g3": round(prediction, 2), "category": category,
            "suggestions": " | ".join(f"{t}: {d}" for _, t, d in tips),
        })
        if age > 22:
            st.caption("Note: the model was trained on students aged 15-22, so predictions "
                       "for older students are less reliable.")
        st.caption("Saved to the class report.")


# ==================================================================
# PAGE 2 - Class report
# ==================================================================
elif page == "Class report":
    page_header("Class report",
                "Every prediction made so far, with a summary of how the class is doing.")

    history = load_history()
    if history.empty:
        st.info("No records yet. Make a prediction on the Predict page to start the report.")
    else:
        total = len(history)
        avg = history["predicted_g3"].mean()
        at_risk = int((history["predicted_g3"] < 10).sum())

        st.markdown(f"""
        <div class='stats'>
          <div class='stat'><div class='n'>{total}</div><div class='l'>Predictions recorded</div></div>
          <div class='stat'><div class='n'>{avg:.1f}</div><div class='l'>Average predicted grade (of 20)</div></div>
          <div class='stat'><div class='n'>{at_risk}</div><div class='l'>Students below passing level</div></div>
        </div>""", unsafe_allow_html=True)

        section("Grade distribution", "Number of predictions in each performance category.")
        counts = (history["category"].value_counts()
                  .reindex(CATEGORY_ORDER, fill_value=0).rename_axis("Category")
                  .reset_index(name="Students"))
        counts["Category"] = pd.Categorical(counts["Category"], CATEGORY_ORDER, ordered=True)
        st.bar_chart(counts, x="Category", y="Students", color="#C2A15F",
                     height=260, sort=False)

        section("All records")
        table = history.drop(columns=["id"]).rename(columns={
            "student_name": "Student", "predicted_g3": "Predicted grade",
            "category": "Category", "created_at": "Date", "suggestions": "Guidance"})
        st.dataframe(table, width="stretch", hide_index=True,
                     column_config={"Predicted grade": st.column_config.NumberColumn(format="%.1f")})

        d1, d2, _ = st.columns([1, 1, 2])
        with d1:
            st.download_button("Download CSV report",
                               data=table.to_csv(index=False).encode("utf-8"),
                               file_name="class_performance_report.csv", mime="text/csv")
        with d2:
            confirm = st.checkbox("Confirm deleting all records")
            if st.button("Clear all records", disabled=not confirm):
                conn = sqlite3.connect(DB_PATH)
                conn.execute("DELETE FROM predictions")
                conn.commit()
                conn.close()
                st.rerun()


# ==================================================================
# PAGE 3 - Model
# ==================================================================
else:
    page_header("About the model",
                "How the grade forecast is made and how accurate it is on students it has never seen.")

    rows = ""
    for name, m in meta["metrics"].items():
        best = " class='best'" if name == meta["model_name"] else ""
        tag = " (selected)" if name == meta["model_name"] else ""
        rows += (f"<tr{best}><td class='name'>{name}{tag}</td><td>{m['MAE']:.3f}</td>"
                 f"<td>{m['RMSE']:.3f}</td><td>{m['R2']:.3f}</td></tr>")
    st.markdown(f"""
    <table class='mtable'>
      <tr><th>Algorithm</th><th>MAE</th><th>RMSE</th><th>R&sup2;</th></tr>{rows}
    </table>""", unsafe_allow_html=True)

    best_m = meta["metrics"][meta["model_name"]]
    st.markdown(f"""
    <div class='note'>
      The selected {meta['model_name']} model is off by about <b>{best_m['MAE']:.1f} grade points</b>
      on average and explains roughly <b>{best_m['R2']*100:.0f}%</b> of the variation in final grades.
      Predictions use the first two period grades (G1, G2), so they are valid only once those
      grades are known, for example during the school year.
    </div>""", unsafe_allow_html=True)
