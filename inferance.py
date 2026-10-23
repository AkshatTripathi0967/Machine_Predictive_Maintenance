import json
import joblib
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

st.set_page_config(page_title="Plant maintenance control", page_icon="🛠️", layout="wide")

INK, STEEL, OK, WATCH, ALERT = "#1B2A38", "#2F5D7C", "#DDE5EA", "#E8A317", "#C8372D"
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, .stApp, button, input, textarea { font-family: 'IBM Plex Sans', sans-serif; font-variant-numeric: tabular-nums; }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
.block-container { padding-top: 1.4rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: #1B2A38; }
[data-testid="stSidebar"] * { color: #E6ECF1; }
[data-testid="stSidebar"] [data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 600; }
.hdr { display: flex; justify-content: space-between; align-items: center; background: #1B2A38; color: #fff;
       padding: 18px 24px; border-radius: 4px; margin-bottom: 18px; border-bottom: 4px solid #2F5D7C; }
.hdr .ttl { font-size: 1.5rem; font-weight: 600; letter-spacing: .01em; }
.hdr .sub { color: #AFC0CD; font-size: .95rem; margin-top: 2px; }
.pill { padding: 8px 16px; border-radius: 3px; font-weight: 600; font-size: .95rem; }
.pill.bad { background: #C8372D; color: #fff; } .pill.good { background: #2E8B6E; color: #fff; }
.kpi { background: #fff; border: 1px solid #CBD3DA; border-radius: 4px; padding: 14px 16px; }
.kpi .kl { color: #5B6B79; font-size: .85rem; } .kpi .kv { font-size: 2rem; font-weight: 600; line-height: 1.2; }
.kpi .kn { color: #5B6B79; font-size: .8rem; }
.panel { background: #fff; border: 1px solid #CBD3DA; border-radius: 4px; padding: 16px; }
.panel h4 { margin: 0 0 10px 0; font-size: 1rem; font-weight: 600; }
.board { display: grid; grid-template-columns: repeat(10, 1fr); gap: 5px; }
.tile { aspect-ratio: 1.25; display: flex; align-items: center; justify-content: center; border-radius: 3px;
        font-size: .85rem; font-weight: 500; }
.legend { display: flex; gap: 16px; margin-top: 12px; font-size: .82rem; color: #5B6B79; }
.legend i { display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }
.gauge { position: relative; height: 22px; background: #DDE5EA; border-radius: 3px; margin: 10px 0 4px 0; }
.gauge .fill { height: 100%; border-radius: 3px; }
.gauge .mark { position: absolute; top: -5px; width: 3px; height: 32px; background: #1B2A38; }
.stTabs [data-baseweb="tab"] { font-weight: 500; padding: 10px 18px; }
.stTabs [aria-selected="true"] { color: #2F5D7C; }
</style>""", unsafe_allow_html=True)


def kpi(label, value, note="", tone="#2F5D7C"):
    return (f'<div class="kpi" style="border-left:5px solid {tone}"><div class="kl">{label}</div>'
            f'<div class="kv">{value}</div><div class="kn">{note}&nbsp;</div></div>')


MODEL_PATH = r"C:\Machine Learning Projects\predictive maintenance\random_forest_model.pkl"  # your own trained model


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_holdout():
    return pd.read_csv("holdout.csv.gz")


model = load_model()
FEATURES = list(model.feature_names_in_)
hold = load_holdout()
if [c for c in FEATURES if c not in hold.columns]:
    st.error("holdout.csv.gz is missing columns your model expects: "
             + ", ".join(c for c in FEATURES if c not in hold.columns))
    st.stop()


@st.cache_data(show_spinner="Scoring held-out data with your model (first run only)...")
def score():
    return model.predict_proba(hold[FEATURES])[:, 1]


hold["risk"] = score()

# ---------------- sidebar ----------------
st.sidebar.title("Controls")
truth_mode = st.sidebar.radio(
    "Compare alerts against",
    ["Failure exactly 6h ahead", "Any failure in the next 6h"])
if truth_mode.startswith("Failure exactly"):
    hold["truth"] = hold.groupby("machineID")["failure"].shift(-6)
else:
    hold["truth"] = hold["future_failure"]
hold = hold.dropna(subset=["truth"])
hold["step"] = hold.groupby("machineID").cumcount()
thr = st.sidebar.slider("Alert threshold", 0.01, 0.95, 0.085, 0.005,
                        help="Lower catches more failures but raises more false alarms.")
step = st.sidebar.slider("Replay hour", 0, int(hold["step"].max()), 0,
                         help="Moves through the held-out test period.")
y, pred = hold["truth"], (hold["risk"] >= thr)
st.sidebar.markdown("#### Test-period accuracy")
c1, c2 = st.sidebar.columns(2)
c1.metric("Recall", f"{recall_score(y, pred):.1%}")
c2.metric("Precision", f"{precision_score(y, pred, zero_division=0):.1%}")
c1.metric("F1", f"{f1_score(y, pred):.2f}")
c2.metric("PR-AUC", f"{average_precision_score(y, hold['risk']):.2f}")
st.sidebar.caption(f"Caught {int(((y == 1) & pred).sum())} rows, missed {int(((y == 1) & ~pred).sum())}, "
                   f"false alarms {int(((y == 0) & pred).sum())}.")

snap = hold[hold["step"] == step].sort_values("machineID")
n_alert = int((snap["risk"] >= thr).sum())
st.markdown(
    f'<div class="hdr"><div><div class="ttl">Plant maintenance control</div>'
    f'<div class="sub">Failure risk for the next 6 hours, for each of {len(snap)} machines</div></div>'
    f'<div class="pill {"bad" if n_alert else "good"}">'
    f'{f"{n_alert} machines need inspection" if n_alert else "All machines normal"}</div></div>',
    unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Fleet monitor", "Machine detail", "What-if prediction", "Batch upload"])

# ---------------- tab 1: fleet ----------------
with tab1:
    n_watch = int(((snap["risk"] >= thr / 2) & (snap["risk"] < thr)).sum())
    k = st.columns(4)
    k[0].markdown(kpi("Machines in alert", n_alert, f"risk at or above {thr:.1%}", ALERT if n_alert else OK), unsafe_allow_html=True)
    k[1].markdown(kpi("On watch", n_watch, "above half the threshold", WATCH), unsafe_allow_html=True)
    k[2].markdown(kpi("Highest risk", f"{snap['risk'].max():.0%}", f"machine {int(snap.loc[snap['risk'].idxmax(), 'machineID'])}"), unsafe_allow_html=True)
    k[3].markdown(kpi("Failures that followed", int(snap["truth"].sum()), "ground truth from the replay"), unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([5, 6])
    with left:
        tiles = ""
        for m, r in zip(snap["machineID"], snap["risk"]):
            bg, fg = (ALERT, "#fff") if r >= thr else (WATCH, INK) if r >= thr / 2 else (OK, INK)
            tiles += f'<div class="tile" title="Machine {m}: {r:.1%} risk" style="background:{bg};color:{fg}">{m}</div>'
        st.markdown(
            f'<div class="panel"><h4>Plant floor</h4><div class="board">{tiles}</div><div class="legend">'
            f'<span><i style="background:{OK}"></i>Normal</span><span><i style="background:{WATCH}"></i>Watch</span>'
            f'<span><i style="background:{ALERT}"></i>Alert</span></div></div>', unsafe_allow_html=True)
    with right:
        st.markdown("##### Inspection priority")
        show = snap.sort_values("risk", ascending=False).head(12)[
            ["machineID", "risk", "truth", "Total_hours_from_last_maintenance", "errors_till_now", "vibration_6h_mean"]]
        show = show.rename(columns={"machineID": "Machine", "risk": "Failure risk", "truth": "Failure followed?",
                                    "Total_hours_from_last_maintenance": "Hours since maintenance",
                                    "errors_till_now": "Errors to date", "vibration_6h_mean": "Vibration (6h avg)"})
        show["Failure followed?"] = show["Failure followed?"].map({1.0: "Yes", 0.0: "No"})
        st.dataframe(show, hide_index=True, use_container_width=True, height=395,
                     column_config={"Failure risk": st.column_config.ProgressColumn(
                         "Failure risk", min_value=0.0, max_value=1.0, format="%.2f")})

# ---------------- tab 2: machine detail ----------------
with tab2:
    mid = st.selectbox("Machine", sorted(hold["machineID"].unique()))
    m = hold[hold["machineID"] == mid].reset_index(drop=True)
    fails = m.loc[m["failure"] == 1, "hour"].unique()
    k = st.columns(3)
    k[0].markdown(kpi("Hours in alert", int((m["risk"] >= thr).sum()), f"of {len(m)} hours"), unsafe_allow_html=True)
    k[1].markdown(kpi("Actual failures", len(fails), "in the test period", ALERT if len(fails) else OK), unsafe_allow_html=True)
    k[2].markdown(kpi("Peak risk", f"{m['risk'].max():.0%}"), unsafe_allow_html=True)
    st.write("")
    x = alt.X("hour:Q", title="Hour", axis=alt.Axis(grid=False))
    line = alt.Chart(m).mark_line(color=STEEL, strokeWidth=1.6).encode(
        x=x, y=alt.Y("risk:Q", title="Failure risk", scale=alt.Scale(domain=[0, 1])))
    rule = alt.Chart(pd.DataFrame({"t": [thr]})).mark_rule(color=ALERT, strokeDash=[6, 4]).encode(y="t:Q")
    fl = alt.Chart(pd.DataFrame({"hour": fails})).mark_rule(color=INK, opacity=.45, strokeWidth=2).encode(x="hour:Q")
    st.markdown("##### Risk over time")
    st.altair_chart((line + rule + fl).properties(height=300).configure_view(stroke=None)
                    .configure_axis(labelColor="#5B6B79", titleColor="#5B6B79", gridColor="#D5DCE2"),
                    use_container_width=True)
    st.caption("The red dashed line is the alert threshold. Dark vertical lines mark actual failures.")
    sens = st.multiselect("Sensor rolling averages", ["volt_6h_mean", "rotate_6h_mean", "pressure_6h_mean",
                                                       "vibration_6h_mean"], ["vibration_6h_mean"])
    if sens:
        st.line_chart(m.set_index("hour")[sens], height=250,
                      color=["#2F5D7C", "#E8A317", "#2E8B6E", "#C8372D"][:len(sens)])

# ---------------- tab 3: what-if ----------------
with tab3:
    st.caption("Start from a real record, then edit any value to see how the risk changes.")
    a, b = st.columns(2)
    pm = a.selectbox("Start from machine", sorted(hold["machineID"].unique()), key="wm")
    pool = hold[hold["machineID"] == pm].reset_index(drop=True)
    pi = b.slider("Hour in test period", 0, len(pool) - 1, 0, key="wh")
    base = pool.loc[pi, FEATURES]

    num = ["volt", "rotate", "pressure", "vibration", "Total_hours_from_last_maintenance", "errors_till_now", "age"] + \
          [f"{s}_6h_{x}" for s in ("volt", "rotate", "pressure", "vibration") for x in ("mean", "std")]
    row = base.copy()
    cols = st.columns(4)
    for i, c in enumerate(num):
        row[c] = cols[i % 4].number_input(c, value=float(base[c]), key=f"{c}_{pm}_{pi}")

    def pick(prefix, label):
        opts = [c for c in FEATURES if c.startswith(prefix)]
        cur = next((o for o in opts if base[o] == 1), opts[0])
        ch = st.selectbox(label, opts, index=opts.index(cur), key=f"{prefix}{pm}_{pi}")
        for o in opts:
            row[o] = 1 if o == ch else 0

    s1, s2, s3 = st.columns(3)
    with s1: pick("model_", "Machine model")
    with s2: pick("errorID_", "Current error")
    with s3: pick("comp_", "Maintenance this hour")

    risk = float(model.predict_proba(row[FEATURES].astype(float).to_frame().T)[:, 1][0])
    colr = ALERT if risk >= thr else WATCH if risk >= thr / 2 else "#2E8B6E"
    verdict = ("Alert: schedule an inspection for this machine." if risk >= thr else
               "Watch: risk is rising but below the threshold." if risk >= thr / 2 else "Normal: below the alert threshold.")
    st.markdown(
        f'<div class="panel"><h4>Failure risk in the next 6 hours: {risk:.1%}</h4>'
        f'<div class="gauge"><div class="fill" style="width:{min(risk, 1) * 100:.1f}%;background:{colr}"></div>'
        f'<div class="mark" style="left:{thr * 100:.1f}%"></div></div>'
        f'<div style="color:#5B6B79;font-size:.85rem">The black marker is the alert threshold ({thr:.1%}).</div>'
        f'<div style="margin-top:8px;font-weight:600;color:{colr if risk >= thr / 2 else INK}">{verdict}</div></div>',
        unsafe_allow_html=True)
    st.caption(f"Actual outcome for the starting record: "
               f"{'failure ahead' if pool.loc[pi, 'truth'] == 1 else 'no failure ahead'}")
    with st.expander("What drives the model overall"):
        imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False).head(15)
        st.bar_chart(imp, color="#2F5D7C")

# ---------------- tab 4: batch ----------------
with tab4:
    st.caption("Upload a CSV with the model's feature columns. An optional machineID column is kept in the output.")
    st.download_button("Download a template", pd.DataFrame(columns=FEATURES).to_csv(index=False), "template.csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up:
        df = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.warning(f"Missing columns filled with 0: {', '.join(missing)}")
        X = df.reindex(columns=FEATURES, fill_value=0).fillna(0)
        df["failure_risk"] = model.predict_proba(X)[:, 1]
        df["alert"] = (df["failure_risk"] >= thr).astype(int)
        st.write(f"{int(df['alert'].sum())} of {len(df)} rows are above the threshold.")
        st.dataframe(df.sort_values("failure_risk", ascending=False), use_container_width=True)
        st.download_button("Download predictions", df.to_csv(index=False), "predictions.csv")
