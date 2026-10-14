"""
Delta T Determination for Bomb Calorimetry - Streamlit App
----------------------------------------------------------
Streamlit version of the PyQt5 app.

- Upload a 2-column data file (time in seconds, temperature)
- Four sliders: tmin, ti, td, tf (all in seconds)
    * a straight line is fitted to the initial period (tmin -> ti)
    * a straight line is fitted to the final period   (tf -> end of data)
    * Area 1 = area between the data and T(ti), from ti to td
    * Area 2 = area between T(tf) and the data, from td to tf
    * Delta T = (final line at td) - (initial line at td)
- Results update live as you move the sliders; download results or the plot

Run with:
    pip install streamlit numpy pandas matplotlib
    streamlit run streamlit_app.py
"""

import io
from decimal import Decimal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ΔT Analysis - Bomb Calorimetry", layout="wide")


# ---------------------------------------------------------------- helpers

def format_plain(value, sig_figs=4):
    """Format a number in plain decimal notation (never scientific)."""
    if value == 0:
        return "0"
    exponent = Decimal(str(float(value))).adjusted()
    decimal_places = max(sig_figs - exponent - 1, 0)
    return f"{float(value):.{decimal_places}f}"


@st.cache_data(show_spinner=False)
def load_data(raw_bytes):
    """Load a 2-column (time, temperature) file from raw bytes.

    Tries common encodings and delimiters (tab, comma, semicolon), ignores a
    header row and any extra/empty columns.
    """
    for encoding in ("utf-8", "utf-16", "latin1"):
        for sep in ("\t", ",", ";"):
            try:
                raw = pd.read_csv(io.BytesIO(raw_bytes), sep=sep, header=None, encoding=encoding)
            except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
                continue
            raw = raw.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
            if raw.shape[1] >= 2:
                data = raw.iloc[:, :2].dropna()
                if len(data) > 5:
                    data.columns = ["t", "T"]
                    return data.reset_index(drop=True)
    raise ValueError("Could not find two numeric columns (time, temperature) in this file.")


def integrate(func, a, b, n=2000):
    """Trapezoid-rule integral of func from a to b (signed, like Mathematica)."""
    x = np.linspace(a, b, n)
    y = func(x)
    return np.sum((y[1:] + y[:-1]) / 2 * np.diff(x))


def analyse(t, T, tmin, ti, td, tf):
    """Do the Delta T analysis. Returns a dict of results, or None if a fit period has < 2 points."""
    tmax = t.max()

    def temp(x):
        return np.interp(x, t, T)

    t_pre = np.arange(tmin, ti + 1e-9, 1.0)
    t_post = np.arange(tf, tmax + 1e-9, 1.0)
    if len(t_pre) < 2 or len(t_post) < 2:
        return None

    m1, b1 = np.polyfit(t_pre, temp(t_pre), 1)
    m2, b2 = np.polyfit(t_post, temp(t_post), 1)

    def pre(x):
        return m1 * x + b1

    def post(x):
        return m2 * x + b2

    area1 = integrate(temp, ti, td) - temp(ti) * (td - ti)
    area2 = temp(tf) * (tf - td) - integrate(temp, td, tf)

    return {
        "m1": m1, "b1": b1, "m2": m2, "b2": b2,
        "Ti": pre(ti), "Tf": post(tf),
        "dT": post(td) - pre(td),
        "area1": area1, "area2": area2,
        "pre_td": pre(td), "post_td": post(td),
        "temp": temp,
    }


def make_figure(t, T, tmin, ti, td, tf, r):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    dark = "#1e1e2e"
    temp = r["temp"]

    ax.plot(t, T, color="#2e7d32", linewidth=1.8, label="Data")

    if td > ti:
        x1 = np.linspace(ti, td, 400)
        ax.fill_between(x1, temp(x1), temp(ti), color="#42a5f5", alpha=0.35, label="Area 1")
    if tf > td:
        x2 = np.linspace(td, tf, 400)
        ax.fill_between(x2, temp(x2), temp(tf), color="#ffa726", alpha=0.35, label="Area 2")

    x = np.array([tmin, t.max()])
    ax.plot(x, r["m1"] * x + r["b1"], "--", color="#1565c0", linewidth=1.6, label="Initial fit")
    ax.plot(x, r["m2"] * x + r["b2"], "--", color="#ef6c00", linewidth=1.6, label="Final fit")

    ax.axvline(ti, color="#555555", linewidth=1, linestyle=":")
    ax.axvline(tf, color="#555555", linewidth=1, linestyle=":")
    ax.plot([ti], [r["Ti"]], "o", color="#1565c0", markersize=6)
    ax.plot([tf], [r["Tf"]], "o", color="#ef6c00", markersize=6)

    ax.plot([td, td], [r["pre_td"], r["post_td"]], color="#d81b60", linewidth=2.5, label="\u0394T")

    pad = 0.04 * (T.max() - T.min())
    ax.set_ylim(T.min() - pad, T.max() + pad)

    ax.set_xlabel("Time (s)", fontsize=12, color=dark)
    ax.set_ylabel("Temperature (\u00b0C)", fontsize=12, color=dark)
    ax.tick_params(direction="in", top=True, right=True, colors=dark)
    ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color(dark)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig


def fig_bytes(fig, fmt):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=300, bbox_inches="tight")
    return buf.getvalue()


# ---------------------------------------------------------------- app

st.title("\u0394T Analysis \u2013 Bomb Calorimetry")

uploaded = st.file_uploader(
    "Open data file (2 columns: time in s, temperature)",
    type=["csv", "txt", "tsv"],
)

if uploaded is None:
    st.info("Upload a data file to begin.")
    st.stop()

raw_bytes = uploaded.getvalue()
try:
    data = load_data(raw_bytes)
except Exception as e:
    st.error(f"Failed to load file: {e}")
    st.stop()

t = data["t"].to_numpy()
T = data["T"].to_numpy()
tmax = int(t.max())


def clamp(v):
    return int(min(max(v, 0), tmax))


# Reset slider defaults whenever a new file is loaded (same defaults as the notebook)
file_id = (uploaded.name, len(raw_bytes))
if st.session_state.get("file_id") != file_id:
    st.session_state["file_id"] = file_id
    _tmin = clamp(80)
    _ti = clamp(_tmin + 280)
    _td = clamp(_ti + 80)
    _tf = clamp(tmax - 400)
    st.session_state.update(tmin=_tmin, ti=_ti, td=_td, tf=_tf)

st.caption(f"Loaded **{uploaded.name}** \u2013 {len(data)} points, 0\u2013{tmax} s")

with st.sidebar:
    st.header("Parameters")
    tmin = st.slider("t_min (s)", 0, min(1000, tmax), key="tmin")
    ti = st.slider("t_i (s)", 0, tmax, key="ti")
    td = st.slider("t_d (s)", 0, tmax, key="td")
    tf = st.slider("t_f (s)", 0, tmax, key="tf")

r = analyse(t, T, tmin, ti, td, tf)
if r is None:
    st.warning("Each fit period needs at least 2 points \u2013 adjust t_min / t_i / t_f.")
    st.stop()

fp = format_plain

fig = make_figure(t, T, tmin, ti, td, tf, r)
col_plot, col_res = st.columns([3, 1.3])

with col_plot:
    st.pyplot(fig, clear_figure=False)
    st.caption(f"Initial fit: {tmin}\u2013{ti} s   |   Final fit: {tf}\u2013{tmax} s")

results_plain_text = (
    f"\u0394T = {fp(r['dT'])} \u00b0C\n"
    f"T_i = {fp(r['Ti'])} \u00b0C\n"
    f"T_f = {fp(r['Tf'])} \u00b0C\n"
    f"Initial slope = {fp(r['m1'])} \u00b0C/s\n"
    f"Final slope = {fp(r['m2'])} \u00b0C/s\n"
    f"Area_1 = {fp(r['area1'])} \u00b0C s\n"
    f"Area_2 = {fp(r['area2'])} \u00b0C s\n"
    f"(tmin = {tmin} s, ti = {ti} s, td = {td} s, tf = {tf} s)"
)

with col_res:
    with st.container(border=True):
        st.subheader("\u0394T Results")
        st.metric("\u0394T", f"{fp(r['dT'])} \u00b0C")
        st.markdown(
            f"""
- T<sub>i</sub> = {fp(r['Ti'])} \u00b0C
- T<sub>f</sub> = {fp(r['Tf'])} \u00b0C
- Initial slope = {fp(r['m1'])} \u00b0C s<sup>\u22121</sup>
- Final slope = {fp(r['m2'])} \u00b0C s<sup>\u22121</sup>
- Area<sub>1</sub> = {fp(r['area1'])} \u00b0C s
- Area<sub>2</sub> = {fp(r['area2'])} \u00b0C s
""",
            unsafe_allow_html=True,
        )

    st.download_button(
        "Export Results (.txt)",
        data=results_plain_text,
        file_name="delta_t_results.txt",
        mime="text/plain",
        use_container_width=True,
    )
    fmt = st.selectbox("Plot format", ["png", "pdf", "svg"])
    st.download_button(
        f"Save Plot (.{fmt})",
        data=fig_bytes(fig, fmt),
        file_name=f"delta_t_plot.{fmt}",
        mime={"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}[fmt],
        use_container_width=True,
    )

plt.close(fig)
