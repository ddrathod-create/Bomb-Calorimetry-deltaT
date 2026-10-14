"""
Delta T Determination for Bomb Calorimetry - PyQt5 App
-------------------------------------------------------
Interactive version of the Mathematica notebook.

- Open a 2-column data file (time in seconds, temperature)
- Four sliders: tmin, ti, td, tf (all in seconds)
    * a straight line is fitted to the initial period (tmin -> ti)
    * a straight line is fitted to the final period   (tf -> end of data)
    * Area 1 = area between the data and T(ti), from ti to td
    * Area 2 = area between T(tf) and the data, from td to tf
    * Delta T = (final line at td) - (initial line at td)
- Results update live as you drag; export results or save the plot

Requires: PyQt5, numpy, pandas, matplotlib
    pip install PyQt5 numpy pandas matplotlib
"""

import os
import sys
from decimal import Decimal

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QFrame, QSizePolicy,
)
from PyQt5.QtCore import Qt

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

matplotlib.rcParams["font.family"] = "sans-serif"

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #89b4fa;
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 15px;
}
QPushButton:hover {
    background-color: #45475a;
}
QPushButton:disabled {
    color: #6c7086;
    border-color: #313244;
}
QLabel {
    color: #cdd6f4;
}
QLabel#sliderLabel {
    font-size: 15px;
}
QLabel#fileLabel {
    font-size: 15px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #45475a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QStatusBar {
    background-color: #11111b;
    color: #a6adc8;
}
QFrame#resultsCard {
    background-color: #282a3a;
    border: 1px solid #89b4fa;
    border-radius: 10px;
}
QLabel#resultsTitle {
    color: #89b4fa;
    font-size: 16px;
    font-weight: 600;
    padding-bottom: 6px;
}
QLabel#resultsContent {
    color: #cdd6f4;
    font-size: 16px;
}
"""


def format_plain(value, sig_figs=4):
    """Format a number in plain decimal notation (never scientific)."""
    if value == 0:
        return "0"
    exponent = Decimal(str(float(value))).adjusted()
    decimal_places = max(sig_figs - exponent - 1, 0)
    return f"{float(value):.{decimal_places}f}"


def load_data(path):
    """Load a 2-column (time, temperature) file.

    Tries common encodings and delimiters (tab, comma, semicolon), ignores a
    header row and any extra/empty columns.
    """
    for encoding in ("utf-8", "utf-16", "latin1"):
        for sep in ("\t", ",", ";"):
            try:
                raw = pd.read_csv(path, sep=sep, header=None, encoding=encoding)
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

    # Sample the interpolated data at 1 s steps, like Table[{t, Temp[t]}, {t, a, b}]
    t_pre = np.arange(tmin, ti + 1e-9, 1.0)
    t_post = np.arange(tf, tmax + 1e-9, 1.0)
    if len(t_pre) < 2 or len(t_post) < 2:
        return None

    m1, b1 = np.polyfit(t_pre, temp(t_pre), 1)    # initial line: slope, intercept
    m2, b2 = np.polyfit(t_post, temp(t_post), 1)  # final line

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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("\u0394T Analysis - Bomb Calorimetry")
        self.resize(950, 950)

        self.data = None
        self.results_plain_text = ""

        # ---------------- Widgets ----------------
        self.open_button = QPushButton("Open Data File...")
        self.open_button.clicked.connect(self.open_file)

        self.file_label = QLabel("No file loaded")
        self.file_label.setObjectName("fileLabel")

        self.tmin_slider, self.tmin_value, tmin_row = self.make_slider_row("t<sub>min</sub>")
        self.ti_slider, self.ti_value, ti_row = self.make_slider_row("t<sub>i</sub>")
        self.td_slider, self.td_value, td_row = self.make_slider_row("t<sub>d</sub>")
        self.tf_slider, self.tf_value, tf_row = self.make_slider_row("t<sub>f</sub>")

        # Results card
        results_title = QLabel("\u0394T Results")
        results_title.setObjectName("resultsTitle")

        self.results_text = QLabel("Open a data file and adjust the sliders to see results.")
        self.results_text.setObjectName("resultsContent")
        self.results_text.setWordWrap(True)
        self.results_text.setTextFormat(Qt.RichText)
        self.results_text.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.results_card = QFrame()
        self.results_card.setObjectName("resultsCard")
        self.results_card.setFixedHeight(310)
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(20, 14, 20, 14)
        card_layout.addWidget(results_title)
        card_layout.addWidget(self.results_text)
        self.results_card.setLayout(card_layout)

        self.export_button = QPushButton("Export Results...")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)

        self.save_plot_button = QPushButton("Save Plot...")
        self.save_plot_button.clicked.connect(self.save_plot)
        self.save_plot_button.setEnabled(False)

        # Matplotlib canvas
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(420)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.figure.add_subplot(111)

        # ---------------- Layout ----------------
        top_row = QHBoxLayout()
        top_row.addWidget(self.open_button)
        top_row.addWidget(self.file_label)
        top_row.addStretch()

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.export_button)
        bottom_row.addWidget(self.save_plot_button)
        bottom_row.addStretch()

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.results_card, stretch=0)
        layout.addLayout(tmin_row)
        layout.addLayout(ti_row)
        layout.addLayout(td_row)
        layout.addLayout(tf_row)
        layout.addLayout(bottom_row)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.statusBar().showMessage("Open a data file to begin.")

    # ---------------- Helpers ----------------

    def make_slider_row(self, label_html):
        prefix = QLabel(label_html)
        prefix.setObjectName("sliderLabel")
        prefix.setFixedWidth(50)

        slider = QSlider(Qt.Horizontal)
        slider.valueChanged.connect(self.update_analysis)

        value = QLabel("- s")
        value.setObjectName("sliderLabel")
        value.setFixedWidth(80)

        row = QHBoxLayout()
        row.addWidget(prefix)
        row.addWidget(slider)
        row.addWidget(value)
        return slider, value, row

    # ---------------- Logic ----------------

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open data file", "", "Data Files (*.csv *.txt *.tsv);;All Files (*)"
        )
        if not path:
            return
        try:
            self.data = load_data(path)
        except Exception as e:
            self.results_text.setText(f"Failed to load file:<br>{e}")
            return

        self.file_label.setText(os.path.basename(path))

        tmax = int(self.data["t"].max())

        def clamp(v):
            return int(min(max(v, 0), tmax))

        # Default positions, same as the notebook
        tmin = clamp(80)
        ti = clamp(tmin + 280)
        td = clamp(ti + 80)
        tf = clamp(tmax - 400)

        sliders = [
            (self.tmin_slider, min(1000, tmax), tmin),
            (self.ti_slider, tmax, ti),
            (self.td_slider, tmax, td),
            (self.tf_slider, tmax, tf),
        ]
        for slider, maximum, value in sliders:
            slider.blockSignals(True)
            slider.setRange(0, maximum)
            slider.setValue(value)
            slider.blockSignals(False)

        self.export_button.setEnabled(True)
        self.save_plot_button.setEnabled(True)
        self.update_analysis()

    def update_analysis(self):
        if self.data is None:
            return

        tmin = self.tmin_slider.value()
        ti = self.ti_slider.value()
        td = self.td_slider.value()
        tf = self.tf_slider.value()

        self.tmin_value.setText(f"{tmin} s")
        self.ti_value.setText(f"{ti} s")
        self.td_value.setText(f"{td} s")
        self.tf_value.setText(f"{tf} s")

        t = self.data["t"].to_numpy()
        T = self.data["T"].to_numpy()
        r = analyse(t, T, tmin, ti, td, tf)
        if r is None:
            self.statusBar().showMessage(
                "Each fit period needs at least 2 points - adjust tmin / ti / tf."
            )
            return

        fp = format_plain
        self.results_plain_text = (
            f"\u0394T = {fp(r['dT'])} \u00b0C\n"
            f"T_i = {fp(r['Ti'])} \u00b0C\n"
            f"T_f = {fp(r['Tf'])} \u00b0C\n"
            f"Initial slope = {fp(r['m1'])} \u00b0C/s\n"
            f"Final slope = {fp(r['m2'])} \u00b0C/s\n"
            f"Area_1 = {fp(r['area1'])} \u00b0C s\n"
            f"Area_2 = {fp(r['area2'])} \u00b0C s\n"
            f"(tmin = {tmin} s, ti = {ti} s, td = {td} s, tf = {tf} s)"
        )
        self.results_text.setText(
            f"<p style='line-height:170%'>"
            f"&Delta;T = {fp(r['dT'])} &deg;C<br>"
            f"T<sub>i</sub> = {fp(r['Ti'])} &deg;C<br>"
            f"T<sub>f</sub> = {fp(r['Tf'])} &deg;C<br>"
            f"Initial slope = {fp(r['m1'])} &deg;C s<sup>-1</sup><br>"
            f"Final slope = {fp(r['m2'])} &deg;C s<sup>-1</sup><br>"
            f"Area<sub>1</sub> = {fp(r['area1'])} &deg;C s<br>"
            f"Area<sub>2</sub> = {fp(r['area2'])} &deg;C s"
            f"</p>"
        )
        self.statusBar().showMessage(
            f"Initial fit: {tmin}\u2013{ti} s   Final fit: {tf}\u2013{int(t.max())} s"
        )

        self.plot(t, T, tmin, ti, td, tf, r)

    def plot(self, t, T, tmin, ti, td, tf, r):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        ax = self.ax

        self.figure.patch.set_facecolor("white")
        ax.set_facecolor("white")
        dark = "#1e1e2e"
        temp = r["temp"]

        # Data
        ax.plot(t, T, color="#2e7d32", linewidth=1.8, label="Data")

        # Shaded areas
        if td > ti:
            x1 = np.linspace(ti, td, 400)
            ax.fill_between(x1, temp(x1), temp(ti), color="#42a5f5", alpha=0.35, label="Area 1")
        if tf > td:
            x2 = np.linspace(td, tf, 400)
            ax.fill_between(x2, temp(x2), temp(tf), color="#ffa726", alpha=0.35, label="Area 2")

        # Fitted lines (drawn from tmin to the end of the data, like the notebook)
        x = np.array([tmin, t.max()])
        ax.plot(x, r["m1"] * x + r["b1"], "--", color="#1565c0", linewidth=1.6, label="Initial fit")
        ax.plot(x, r["m2"] * x + r["b2"], "--", color="#ef6c00", linewidth=1.6, label="Final fit")

        # ti and tf markers
        ax.axvline(ti, color="#555555", linewidth=1, linestyle=":")
        ax.axvline(tf, color="#555555", linewidth=1, linestyle=":")
        ax.plot([ti], [r["Ti"]], "o", color="#1565c0", markersize=6)
        ax.plot([tf], [r["Tf"]], "o", color="#ef6c00", markersize=6)

        # Delta T bracket at td
        ax.plot([td, td], [r["pre_td"], r["post_td"]], color="#d81b60", linewidth=2.5, label="\u0394T")

        # Keep the y-range on the data (the extended lines can otherwise run off)
        pad = 0.04 * (T.max() - T.min())
        ax.set_ylim(T.min() - pad, T.max() + pad)

        ax.set_xlabel("Time (s)", fontsize=12, color=dark)
        ax.set_ylabel("Temperature (\u00b0C)", fontsize=12, color=dark)
        ax.tick_params(direction="in", top=True, right=True, colors=dark)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color(dark)
        ax.legend(frameon=False, loc="best")

        self.figure.tight_layout()
        self.canvas.draw()

    def export_results(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Results", "delta_t_results.txt", "Text Files (*.txt)")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.results_plain_text)

    def save_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", "delta_t_plot.png",
            "PNG Image (*.png);;PDF (*.pdf);;SVG (*.svg);;All Files (*)"
        )
        if not path:
            return
        self.figure.savefig(path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
