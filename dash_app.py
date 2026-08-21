"""
RL4AXP — Dash Dashboard
AMP Peptide Design: live training monitor, candidate browser, SQA refinement, custom scorer.

Run:  /home/cylin/.venv/bin/python dash_app.py
      then open  http://127.0.0.1:8050
"""

import gpu_setup  # MUST be first — pre-loads PyTorch CUDA before TF

import os
import json
import threading
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px

import config as cfg

# ─────────────────────────────────────────────────────────────────────────────
# Shared in-process state (training runs in a background thread)
# ─────────────────────────────────────────────────────────────────────────────
_lock = threading.Lock()
_state = {
    "status":       "idle",       # idle | initializing | training | paused | done | error
    "error_msg":    "",
    "episode":      0,
    "exp_rows":     [],           # list[dict] — one per parallel trajectory per episode
    "loss_data":    {"actor1_loss": [], "actor2_loss": [], "critic_loss": [],
                     "entropy1": [], "entropy2": []},
    "lr_data":      [],
    "sqa_rows":     [],           # list[dict] from run_sqa_refinement()
    "sqa_status":   "idle",       # idle | running | done | error
    "sqa_error":    "",
    "scorer_result": {},          # {model: score} for custom scorer
}
_framework = None
_stop_event = threading.Event()

ALL_MODELS   = ["AMP", "ACP", "AFP", "AVP", "HEM"]
MODEL_COLORS = {
    "AMP": "#2196F3", "ACP": "#4CAF50", "AFP": "#FF9800",
    "AVP": "#9C27B0", "HEM": "#F44336",
}
_WEIGHT_INFO = {
    "AMP": "Antimicrobial activity weight. Reward multiplier for AMP probability improvement. ↑ maximize (range 0.1–5.0).",
    "ACP": "Anticancer peptide weight. Reward multiplier for ACP probability improvement. ↑ maximize (range 0.1–5.0).",
    "AFP": "Antifungal activity weight. Reward multiplier for AFP probability improvement. ↑ maximize (range 0.1–5.0).",
    "AVP": "Antiviral activity weight. Reward multiplier for AVP probability improvement. ↑ maximize (range 0.1–5.0).",
    "HEM": "Hemolysis penalty weight. Set higher than activity weights to counteract the 4-vs-1 structural imbalance. ↓ minimize (range 0.1–5.0).",
}

# ─────────────────────────────────────────────────────────────────────────────
# Thread helpers
# ─────────────────────────────────────────────────────────────────────────────

def _on_episode_end(episode, exp_df, loss_data, lr_data):
    with _lock:
        _state["episode"]   = episode
        _state["exp_rows"]  = exp_df.to_dict("records")
        _state["loss_data"] = loss_data
        _state["lr_data"]   = lr_data


def _training_thread(target_peptide, reward_models, n_parallels, time_horizon,
                     reward_weights=None, hem_threshold=None, hem_penalty=None):
    global _framework
    with _lock:
        _state["status"] = "initializing"
        _state["error_msg"] = ""

    try:
        # Reconfigure before building framework
        cfg.TARGET_PEPTIDE = target_peptide
        cfg.REWARD_MODELS  = reward_models
        cfg.N_PARALLELS    = n_parallels
        cfg.TIME_HORIZON   = time_horizon
        if reward_weights  is not None: cfg.REWARD_WEIGHTS    = reward_weights
        if hem_threshold   is not None: cfg.HEM_THRESHOLD     = hem_threshold
        if hem_penalty     is not None: cfg.HEM_PENALTY_SCALE = hem_penalty

        from peptide_optimization.framework import Framework
        _framework = Framework()

        with _lock:
            _state["status"]    = "training"
            _state["episode"]   = 0
            _state["exp_rows"]  = []
            _state["loss_data"] = {"actor1_loss": [], "actor2_loss": [], "critic_loss": [],
                                   "entropy1": [], "entropy2": []}
            _state["lr_data"]   = []

        _framework.train(
            on_episode_end=_on_episode_end,
            stop_event=_stop_event,
        )

        with _lock:
            _state["status"] = "done" if not _stop_event.is_set() else "paused"

    except Exception as exc:
        with _lock:
            _state["status"]    = "error"
            _state["error_msg"] = str(exc)


def _sqa_thread(top_n):
    if _framework is None:
        return
    with _lock:
        _state["sqa_status"] = "running"
        _state["sqa_error"]  = ""
    try:
        df = _framework.run_sqa_refinement(top_n=top_n)
        with _lock:
            _state["sqa_rows"]   = df.to_dict("records") if df is not None and len(df) else []
            _state["sqa_status"] = "done"
    except Exception as exc:
        with _lock:
            _state["sqa_status"] = "error"
            _state["sqa_error"]  = str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Utility: smooth a series with Gaussian kernel
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(data, sigma=30):
    if len(data) < 4:
        return data
    return gaussian_filter1d(data, sigma=min(sigma, max(1, len(data) // 10))).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="RL4AXP · AMP Design",
    suppress_callback_exceptions=True,
)

# ── Color palette ─────────────────────────────────────────────────────────────
C = {
    "bg":      "#1a1d23",
    "panel":   "#22262f",
    "border":  "#2e3340",
    "text":    "#e0e4f0",
    "muted":   "#8892a4",
    "accent":  "#4a9eff",
    "success": "#52c78e",
    "warning": "#f5a623",
    "danger":  "#e85c5c",
}

BADGE_MAP = {
    "idle":         ("secondary", "Idle"),
    "initializing": ("warning",   "Initializing…"),
    "training":     ("success",   "Training"),
    "paused":       ("info",      "Paused"),
    "done":         ("primary",   "Done"),
    "error":        ("danger",    "Error"),
}


# ─── Layout helpers ─────────────────────────────────────────────────────────

def _section(title, children, id_=None):
    kwargs = {"id": id_} if id_ else {}
    return html.Div([
        html.Div(title, style={"fontSize": "11px", "fontWeight": "700",
                               "letterSpacing": "0.1em", "color": C["muted"],
                               "marginBottom": "8px", "textTransform": "uppercase"}),
        *children,
    ], style={"marginBottom": "20px"}, **kwargs)


def _card(children, **style):
    base = {"background": C["panel"], "border": f"1px solid {C['border']}",
            "borderRadius": "8px", "padding": "16px", "height": "100%"}
    base.update(style)
    return html.Div(children, style=base)


def _labeled(label, component):
    return html.Div([
        html.Label(label, style={"fontSize": "12px", "color": C["muted"],
                                 "marginBottom": "4px", "display": "block"}),
        component,
    ], style={"marginBottom": "12px"})


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def _sidebar():
    inp_style = {
        "background": C["bg"], "border": f"1px solid {C['border']}",
        "color": C["text"], "borderRadius": "6px",
        "padding": "6px 10px", "width": "100%", "fontSize": "13px",
    }
    num_style = {**inp_style, "padding": "6px 8px"}

    return html.Div([
        html.Div([
            html.Span("RL4AXP", style={"fontSize": "20px", "fontWeight": "800",
                                       "color": C["accent"], "letterSpacing": "0.05em"}),
            html.Span(" AMP", style={"fontSize": "20px", "fontWeight": "300",
                                     "color": C["text"]}),
            html.Div("Antimicrobial Peptide Design", style={"fontSize": "11px",
                      "color": C["muted"], "marginTop": "2px"}),
        ], style={"marginBottom": "24px", "paddingBottom": "16px",
                  "borderBottom": f"1px solid {C['border']}"}),

        _section("Status", [
            html.Div(id="status-badge",
                     style={"display": "flex", "alignItems": "center", "gap": "8px"}),
            html.Div(id="status-episode",
                     style={"fontSize": "12px", "color": C["muted"], "marginTop": "4px"}),
        ]),

        _section("Target Peptide", [
            dcc.Input(id="inp-peptide", value=cfg.TARGET_PEPTIDE, type="text",
                      debounce=True, style={**inp_style, "fontFamily": "monospace"}),
        ]),

        _section("Reward Models", [
            dcc.Checklist(
                id="chk-models",
                options=[{"label": html.Span(m, style={"color": MODEL_COLORS[m],
                           "fontWeight": "600"}), "value": m} for m in ALL_MODELS],
                value=list(cfg.REWARD_MODELS),
                labelStyle={"display": "flex", "alignItems": "center",
                            "gap": "6px", "marginBottom": "4px"},
                inputStyle={"cursor": "pointer"},
            ),
            html.Div(style={"height": "8px"}),
            *[
                html.Div([
                    html.Div([
                        html.Span(("↓ " if m == "HEM" else "↑ ") + m,
                                  style={"color": MODEL_COLORS[m], "fontSize": "11px",
                                         "fontWeight": "700"}),
                        html.Div([
                            html.Span("ⓘ", id=f"info-w-{m}",
                                      style={"color": C["muted"], "fontSize": "10px",
                                             "cursor": "help", "marginRight": "5px"}),
                            html.Span(f"{cfg.REWARD_WEIGHTS.get(m, 1.0):.1f}",
                                      id=f"val-w-{m}",
                                      style={"color": C["muted"], "fontSize": "11px"}),
                        ], style={"display": "flex", "alignItems": "center"}),
                    ], style={"display": "flex", "justifyContent": "space-between",
                              "marginBottom": "1px"}),
                    dcc.Slider(id=f"slider-w-{m}", min=0.1, max=5.0, step=0.1,
                               value=cfg.REWARD_WEIGHTS.get(m, 1.0),
                               marks=None, updatemode="drag"),
                ], style={"marginBottom": "4px"})
                for m in ALL_MODELS
            ],
            # Tooltips for model weights
            *[dbc.Tooltip(_WEIGHT_INFO[m], target=f"info-w-{m}",
                          placement="right", style={"maxWidth": "220px"})
              for m in ALL_MODELS],
            html.Div(style={"height": "4px"}),
            html.Div([
                html.Div([
                    html.Div([
                        html.Span("HEM thr", style={"color": C["muted"], "fontSize": "11px"}),
                        html.Span("ⓘ", id="info-hem-thr",
                                  style={"color": C["muted"], "fontSize": "10px",
                                         "cursor": "help", "marginLeft": "4px"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                    html.Span(f"{cfg.HEM_THRESHOLD:.2f}", id="val-hem-thr",
                              style={"color": C["muted"], "fontSize": "11px"}),
                ], style={"display": "flex", "justifyContent": "space-between",
                          "marginBottom": "1px"}),
                dcc.Slider(id="slider-hem-thr", min=0.0, max=1.0, step=0.05,
                           value=cfg.HEM_THRESHOLD, marks=None, updatemode="drag"),
                dbc.Tooltip(
                    "HEM threshold: extra penalty is applied every training step "
                    "when the hemolysis probability exceeds this value. "
                    "Lower = stricter hemolysis control.",
                    target="info-hem-thr", placement="right", style={"maxWidth": "220px"}),
            ], style={"marginBottom": "4px"}),
            html.Div([
                html.Div([
                    html.Div([
                        html.Span("HEM penalty", style={"color": C["muted"], "fontSize": "11px"}),
                        html.Span("ⓘ", id="info-hem-pen",
                                  style={"color": C["muted"], "fontSize": "10px",
                                         "cursor": "help", "marginLeft": "4px"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                    html.Span(f"{cfg.HEM_PENALTY_SCALE:.1f}", id="val-hem-pen",
                              style={"color": C["muted"], "fontSize": "11px"}),
                ], style={"display": "flex", "justifyContent": "space-between",
                          "marginBottom": "1px"}),
                dcc.Slider(id="slider-hem-pen", min=0.0, max=5.0, step=0.1,
                           value=cfg.HEM_PENALTY_SCALE, marks=None, updatemode="drag"),
                dbc.Tooltip(
                    "Penalty scale: additional reward deduction per unit of hemolysis "
                    "probability above the threshold, applied each step. "
                    "Increase if HEM remains high during training.",
                    target="info-hem-pen", placement="right", style={"maxWidth": "220px"}),
            ]),
        ]),

        _section("Hyperparameters", [
            _labeled("N Parallels",
                dcc.Input(id="inp-npar", value=cfg.N_PARALLELS, type="number",
                          min=1, max=1000, step=1, debounce=True, style=num_style)),
            _labeled("Time Horizon",
                dcc.Input(id="inp-th", value=cfg.TIME_HORIZON, type="number",
                          min=1, max=50, step=1, debounce=True, style=num_style)),
        ]),

        html.Div([
            dbc.Button("▶ Start Training", id="btn-start", color="success", size="sm",
                       style={"width": "100%", "marginBottom": "8px", "fontWeight": "600"}),
            dbc.Button("⏸ Pause / Stop", id="btn-stop", color="warning", size="sm",
                       style={"width": "100%", "marginBottom": "8px"}),
            dbc.Button("↺ Reset", id="btn-reset", color="danger", outline=True,
                       size="sm", style={"width": "100%"}),
        ]),

    ], style={
        "width": "270px", "minWidth": "270px",
        "background": C["panel"], "borderRight": f"1px solid {C['border']}",
        "padding": "20px 16px", "height": "100vh",
        "overflowY": "auto", "flexShrink": "0",
    })


# ─── Tab: Monitor ──────────────────────────────────────────────────────────────

def _tab_monitor():
    chart_style = {"height": "240px"}
    return html.Div([
        dbc.Row([
            dbc.Col(_card([
                html.Div("Cumulative Reward", style={"fontSize": "12px",
                          "color": C["muted"], "marginBottom": "4px"}),
                dcc.Graph(id="chart-reward", style=chart_style, config={"displayModeBar": False}),
            ]), md=6),
            dbc.Col(_card([
                html.Div("Heuristic Score", style={"fontSize": "12px",
                          "color": C["muted"], "marginBottom": "4px"}),
                dcc.Graph(id="chart-heuristic", style=chart_style, config={"displayModeBar": False}),
            ]), md=6),
        ], className="g-2 mb-2"),
        dbc.Row([
            dbc.Col(_card([
                html.Div("Model Probabilities", style={"fontSize": "12px",
                          "color": C["muted"], "marginBottom": "4px"}),
                dcc.Graph(id="chart-probs", style={"height": "260px"}, config={"displayModeBar": False}),
            ]), md=12),
        ], className="g-2 mb-2"),
        dbc.Row([
            dbc.Col(_card([
                html.Div("Actor Losses", style={"fontSize": "12px",
                          "color": C["muted"], "marginBottom": "4px"}),
                dcc.Graph(id="chart-losses", style=chart_style, config={"displayModeBar": False}),
            ]), md=8),
            dbc.Col(_card([
                html.Div("Learning Rate", style={"fontSize": "12px",
                          "color": C["muted"], "marginBottom": "4px"}),
                dcc.Graph(id="chart-lr", style=chart_style, config={"displayModeBar": False}),
            ]), md=4),
        ], className="g-2"),
    ], style={"padding": "12px", "overflowY": "auto", "height": "calc(100vh - 100px)"})


# ─── Tab: Candidates ──────────────────────────────────────────────────────────

def _tab_candidates():
    return html.Div([
        html.Div([
            html.Div("Optimization Candidates", style={"fontSize": "16px",
                      "fontWeight": "700", "color": C["text"]}),
            html.Div(id="cand-count", style={"fontSize": "12px",
                      "color": C["muted"], "marginTop": "2px"}),
        ], style={"marginBottom": "12px",
                  "display": "flex", "justifyContent": "space-between",
                  "alignItems": "flex-end"}),
        html.Div([
            html.Div("Sort by:", style={"fontSize": "12px", "color": C["muted"],
                                       "marginRight": "8px", "alignSelf": "center"}),
            dcc.Dropdown(
                id="cand-sort-col",
                options=[
                    {"label": "Cumulative Reward", "value": "Cumulative-Reward"},
                    {"label": "Heuristic", "value": "Heuristic_T"},
                    *[{"label": f"{m} Prob", "value": f"{m}-Prob_T"} for m in ALL_MODELS],
                ],
                value="Cumulative-Reward",
                clearable=False,
                style={"width": "200px", "fontSize": "12px"},
            ),
            dcc.Input(id="cand-filter", placeholder="Filter sequence…", type="text",
                      debounce=True, style={
                          "background": C["bg"], "border": f"1px solid {C['border']}",
                          "color": C["text"], "borderRadius": "6px",
                          "padding": "6px 10px", "fontSize": "12px",
                          "marginLeft": "10px", "width": "200px",
                      }),
            dbc.Button("⬇ Export CSV", id="btn-export", color="primary", outline=True,
                       size="sm", style={"marginLeft": "auto"}),
            dcc.Download(id="download-csv"),
        ], style={"display": "flex", "alignItems": "center",
                  "gap": "8px", "marginBottom": "10px", "flexWrap": "wrap"}),
        html.Div(id="cand-table-wrapper",
                 style={"height": "calc(100vh - 210px)", "overflowY": "auto"}),
    ], style={"padding": "16px"})


# ─── Tab: SQA Refinement ──────────────────────────────────────────────────────

def _tab_sqa():
    inp_style = {
        "background": C["bg"], "border": f"1px solid {C['border']}",
        "color": C["text"], "borderRadius": "6px",
        "padding": "6px 10px", "fontSize": "13px",
    }
    return html.Div([
        html.Div("SQA Quantum Refinement", style={"fontSize": "16px",
                  "fontWeight": "700", "color": C["text"], "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col(_card([
                html.Div("Parameters", style={"fontSize": "13px", "fontWeight": "700",
                          "color": C["text"], "marginBottom": "12px"}),
                _labeled("Top-N candidates", dcc.Input(id="sqa-topn", value=10, type="number",
                          min=1, max=100, style={**inp_style, "width": "100%"})),
                _labeled("N Positions (Actor1 top-k)",
                    dcc.Input(id="sqa-npos", value=cfg.SQA_N_POSITIONS, type="number",
                              min=1, max=20, style={**inp_style, "width": "100%"})),
                _labeled("N AAs per position",
                    dcc.Input(id="sqa-naas", value=cfg.SQA_N_AAS, type="number",
                              min=1, max=10, style={**inp_style, "width": "100%"})),
                _labeled("Trotter slices",
                    dcc.Input(id="sqa-trotter", value=cfg.SQA_N_TROTTER, type="number",
                              min=4, max=80, style={**inp_style, "width": "100%"})),
                _labeled("Annealing steps",
                    dcc.Input(id="sqa-steps", value=cfg.SQA_N_STEPS, type="number",
                              min=50, max=5000, style={**inp_style, "width": "100%"})),
                dbc.Button("⚛ Run SQA", id="btn-sqa", color="info",
                           style={"width": "100%", "marginTop": "8px", "fontWeight": "600"}),
                html.Div(id="sqa-status-msg", style={"fontSize": "12px",
                          "color": C["muted"], "marginTop": "8px"}),
            ]), md=3),
            dbc.Col(_card([
                html.Div("Refinement Results", style={"fontSize": "13px", "fontWeight": "700",
                          "color": C["text"], "marginBottom": "12px"}),
                html.Div(id="sqa-results-area"),
            ]), md=9),
        ], className="g-2"),
    ], style={"padding": "16px", "overflowY": "auto",
              "height": "calc(100vh - 100px)"})


# ─── Tab: Scorer ──────────────────────────────────────────────────────────────

def _tab_scorer():
    inp_style = {
        "background": C["bg"], "border": f"1px solid {C['border']}",
        "color": C["text"], "borderRadius": "6px",
        "padding": "8px 12px", "fontSize": "14px", "fontFamily": "monospace",
        "width": "100%",
    }
    return html.Div([
        html.Div("Custom Sequence Scorer", style={"fontSize": "16px",
                  "fontWeight": "700", "color": C["text"], "marginBottom": "12px"}),
        dbc.Row([
            dbc.Col(_card([
                _labeled("Peptide Sequence (single-letter AA code)", [
                    dcc.Input(id="scorer-seq", placeholder="e.g. RVKRVWPLVIR",
                              type="text", debounce=True, style=inp_style),
                ]),
                _labeled("HEM Concentration (μg/mL)", [
                    dcc.Input(id="scorer-conc", value=50.0, type="number",
                              style={**inp_style, "fontFamily": "inherit"}),
                ]),
                html.Div("Score with:", style={"fontSize": "12px", "color": C["muted"],
                                              "marginBottom": "6px"}),
                dcc.Checklist(
                    id="scorer-models",
                    options=[{"label": html.Span(m, style={"color": MODEL_COLORS[m],
                               "fontWeight": "600"}), "value": m} for m in ALL_MODELS],
                    value=list(ALL_MODELS),
                    labelStyle={"display": "flex", "alignItems": "center",
                                "gap": "6px", "marginBottom": "5px"},
                    inputStyle={"cursor": "pointer"},
                ),
                dbc.Button("Score", id="btn-score", color="primary",
                           style={"width": "100%", "marginTop": "8px", "fontWeight": "600"}),
                html.Div(id="scorer-error", style={"color": C["danger"],
                          "fontSize": "12px", "marginTop": "6px"}),
            ]), md=4),
            dbc.Col(_card([
                html.Div("Scores", style={"fontSize": "13px", "fontWeight": "700",
                          "color": C["text"], "marginBottom": "12px"}),
                html.Div(id="scorer-results"),
            ]), md=8),
        ], className="g-2"),
    ], style={"padding": "16px", "overflowY": "auto",
              "height": "calc(100vh - 50px)"})


# ─── Full layout ─────────────────────────────────────────────────────────────

TABS_STYLE = {
    "background": C["bg"], "borderBottom": f"1px solid {C['border']}",
    "padding": "0 12px",
}
TAB_STYLE = {
    "color": C["muted"], "background": "transparent",
    "border": "none", "borderBottom": "2px solid transparent",
    "padding": "10px 16px", "fontSize": "13px", "fontWeight": "500",
    "cursor": "pointer",
}
TAB_SEL_STYLE = {
    **TAB_STYLE,
    "color": C["accent"], "borderBottom": f"2px solid {C['accent']}",
    "fontWeight": "700",
}

app.layout = html.Div([
    # Polling interval
    dcc.Interval(id="interval", interval=2000, n_intervals=0),
    # Notification toast
    dbc.Toast(id="toast", is_open=False, duration=3000,
              style={"position": "fixed", "top": 12, "right": 12, "zIndex": 9999,
                     "minWidth": "280px"}),

    html.Div([
        # Sidebar
        _sidebar(),
        # Main content — inline tabs so all component IDs exist in DOM at all times
        html.Div([
            dcc.Tabs(id="main-tabs", value="monitor",
                     style={**TABS_STYLE, "flex": "0 0 auto"},
                     children=[
                         dcc.Tab(label="📊 Monitor",    value="monitor",
                                 style=TAB_STYLE, selected_style=TAB_SEL_STYLE,
                                 children=_tab_monitor()),
                         dcc.Tab(label="🧬 Candidates", value="candidates",
                                 style=TAB_STYLE, selected_style=TAB_SEL_STYLE,
                                 children=_tab_candidates()),
                         dcc.Tab(label="⚛ SQA",        value="sqa",
                                 style=TAB_STYLE, selected_style=TAB_SEL_STYLE,
                                 children=_tab_sqa()),
                         dcc.Tab(label="🔬 Scorer",     value="scorer",
                                 style=TAB_STYLE, selected_style=TAB_SEL_STYLE,
                                 children=_tab_scorer()),
                     ]),
        ], style={"flex": "1", "overflow": "hidden",
                  "display": "flex", "flexDirection": "column"}),
    ], style={"display": "flex", "height": "100vh",
              "background": C["bg"], "color": C["text"],
              "fontFamily": "'Inter', 'Segoe UI', sans-serif"}),
], style={"margin": "0", "padding": "0"})


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

# ── Training control ─────────────────────────────────────────────────────────

# ── Slider value labels (live update while dragging) ─────────────────────────

@app.callback(
    Output("val-w-AMP",  "children"),
    Output("val-w-ACP",  "children"),
    Output("val-w-AFP",  "children"),
    Output("val-w-AVP",  "children"),
    Output("val-w-HEM",  "children"),
    Output("val-hem-thr","children"),
    Output("val-hem-pen","children"),
    Input("slider-w-AMP",   "value"),
    Input("slider-w-ACP",   "value"),
    Input("slider-w-AFP",   "value"),
    Input("slider-w-AVP",   "value"),
    Input("slider-w-HEM",   "value"),
    Input("slider-hem-thr", "value"),
    Input("slider-hem-pen", "value"),
)
def update_weight_labels(w_amp, w_acp, w_afp, w_avp, w_hem, hem_thr, hem_pen):
    fmt  = lambda v, d: f"{float(v):.1f}" if v is not None else d
    fmt2 = lambda v, d: f"{float(v):.2f}" if v is not None else d
    return (fmt(w_amp, "1.0"), fmt(w_acp, "0.6"), fmt(w_afp, "0.6"),
            fmt(w_avp, "0.6"), fmt(w_hem, "2.5"),
            fmt2(hem_thr, "0.30"), fmt(hem_pen, "1.0"))


# ── Training control ─────────────────────────────────────────────────────────

@app.callback(
    Output("toast", "children"),
    Output("toast", "is_open"),
    Output("toast", "header"),
    Output("toast", "icon"),
    Input("btn-start", "n_clicks"),
    Input("btn-stop",  "n_clicks"),
    Input("btn-reset", "n_clicks"),
    State("inp-peptide",    "value"),
    State("chk-models",     "value"),
    State("inp-npar",       "value"),
    State("inp-th",         "value"),
    State("slider-w-AMP",   "value"),
    State("slider-w-ACP",   "value"),
    State("slider-w-AFP",   "value"),
    State("slider-w-AVP",   "value"),
    State("slider-w-HEM",   "value"),
    State("slider-hem-thr", "value"),
    State("slider-hem-pen", "value"),
    prevent_initial_call=True,
)
def control_training(start, stop, reset, peptide, models, n_par, time_h,
                     w_amp, w_acp, w_afp, w_avp, w_hem, hem_thr, hem_pen):
    global _framework, _stop_event

    triggered = ctx.triggered_id

    if triggered == "btn-start":
        with _lock:
            st = _state["status"]
        if st in ("training", "initializing"):
            return "Training is already running.", True, "Info", "info"
        if not peptide or not models:
            return "Provide peptide and select at least one model.", True, "Warning", "warning"

        aa_valid = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(c in aa_valid for c in peptide.upper()):
            return "Sequence contains invalid amino acids.", True, "Error", "danger"

        reward_weights = {
            "AMP": float(w_amp or 1.0),
            "ACP": float(w_acp or 0.6),
            "AFP": float(w_afp or 0.6),
            "AVP": float(w_avp or 0.6),
            "HEM": float(w_hem or 2.5),
        }
        _stop_event.clear()
        t = threading.Thread(
            target=_training_thread,
            args=(peptide.upper(), models, int(n_par or 200), int(time_h or 5)),
            kwargs={
                "reward_weights": reward_weights,
                "hem_threshold":  float(hem_thr if hem_thr is not None else 0.3),
                "hem_penalty":    float(hem_pen if hem_pen is not None else 1.0),
            },
            daemon=True,
        )
        t.start()
        return "Training started!", True, "Success", "success"

    if triggered == "btn-stop":
        _stop_event.set()
        return "Stop signal sent.", True, "Info", "info"

    if triggered == "btn-reset":
        _stop_event.set()
        with _lock:
            _state.update({
                "status": "idle", "error_msg": "", "episode": 0,
                "exp_rows": [], "loss_data": {"actor1_loss": [], "actor2_loss": [],
                                               "critic_loss": [], "entropy1": [], "entropy2": []},
                "lr_data": [], "sqa_rows": [], "sqa_status": "idle", "sqa_error": "",
                "scorer_result": {},
            })
        _framework = None
        return "State reset.", True, "Info", "secondary"

    return "", False, "", "primary"


# ── Status badge & episode counter ───────────────────────────────────────────

@app.callback(
    Output("status-badge",   "children"),
    Output("status-episode", "children"),
    Input("interval", "n_intervals"),
)
def update_status(_):
    with _lock:
        st      = _state["status"]
        ep      = _state["episode"]
        err     = _state["error_msg"]

    color, label = BADGE_MAP.get(st, ("secondary", st))
    badge = dbc.Badge(label, color=color, pill=True, style={"fontSize": "12px"})
    episode_txt = f"Episode {ep:,}" if ep > 0 else ""
    if st == "error" and err:
        episode_txt = err[:60]
    return badge, episode_txt


# ── Monitor charts ───────────────────────────────────────────────────────────

_EMPTY_FIG = go.Figure(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=16, t=8, b=36),
    xaxis=dict(color="#8892a4", gridcolor="#2e3340", zeroline=False),
    yaxis=dict(color="#8892a4", gridcolor="#2e3340", zeroline=False),
    font=dict(color="#8892a4", size=11),
))


def _base_layout():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=16, t=8, b=36),
        xaxis=dict(color=C["muted"], gridcolor=C["border"], zeroline=False),
        yaxis=dict(color=C["muted"], gridcolor=C["border"], zeroline=False),
        font=dict(color=C["text"], size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=C["panel"],
            bordercolor=C["border"],
            font=dict(color=C["text"], size=11),
        ),
    )


@app.callback(
    Output("chart-reward",    "figure"),
    Output("chart-heuristic", "figure"),
    Output("chart-probs",     "figure"),
    Output("chart-losses",    "figure"),
    Output("chart-lr",        "figure"),
    Input("interval", "n_intervals"),
    State("main-tabs", "value"),
)
def update_charts(_, tab):
    if tab != "monitor":
        raise PreventUpdate

    with _lock:
        rows      = list(_state["exp_rows"])
        loss_data = {k: list(v) for k, v in _state["loss_data"].items()}
        lr_data   = list(_state["lr_data"])
        models    = list(cfg.REWARD_MODELS)
        status    = _state["status"]
        err_msg   = _state["error_msg"]

    if not rows:
        # Show a waiting annotation while initializing/training
        if status in ("training", "initializing"):
            label = "Initializing models…" if status == "initializing" else "Waiting for first episode…"
            def _waiting_fig(title=""):
                f = go.Figure(layout=_base_layout())
                f.add_annotation(text=label, xref="paper", yref="paper",
                                 x=0.5, y=0.5, showarrow=False,
                                 font=dict(color=C["muted"], size=13))
                return f
            return _waiting_fig(), _waiting_fig(), _waiting_fig(), _waiting_fig(), _waiting_fig()
        if status == "error":
            def _err_fig():
                f = go.Figure(layout=_base_layout())
                f.add_annotation(text=f"Error: {err_msg[:60]}", xref="paper", yref="paper",
                                 x=0.5, y=0.5, showarrow=False,
                                 font=dict(color=C["danger"], size=12))
                return f
            return _err_fig(), _err_fig(), _err_fig(), _err_fig(), _err_fig()
        return [_EMPTY_FIG] * 5

    try:
        df = pd.DataFrame(rows)

        # ── Reward
        rewards = pd.to_numeric(df["Cumulative-Reward"], errors="coerce").values
        fig_rew = go.Figure(layout=_base_layout())
        fig_rew.add_trace(go.Scatter(
            y=_smooth(rewards), mode="lines",
            line=dict(color=C["accent"], width=2), name="Reward",
        ))
        fig_rew.add_trace(go.Scatter(
            y=rewards, mode="lines",
            line=dict(color=C["accent"], width=1, dash="dot"), opacity=0.3, name="Raw",
        ))

        # ── Heuristic
        heur = pd.to_numeric(df["Heuristic_T"], errors="coerce").values
        fig_heu = go.Figure(layout=_base_layout())
        fig_heu.add_trace(go.Scatter(
            y=_smooth(heur), mode="lines",
            line=dict(color=C["success"], width=2), name="Heuristic",
        ))

        # ── Probs per model
        fig_prob = go.Figure(layout=_base_layout())
        for m in models:
            col = f"{m}-Prob_T"
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").values
            fig_prob.add_trace(go.Scatter(
                y=_smooth(vals), mode="lines", name=m,
                line=dict(color=MODEL_COLORS.get(m, "#aaa"), width=2),
            ))

        # ── Losses
        fig_loss = go.Figure(layout=_base_layout())
        loss_colors = {"actor1_loss": C["accent"], "actor2_loss": C["warning"],
                       "critic_loss": C["danger"]}
        loss_names  = {"actor1_loss": "Actor1", "actor2_loss": "Actor2", "critic_loss": "Critic"}
        has_loss = False
        for key, color in loss_colors.items():
            vals = loss_data.get(key, [])
            if vals:
                has_loss = True
                fig_loss.add_trace(go.Scatter(
                    y=_smooth(vals, sigma=5), mode="lines",
                    name=loss_names[key], line=dict(color=color, width=2),
                ))
        if not has_loss:
            buf_need = cfg.BUFFER_SIZE
            buf_have = len(rows)
            fig_loss.add_annotation(
                text=f"Accumulating buffer… ({buf_have}/{buf_need} steps)",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(color=C["muted"], size=12))

        # ── LR
        fig_lr = go.Figure(layout=_base_layout())
        if lr_data:
            fig_lr.add_trace(go.Scatter(
                y=lr_data, mode="lines",
                line=dict(color=C["success"], width=2), name="LR",
            ))
        else:
            fig_lr.add_annotation(
                text="Available after first learning step",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(color=C["muted"], size=12))

        return fig_rew, fig_heu, fig_prob, fig_loss, fig_lr

    except Exception as exc:
        def _exc_fig():
            f = go.Figure(layout=_base_layout())
            f.add_annotation(text=f"Chart error: {str(exc)[:80]}", xref="paper", yref="paper",
                             x=0.5, y=0.5, showarrow=False,
                             font=dict(color=C["danger"], size=11))
            return f
        return _exc_fig(), _exc_fig(), _exc_fig(), _exc_fig(), _exc_fig()


# ── Candidates table ─────────────────────────────────────────────────────────

@app.callback(
    Output("cand-table-wrapper", "children"),
    Output("cand-count",         "children"),
    Input("interval",     "n_intervals"),
    Input("cand-sort-col","value"),
    Input("cand-filter",  "value"),
    State("main-tabs",    "value"),
)
def update_candidates(_, sort_col, flt, tab):
    if tab != "candidates":
        raise PreventUpdate

    with _lock:
        rows   = list(_state["exp_rows"])
        models = list(cfg.REWARD_MODELS)

    if not rows:
        return html.Div("No data yet — start training.",
                        style={"color": C["muted"], "textAlign": "center",
                               "marginTop": "40px"}), ""

    df = pd.DataFrame(rows)
    # Deduplicate and keep best by sort column
    sc = sort_col if sort_col in df.columns else "Cumulative-Reward"
    df[sc] = pd.to_numeric(df[sc], errors="coerce")
    df = df.sort_values(sc, ascending=False).drop_duplicates("Peptide_T").reset_index(drop=True)

    if flt:
        df = df[df["Peptide_T"].str.contains(flt.upper(), na=False)]

    count_txt = f"{len(df):,} unique candidates"

    keep_cols = ["Peptide_T", "Cumulative-Reward", "Heuristic_T"] + \
                [f"{m}-Prob_T" for m in models if f"{m}-Prob_T" in df.columns]
    df = df[keep_cols].head(500)

    col_defs = [
        {"name": "Peptide", "id": "Peptide_T",
         "type": "text", "presentation": "markdown"},
        {"name": "Reward",  "id": "Cumulative-Reward", "type": "numeric"},
        {"name": "Heuristic", "id": "Heuristic_T",     "type": "numeric"},
    ] + [{"name": m, "id": f"{m}-Prob_T", "type": "numeric"} for m in models
         if f"{m}-Prob_T" in df.columns]

    tbl = dash_table.DataTable(
        data=df.to_dict("records"),
        columns=col_defs,
        sort_action="native",
        filter_action="native",
        page_action="native",
        page_size=30,
        style_table={"overflowX": "auto"},
        style_cell={
            "background": C["panel"], "color": C["text"],
            "border": f"1px solid {C['border']}",
            "textAlign": "left", "padding": "6px 10px",
            "fontFamily": "monospace", "fontSize": "12px",
        },
        style_header={
            "background": C["bg"], "color": C["accent"],
            "fontWeight": "700", "border": f"1px solid {C['border']}",
            "fontSize": "11px", "textTransform": "uppercase",
        },
        style_data_conditional=[
            {"if": {"filter_query": "{HEM-Prob_T} > 0.5"},
             "color": MODEL_COLORS["HEM"]},
            {"if": {"row_index": "odd"},
             "backgroundColor": "#1e2228"},
        ],
    )
    return tbl, count_txt


@app.callback(
    Output("download-csv", "data"),
    Input("btn-export", "n_clicks"),
    prevent_initial_call=True,
)
def export_csv(_):
    with _lock:
        rows = list(_state["exp_rows"])
    if not rows:
        return no_update
    df = pd.DataFrame(rows)
    df = df.drop_duplicates("Peptide_T").sort_values("Cumulative-Reward", ascending=False)
    return dcc.send_data_frame(df.to_csv, "rl4axp_candidates.csv", index=False)


# ── SQA ──────────────────────────────────────────────────────────────────────

@app.callback(
    Output("sqa-status-msg",  "children"),
    Output("sqa-results-area","children"),
    Input("btn-sqa",   "n_clicks"),
    Input("interval",  "n_intervals"),
    State("sqa-topn",   "value"),
    State("sqa-npos",   "value"),
    State("sqa-naas",   "value"),
    State("sqa-trotter","value"),
    State("sqa-steps",  "value"),
    State("main-tabs",  "value"),
    prevent_initial_call=False,
)
def sqa_panel(n_clicks, _, topn, npos, naas, trotter, steps, tab):
    triggered = ctx.triggered_id

    if triggered == "btn-sqa":
        with _lock:
            st = _state["status"]
        if _framework is None or st not in ("training", "paused", "done"):
            return "Start training first to populate candidates.", no_update

        # Apply SQA config overrides
        cfg.SQA_N_POSITIONS = int(npos or cfg.SQA_N_POSITIONS)
        cfg.SQA_N_AAS       = int(naas or cfg.SQA_N_AAS)
        cfg.SQA_N_TROTTER   = int(trotter or cfg.SQA_N_TROTTER)
        cfg.SQA_N_STEPS     = int(steps or cfg.SQA_N_STEPS)

        t = threading.Thread(
            target=_sqa_thread, args=(int(topn or 10),), daemon=True
        )
        t.start()
        return "Running SQA…", no_update

    # Interval: refresh results
    if tab != "sqa":
        raise PreventUpdate

    with _lock:
        sqa_st = _state["sqa_status"]
        sqa_err = _state["sqa_error"]
        rows = list(_state["sqa_rows"])

    if sqa_st == "running":
        return "⏳ Running SQA refinement…", no_update
    if sqa_st == "error":
        return f"Error: {sqa_err}", no_update
    if sqa_st == "idle":
        return "Configure and click Run SQA.", no_update

    if not rows:
        return "SQA done — no results.", html.Div()

    df = pd.DataFrame(rows)
    models_present = [m for m in ALL_MODELS if f"{m}_Delta" in df.columns]

    # Delta bar chart
    fig = go.Figure(layout=_base_layout())
    fig.update_layout(height=220, margin=dict(l=40, r=16, t=16, b=36))
    for m in models_present:
        vals = df[f"{m}_Delta"].astype(float).values
        fig.add_trace(go.Bar(
            name=m, y=df["Original_Seq"].str[:10] + "…",
            x=vals, orientation="h",
            marker_color=MODEL_COLORS.get(m, "#aaa"),
        ))
    fig.update_layout(barmode="group", xaxis_title="Δ Score (refined − original)")

    # Table
    show_cols = (
        ["Original_Seq", "Refined_Seq", "Mutations", "N_Mutations",
         "Heuristic_Orig", "Heuristic_Ref", "Heuristic_Delta"] +
        [c for m in models_present for c in [f"{m}_Orig", f"{m}_Ref", f"{m}_Delta"]]
    )
    show_cols = [c for c in show_cols if c in df.columns]

    tbl = dash_table.DataTable(
        data=df[show_cols].to_dict("records"),
        columns=[{"name": c.replace("_", " "), "id": c} for c in show_cols],
        sort_action="native",
        page_size=10,
        style_table={"overflowX": "auto"},
        style_cell={
            "background": C["panel"], "color": C["text"],
            "border": f"1px solid {C['border']}",
            "padding": "5px 8px", "fontSize": "11px", "fontFamily": "monospace",
        },
        style_header={
            "background": C["bg"], "color": C["accent"],
            "fontWeight": "700", "border": f"1px solid {C['border']}",
            "fontSize": "10px",
        },
    )

    return f"Done — {len(df)} sequences refined.", html.Div([
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        html.Div(style={"marginTop": "12px"}),
        tbl,
    ])


# ── Scorer ────────────────────────────────────────────────────────────────────

@app.callback(
    Output("scorer-results", "children"),
    Output("scorer-error",   "children"),
    Input("btn-score", "n_clicks"),
    State("scorer-seq",    "value"),
    State("scorer-conc",   "value"),
    State("scorer-models", "value"),
    prevent_initial_call=True,
)
def score_sequence(_, seq, conc, selected_models):
    if not seq:
        return no_update, "Enter a sequence."

    aa_valid = set("ACDEFGHIKLMNPQRSTVWY")
    seq = seq.strip().upper()
    if not seq or not all(c in aa_valid for c in seq):
        return no_update, "Invalid amino acid characters."
    if len(seq) > 49:
        return no_update, f"Sequence too long ({len(seq)} > 49 aa)."

    from peptide_optimization.environment import _PROB_FNS, _heuristic_reward_single

    results = {}
    errors  = []
    for m in (selected_models or []):
        try:
            if m == "HEM":
                from hem_prediction.inference import get_hem_probs
                val = float(get_hem_probs([seq], [float(conc or 50)])[0])
            else:
                val = float(_PROB_FNS[m]([seq])[0])
            results[m] = val
        except Exception as exc:
            errors.append(f"{m}: {exc}")

    heur = _heuristic_reward_single(seq)

    gauges = []
    for m, val in results.items():
        direction = "↑ higher is better" if m != "HEM" else "↓ lower is better"
        color = MODEL_COLORS.get(m, C["accent"])
        if m == "HEM":
            bar_color = C["danger"] if val > 0.5 else C["success"]
        else:
            bar_color = C["success"] if val > 0.6 else (C["warning"] if val > 0.3 else C["danger"])

        gauge = html.Div([
            html.Div([
                html.Span(m, style={"fontWeight": "700", "color": color,
                                    "fontSize": "13px", "minWidth": "40px"}),
                html.Span(f"{val:.4f}", style={"fontWeight": "700", "color": C["text"],
                                               "fontSize": "15px", "marginLeft": "auto"}),
            ], style={"display": "flex", "alignItems": "center",
                      "justifyContent": "space-between", "marginBottom": "4px"}),
            html.Div([  # progress bar
                html.Div(style={
                    "width": f"{val*100:.1f}%", "height": "6px",
                    "background": bar_color, "borderRadius": "3px",
                    "transition": "width 0.4s ease",
                }),
            ], style={"background": C["border"], "borderRadius": "3px",
                      "marginBottom": "3px"}),
            html.Div(direction, style={"fontSize": "10px", "color": C["muted"]}),
        ], style={"marginBottom": "14px"})
        gauges.append(gauge)

    content = html.Div([
        html.Div([
            html.Code(seq, style={"fontSize": "13px", "color": C["accent"],
                                  "letterSpacing": "0.05em"}),
            html.Span(f"  {len(seq)} aa", style={"fontSize": "11px",
                                                  "color": C["muted"], "marginLeft": "8px"}),
        ], style={"marginBottom": "16px", "paddingBottom": "12px",
                  "borderBottom": f"1px solid {C['border']}"}),
        html.Div([
            html.Div("Heuristic score", style={"fontSize": "11px", "color": C["muted"]}),
            html.Div(f"{heur:+.4f}", style={"fontSize": "18px", "fontWeight": "700",
                                             "color": C["success"] if heur > 0 else C["danger"]}),
        ], style={"marginBottom": "16px"}),
        *gauges,
        html.Div([html.Div(e, style={"color": C["danger"], "fontSize": "11px"})
                  for e in errors]) if errors else html.Div(),
    ])
    return content, ""


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    print(f"\n  RL4AXP Dashboard → http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
